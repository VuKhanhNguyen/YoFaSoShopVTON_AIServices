"""Factory phục vụ runtime và test: không yêu cầu khóa thật lúc import test."""
import logging
import asyncio
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, File, Form, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException

from api.core.config import Settings
from api.core.body_limit import BodyLimitMiddleware, BodyTooLarge
from api.core.errors import ServiceError
from api.core.store import Store
from api.schemas import AssetResponse, ErrorResponse, JobResponse, PersonRequest, ProductRequest, RouteRequest, Status, TryOnRequest
from api.services.assets import Assets
from api.services.routing import DRESSES, LOWER, UPPER, route
from api.services.vision import Vision

LOG = logging.getLogger("yofaso.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Lắp API, dependencies và lifespan; không import/load model diffusion ở đây."""
    settings = settings or Settings()
    store = Store(settings)
    assets = Assets(settings, store)
    vision = Vision(settings)
    rate_buckets = defaultdict(deque)
    rate_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        """Chỉ embedded worker mock để test; production chạy process worker riêng."""
        deadline = time.monotonic() + 60
        while True:
            try:
                await asyncio.to_thread(store.check_schema)
                break
            except ServiceError:
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(1)
        worker = thread = None
        if settings.embedded_worker:
            from workers.runner import Worker
            worker = Worker(settings)
            thread = threading.Thread(target=worker.serve, daemon=True)
            thread.start()
        yield
        if worker:
            worker.stop.set()
            thread.join(timeout=10)

    app = FastAPI(title="YoFaSoShop Internal AI API", version="1.0.0", lifespan=lifespan,
                  description="API AI Services cho VTON, không chứa nghiệp vụ đơn hàng/điểm thẩm mỹ.",
                  docs_url="/docs" if settings.expose_docs else None, redoc_url=None,
                  openapi_url="/openapi.json" if settings.expose_docs else None)
    app.state.settings, app.state.store, app.state.assets = settings, store, assets
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_bytes + 65536)
    key_header = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)

    def error_response(request, code, message, status, details=None, retryable=False):
        """Một dạng lỗi cho domain, validation, HTTP và unexpected exception."""
        return JSONResponse(status_code=status, content={"errorCode": code, "message": message, "details": details or {},
                            "retryable": retryable, "correlationId": getattr(request.state, "correlation_id", "unknown")},
                            headers={"Retry-After": "5"} if status == 429 else None)

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        """Đổi domain error sang HTTP, không lộ exception nội bộ."""
        return error_response(request, exc.code, exc.message, exc.status, exc.details, exc.retryable)

    @app.exception_handler(BodyTooLarge)
    async def body_too_large(request, exc):
        """Lỗi wire body không được biến thành 400/500 do parser multipart."""
        return error_response(request, "UPLOAD_TOO_LARGE", "Request vượt dung lượng cho phép.", 413)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        """Không echo input thô vì có thể chứa thông tin nhạy cảm."""
        details = {"fields": [{"location": list(e["loc"]), "type": e["type"], "message": e["msg"]} for e in exc.errors()]}
        return error_response(request, "REQUEST_INVALID", "Tham số không hợp lệ; kiểm tra fields.", 422, details)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        """Giữ 404/405 và lỗi form của framework cùng dạng response."""
        return error_response(request, f"HTTP_{exc.status_code}", "Không tìm thấy tài nguyên hoặc phương thức/tham số HTTP không hợp lệ.", exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        """Log correlation ID để tìm lỗi; response không có traceback/key/path."""
        LOG.exception("request_failed correlationId=%s", getattr(request.state, "correlation_id", "unknown"))
        return error_response(request, "INTERNAL_ERROR", "Lỗi nội bộ; kiểm tra log theo correlationId.", 500)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Correlation, kích thước request và rate limit theo dịch vụ trên một process API."""
        provided = request.headers.get("X-Correlation-ID", "")
        request.state.correlation_id = provided if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", provided) else uuid4().hex
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > settings.max_upload_bytes + 65536):
            response = error_response(request, "UPLOAD_TOO_LARGE", "Request vượt dung lượng cho phép.", 413)
        else:
            incoming = request.headers.get("X-Internal-API-Key", "")
            valid_key = secrets.compare_digest(incoming.encode(), settings.ai_internal_api_key.get_secret_value().encode())
            remaining, reset = settings.rate_limit_per_minute, int(time.time()) + 60
            limited = False
            if request.url.path.startswith("/internal/") and valid_key:
                # Một khóa service dùng một bucket, tránh spoof X-Forwarded-For để lách limit.
                with rate_lock:
                    bucket = rate_buckets["internal"]
                    timestamp = time.time()
                    while bucket and bucket[0] <= timestamp - 60:
                        bucket.popleft()
                    limited = len(bucket) >= settings.rate_limit_per_minute
                    if not limited:
                        bucket.append(timestamp)
                    remaining = max(0, settings.rate_limit_per_minute - len(bucket))
                    reset = int(bucket[0] + 60) + 1 if bucket else int(timestamp + 60)
            # Từ chối trước multipart parser: request thiếu key không được spool file lên disk.
            if request.url.path.startswith("/internal/") and not valid_key:
                response = error_response(request, "UNAUTHORIZED", "Khóa kết nối AI không hợp lệ.", 401)
            elif limited:
                response = error_response(request, "RATE_LIMITED", "Đã vượt số request/phút; vui lòng đợi.", 429, retryable=True)
            else:
                response = await call_next(request)
            response.headers.update({"X-RateLimit-Limit": str(settings.rate_limit_per_minute), "X-RateLimit-Remaining": str(remaining), "X-RateLimit-Reset": str(reset)})
            if limited:
                response.headers["Retry-After"] = str(max(1, reset - int(time.time())))
        response.headers["X-Correlation-ID"] = request.state.correlation_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def require_key(key: str | None = Depends(key_header)):
        """Xác thực service-to-service; BE vẫn phải kiểm tra quyền khách xem session."""
        if not key or not secrets.compare_digest(key.encode(), settings.ai_internal_api_key.get_secret_value().encode()):
            raise ServiceError("UNAUTHORIZED", "Khóa kết nối AI không hợp lệ.", 401)

    error_models = {s: {"model": ErrorResponse} for s in (400, 401, 404, 409, 413, 415, 422, 429, 500, 503)}
    router = APIRouter(prefix="/internal/v1", dependencies=[Depends(require_key)], responses=error_models)

    @app.get("/health/live", tags=["Health"])
    def live():
        """Chỉ báo API còn sống; không khẳng định GPU sẵn sàng."""
        return {"status": "ok", "service": "ai-api", "mode": settings.ai_mode}

    @app.get("/health/ready", tags=["Health"], responses={503: {"model": ErrorResponse}})
    def ready():
        """Queue storage + heartbeat + file checkpoint; chưa thay inference nghiệm thu."""
        heartbeat = store.heartbeat()
        worker_ready = bool(heartbeat and heartbeat["heartbeat"] is not None and heartbeat["mode"] == settings.ai_mode and time.time() - heartbeat["heartbeat"] < settings.worker_lease_seconds)
        files_ready = settings.ai_mode == "mock" or all(p.is_file() for p in (settings.deepfashion_weights, settings.pose_weights, settings.ootd_root / "checkpoints/ootd/model_index.json"))
        if not worker_ready or not files_ready:
            raise ServiceError("SERVICE_NOT_READY", "Worker chưa sẵn sàng hoặc thiếu checkpoint.", 503)
        return {"status": "ready", "simulated": settings.ai_mode == "mock", "inferenceVerified": False}

    @router.get("/config-check", tags=["System"])
    def config_check():
        """Kiểm tra cấu hình không lộ secret hoặc absolute path."""
        return {"env": settings.app_env, "mode": settings.ai_mode, "simulated": settings.ai_mode == "mock", "queueBackend": "file-manifest", "vtonReady": False}

    @router.get("/capabilities", tags=["System"])
    def capabilities():
        """Liệt kê phạm vi AI, category và giới hạn để BE xây form phù hợp."""
        return {"apiVersion": "1.0.0", "mode": settings.ai_mode, "simulated": settings.ai_mode == "mock",
                "categories": ["upper_body", "lower_body", "dress"], "subtypes": sorted(UPPER | LOWER | DRESSES),
                "queueBackend": "file-manifest", "scoringOwner": "Spring Boot", "maxUploadBytes": settings.max_upload_bytes,
                "maxVariants": settings.max_variants, "maxSamples": 4}

    @router.get("/models", tags=["System"])
    def models():
        """Kiểm tra file local, không tải model và không tuyên bố inference thành công."""
        return {"mode": settings.ai_mode, "models": [{"name": name, "present": path.is_file()} for name, path in (
            ("DeepFashion2", settings.deepfashion_weights), ("YOLO Pose", settings.pose_weights),
            ("OOTDiffusion", settings.ootd_root / "checkpoints/ootd/model_index.json"))], "inferenceVerified": False}

    @router.post("/assets", response_model=AssetResponse, status_code=201, tags=["Assets"])
    async def upload_asset(file: UploadFile = File(...), role: Literal["person", "garment", "reference"] = Form(...)):
        """Upload tên file tùy ý; role chỉ là khai báo, model sẽ xác minh nội dung ở pipeline."""
        try:
            data = await file.read(settings.max_upload_bytes + 1)
            return assets.upload(data, role)
        finally:
            await file.close()

    @router.get("/assets/{asset_id}", response_model=AssetResponse, tags=["Assets"])
    def asset_info(asset_id: str):
        """Metadata ảnh; không trả bytes hay đường dẫn server."""
        return store.public_asset(store.asset(asset_id))

    @router.get("/assets/{asset_id}/content", tags=["Assets"], response_class=FileResponse, responses={200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}}})
    def asset_content(asset_id: str):
        """Tải ảnh private bằng API key; URL này không dành gọi thẳng từ FE."""
        path = assets.path(asset_id)
        if not path.is_file():
            raise ServiceError("ASSET_UNAVAILABLE", "File ảnh không còn trên storage.", 503)
        return FileResponse(path, media_type="image/png", filename=f"{asset_id}.png")

    @router.delete("/assets/{asset_id}", status_code=204, tags=["Assets"])
    def delete_asset(asset_id: str):
        """Xóa ảnh chưa được job tham chiếu; không xóa kết quả lịch sử đang dùng."""
        assets.delete(asset_id)
        return Response(status_code=204)

    @router.post("/persons/validate", tags=["Vision"])
    def validate_person(body: PersonRequest):
        """Kiểm tra nội dung ảnh người; có thể kiểm tra thêm frame/category trước submit."""
        image = assets.open(body.asset_id, {"person", "reference"})
        result = vision.validate_person(image)
        if body.vton_category:
            result["route"] = route(body.vton_category, result["personFrame"], body.garment_subtype, body.garment_length)
            result["accepted"] = result["accepted"] and result["route"]["accepted"]
        return result

    @router.post("/routing/preview", tags=["Vision"])
    def routing_preview(body: RouteRequest):
        """Rule thuần, không tốn GPU; frame nhập tay không thay validation ảnh thật."""
        return route(body.vton_category, body.person_frame, body.garment_subtype, body.garment_length)

    def validate_assets(asset_ids, roles):
        """Chặn role sai/ID lặp trước khi tạo job trong queue."""
        if len(set(asset_ids)) != len(asset_ids):
            raise ServiceError("DUPLICATE_ASSET", "Danh sách ảnh có ID bị lặp.")
        for aid in asset_ids:
            if store.asset(aid)["role"] not in roles:
                raise ServiceError("ASSET_ROLE_MISMATCH", "Một ảnh không đúng vai trò yêu cầu.")

    @router.post("/products/analyze", response_model=JobResponse, status_code=202, tags=["Jobs"])
    def analyze_product(body: ProductRequest):
        """Enqueue vào manifest AI; không sửa bảng catalog hay tự công bố sản phẩm."""
        if len(body.product_asset_ids) > settings.max_variants:
            raise ServiceError("TOO_MANY_VARIANTS", "Vượt số ảnh biến thể cho phép.")
        validate_assets(body.product_asset_ids, {"garment"})
        validate_assets(body.reference_asset_ids, {"reference"})
        row = store.submit("PRODUCT_ANALYSIS", body.analysis_id, body.model_dump(by_alias=True), body.product_asset_ids + body.reference_asset_ids, settings.ai_mode, settings.max_queue_size)
        return store.public_job(row)

    @router.post("/try-on/jobs", response_model=JobResponse, status_code=202, tags=["Jobs"])
    def create_tryon(body: TryOnRequest):
        """Tạo VTON idempotent theo sessionId; validation ảnh chạy trước diffusion trong worker."""
        validate_assets([body.person_asset_id], {"person"})
        validate_assets([body.garment_asset_id], {"garment"})
        route(body.vton_category, "full_body", body.garment_subtype, body.garment_length)
        return store.public_job(store.submit("TRY_ON", body.session_id, body.model_dump(by_alias=True), [body.person_asset_id, body.garment_asset_id], settings.ai_mode, settings.max_queue_size))

    @router.get("/jobs", tags=["Jobs"])
    def list_jobs(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), status: Status | None = None):
        """Phân trang job cho service quản trị; không chứa ảnh/input nhạy cảm."""
        return store.list_jobs(limit, offset, status.value if status else None)

    @router.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
    def get_job(job_id: str):
        """Poll một job phân tích sản phẩm hoặc VTON."""
        return store.public_job(store.job(job_id))

    @router.get("/try-on/jobs/{session_id}", response_model=JobResponse, tags=["Jobs"])
    def get_session(session_id: str):
        """BE tra bằng sessionId khi chưa nhận response submit do mất mạng."""
        return store.public_job(store.find_job("TRY_ON", session_id))

    @router.post("/jobs/{job_id}/cancel", response_model=JobResponse, tags=["Jobs"])
    def cancel_job(job_id: str):
        """Hủy idempotent job queued; active GPU trả 409 để tránh giả vờ đã dừng."""
        return store.public_job(store.transition(job_id, "cancel", settings.max_queue_size, settings.max_job_attempts))

    @router.post("/jobs/{job_id}/retry", response_model=JobResponse, status_code=202, tags=["Jobs"])
    def retry_job(job_id: str):
        """Thử lại job lỗi tạm theo policy, không tạo session mới hoặc chạy vô hạn."""
        return store.public_job(store.transition(job_id, "retry", settings.max_queue_size, settings.max_job_attempts))

    app.include_router(router)
    return app
