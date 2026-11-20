"""Cloud VTON Provider Adapters (FASHN v1.6, Genlook, Kling Image O1).

Mọi provider đều nhận cùng một hợp đồng đầu vào chuẩn hóa và trả về (PIL.Image, metadata),
nhờ đó đổi model chỉ cần thay VTON_PROVIDER trong .env mà không đụng pipeline hay Backend.
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx
from PIL import Image, UnidentifiedImageError

from api.core.errors import ServiceError

LOGGER = logging.getLogger("yofaso.vton_provider")

FASHN_CATEGORY_MAP = {
    "upper_body": "tops",
    "lower_body": "bottoms",
    "dress": "one-pieces",
}


def resolve_fashn_category(vton_category: str) -> str:
    """Ánh xạ category catalog của YoFaSoShop sang enum của FASHN v1.6."""
    mapped = FASHN_CATEGORY_MAP.get(vton_category)
    if not mapped:
        raise ServiceError("GARMENT_CATEGORY_UNKNOWN", f"Category không hỗ trợ cho FASHN: {vton_category}")
    return mapped


def resolve_garment_photo_type(configured: str, *, has_cutout: bool) -> str:
    """Chọn garment_photo_type tối ưu: cutout sạch nền luôn dùng flat-lay khi cấu hình auto."""
    if configured in ("flat-lay", "model"):
        return configured
    return "flat-lay" if has_cutout else "auto"


def _load_fal_client(settings):
    """Kiểm tra FAL_KEY và import fal_client lười để API mock không phụ thuộc mạng."""
    if not settings.has_fal_key():
        raise ServiceError(
            "VTON_API_KEY_MISSING",
            "Chưa cấu hình FAL_KEY hợp lệ trong .env để chạy VTON.",
            status=503,
            retryable=False,
        )
    try:
        import fal_client
    except ImportError as exc:
        raise ServiceError(
            "MODEL_DEPENDENCY_MISSING",
            "Chưa cài fal-client; chạy pip install -r requirements-api.txt.",
            status=503,
            retryable=False,
        ) from exc
    os.environ["FAL_KEY"] = settings.fal_key_value()
    return fal_client


def _download_result_image(url: str, *, max_bytes: int, timeout: float = 60.0) -> Image.Image:
    """Tải ảnh kết quả từ CDN provider qua HTTPS có giới hạn dung lượng, trả về RGB sạch."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ServiceError("VTON_RESULT_INVALID", "URL ảnh kết quả từ provider không hợp lệ.", 502)

    chunks = bytearray()
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise ServiceError(
                        "VTON_PROVIDER_UNAVAILABLE",
                        f"Không tải được ảnh kết quả VTON (HTTP {response.status_code}).",
                        status=503,
                        retryable=True,
                    )
                for chunk in response.iter_bytes():
                    if len(chunks) + len(chunk) > max_bytes:
                        raise ServiceError("VTON_RESULT_TOO_LARGE", "Ảnh kết quả VTON vượt dung lượng cho phép.", 502)
                    chunks.extend(chunk)
    except ServiceError:
        raise
    except httpx.HTTPError as exc:
        LOGGER.warning("Download VTON result failed: %s", exc)
        raise ServiceError("VTON_PROVIDER_UNAVAILABLE", "Lỗi mạng khi tải ảnh kết quả VTON.", 503, retryable=True) from exc

    try:
        with Image.open(io.BytesIO(chunks)) as img:
            rgb = img.convert("RGB")
            rgb.load()
            if min(rgb.size) < 64:
                raise ServiceError("VTON_RESULT_INVALID", "Ảnh kết quả VTON có kích thước bất thường.", 502)
            return rgb
    except ServiceError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ServiceError("VTON_RESULT_INVALID", "Dữ liệu ảnh trả về từ provider không hợp lệ.", 502) from exc


def _extract_image_url(result: dict | None) -> str:
    """Hỗ trợ cả output schema `images: [{url}]` (FASHN/Kling O1) lẫn `image: {url}`."""
    if not isinstance(result, dict):
        raise ServiceError("VTON_GENERATION_REJECTED", "Provider không trả về ảnh kết quả.", 502, retryable=True)
    images = result.get("images")
    if isinstance(images, list) and images and isinstance(images[0], dict) and images[0].get("url"):
        return str(images[0]["url"])
    image = result.get("image")
    if isinstance(image, dict) and image.get("url"):
        return str(image["url"])
    raise ServiceError(
        "VTON_GENERATION_REJECTED",
        "Không tạo được ảnh thử đồ (có thể bị từ chối kiểm duyệt nội dung); hãy thử ảnh khác.",
        status=422,
        retryable=False,
    )


def _translate_fal_error(exc: Exception, provider_name: str) -> ServiceError:
    """Đổi exception từ fal-client sang ServiceError an toàn, không lộ FAL_KEY."""
    text = str(exc).lower()
    LOGGER.warning("VTON provider %s failed: %s", provider_name, exc)
    if any(tok in text for tok in ("401", "403", "unauthorized", "invalid api key", "exhausted", "balance", "credit")):
        return ServiceError(
            "VTON_API_AUTH_FAILED",
            "FAL_KEY không hợp lệ hoặc tài khoản fal.ai hết credit.",
            status=503,
            retryable=False,
        )
    if any(tok in text for tok in ("moderation", "nsfw", "safety", "content policy", "explicit", "422")):
        return ServiceError(
            "VTON_GENERATION_REJECTED",
            "Ảnh không đạt kiểm duyệt hoặc không nhận diện được trang phục để thử đồ.",
            status=422,
            retryable=False,
        )
    return ServiceError(
        "VTON_PROVIDER_UNAVAILABLE",
        "Dịch vụ thử đồ AI tạm gián đoạn; vui lòng thử lại sau.",
        status=503,
        retryable=True,
    )


FASHN_EXPECTED_SECONDS = {
    "performance": 6.0,
    "balanced": 10.5,
    "quality": 16.0,
}


@contextmanager
def _cloud_render_ticker(
    report: Callable[[int, str, str], None],
    *,
    expected_seconds: float,
    stage: str,
    message: str,
    start_pct: int = 42,
    cap_pct: int = 80,
    interval_seconds: float = 0.8,
):
    """Báo tiến trình tiệm cận theo tốc độ thực của FASHN/Kling; dừng sạch trước khi thoát."""
    stop = threading.Event()
    started_at = time.monotonic()
    tau = max(2.0, expected_seconds * 0.52)

    def _run() -> None:
        while not stop.wait(interval_seconds):
            elapsed = time.monotonic() - started_at
            frac = 1.0 - math.exp(-elapsed / tau)
            pct = min(cap_pct, start_pct + int((cap_pct - start_pct) * frac))
            report(pct, stage, message)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=2.0)


def render_fashn_v16(
    person_path: Path,
    garment_path: Path,
    vton_category: str,
    metadata: dict,
    settings,
    progress_cb: Callable[[int, str, str], None] | None = None,
) -> tuple[Image.Image, dict]:
    """Gọi fal-ai/fashn/tryon/v1.6 với tham số tối ưu cho shop thời trang."""
    fal_client = _load_fal_client(settings)
    category = resolve_fashn_category(vton_category)
    photo_type = resolve_garment_photo_type(
        settings.fashn_garment_photo_type,
        has_cutout=bool(metadata.get("has_cutout", False)),
    )

    progress_lock = threading.Lock()
    last_pct = 0

    def report(pct: int, stage: str, msg: str) -> None:
        nonlocal last_pct
        if not progress_cb:
            return
        with progress_lock:
            if pct < last_pct:
                return
            last_pct = pct
            progress_cb(pct, stage, msg)

    report(28, "UPLOADING", "Đang tải ảnh lên FASHN v1.6...")
    try:
        model_url = fal_client.upload_file(str(person_path))
        garment_url = fal_client.upload_file(str(garment_path))

        def on_queue_update(update) -> None:
            queued = getattr(fal_client, "Queued", None)
            in_progress = getattr(fal_client, "InProgress", None)
            if queued is not None and isinstance(update, queued):
                report(40, "VTON_RENDER", "Đang chờ slot xử lý trên FASHN v1.6...")
            elif in_progress is not None and isinstance(update, in_progress):
                report(44, "VTON_RENDER", "FASHN v1.6 đang ướm trang phục lên cơ thể...")

        arguments = {
            "model_image": model_url,
            "garment_image": garment_url,
            "category": category,
            "mode": settings.fashn_mode,
            "garment_photo_type": photo_type,
            "moderation_level": settings.fashn_moderation_level,
            "num_samples": settings.fashn_num_samples,
            "segmentation_free": settings.fashn_segmentation_free,
            "output_format": "png",
            "enable_safety_checker": True,
            "seed": int(metadata.get("seed", 42)),
        }
        report(38, "VTON_RENDER", "Đang kết nối FASHN v1.6...")
        expected = FASHN_EXPECTED_SECONDS.get(settings.fashn_mode, 14.0)
        with _cloud_render_ticker(
            report,
            expected_seconds=expected,
            stage="VTON_RENDER",
            message="FASHN v1.6 đang ướm trang phục lên cơ thể...",
            start_pct=44,
            cap_pct=80,
        ):
            result = fal_client.subscribe(
                settings.fashn_endpoint,
                arguments=arguments,
                with_logs=True,
                on_queue_update=on_queue_update,
            )
    except ServiceError:
        raise
    except Exception as exc:
        raise _translate_fal_error(exc, "fashn_v16") from exc

    image_url = _extract_image_url(result)
    report(84, "DOWNLOADING", "Đang tải ảnh kết quả chất lượng cao...")
    image = _download_result_image(
        image_url,
        max_bytes=settings.max_upload_bytes,
        timeout=float(settings.vton_download_timeout_seconds),
    )
    return image, {
        "provider": "fashn_v16",
        "endpoint": settings.fashn_endpoint,
        "category": category,
        "mode": settings.fashn_mode,
        "garmentPhotoType": photo_type,
        "segmentationFree": settings.fashn_segmentation_free,
        "numSamples": settings.fashn_num_samples,
    }


def _build_kling_prompt(vton_category: str, metadata: dict) -> str:
    """Ghép prompt chuẩn cho Kling Multi-Reference (khi đổi VTON_PROVIDER=kling_o1)."""
    region = {
        "upper_body": "upper-body garment",
        "lower_body": "lower-body garment",
        "dress": "full-body dress or outfit",
    }.get(vton_category, "garment")
    details = [
        metadata.get("productName"),
        metadata.get("colorName"),
        metadata.get("material"),
    ]
    detail_str = ", ".join(str(x).strip() for x in details if x and str(x).strip())
    garment_desc = f"the {region} ({detail_str})" if detail_str else f"the {region}"
    return (
        f"Keep the exact person, face, hair, pose, body shape, hands, and background from @Image1. "
        f"Replace only the clothing with {garment_desc} from @Image2, preserving exact color, "
        f"fabric texture, pattern, prints, and natural drape. Photorealistic studio lighting and shadows."
    )


def render_kling_o1(
    person_path: Path,
    garment_path: Path,
    vton_category: str,
    metadata: dict,
    settings,
    progress_cb: Callable[[int, str, str], None] | None = None,
) -> tuple[Image.Image, dict]:
    """Adapter dự phòng cho fal-ai/kling-image/o1; kích hoạt bằng VTON_PROVIDER=kling_o1."""
    fal_client = _load_fal_client(settings)

    progress_lock = threading.Lock()
    last_pct = 0

    def report(pct: int, stage: str, msg: str) -> None:
        nonlocal last_pct
        if not progress_cb:
            return
        with progress_lock:
            if pct < last_pct:
                return
            last_pct = pct
            progress_cb(pct, stage, msg)

    report(28, "UPLOADING", "Đang gửi ảnh lên Kling Image O1...")
    try:
        human_url = fal_client.upload_file(str(person_path))
        garment_url = fal_client.upload_file(str(garment_path))
        report(40, "VTON_RENDER", "Đang xử lý thử đồ trên Kling Image O1...")
        with _cloud_render_ticker(
            report,
            expected_seconds=18.0,
            stage="VTON_RENDER",
            message="Kling Image O1 đang ướm trang phục lên cơ thể...",
            start_pct=44,
            cap_pct=80,
        ):
            result = fal_client.subscribe(
                settings.kling_endpoint,
                arguments={
                    "prompt": _build_kling_prompt(vton_category, metadata),
                    "image_urls": [human_url, garment_url],
                },
                with_logs=True,
            )
    except ServiceError:
        raise
    except Exception as exc:
        raise _translate_fal_error(exc, "kling_o1") from exc

    image_url = _extract_image_url(result)
    report(84, "DOWNLOADING", "Đang tải ảnh kết quả chất lượng cao...")
    image = _download_result_image(
        image_url,
        max_bytes=settings.max_upload_bytes,
        timeout=float(settings.vton_download_timeout_seconds),
    )
    return image, {
        "provider": "kling_o1",
        "endpoint": settings.kling_endpoint,
        "category": vton_category,
    }


_GENLOOK_MAX_UPLOAD_BYTES = 9_500_000  # Genlook giới hạn 10 MB/ảnh upload


def _prepare_genlook_upload(path: Path, filename_stem: str) -> tuple[str, bytes, str]:
    """Đọc ảnh local; tự nén JPEG chất lượng cao nếu ảnh > 9.5 MB để luôn vượt trần 10 MB của Genlook."""
    raw = path.read_bytes()
    if len(raw) <= _GENLOOK_MAX_UPLOAD_BYTES:
        if raw.startswith(b"\xff\xd8"):
            return f"{filename_stem}.jpg", raw, "image/jpeg"
        if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            return f"{filename_stem}.webp", raw, "image/webp"
        return f"{filename_stem}.png", raw, "image/png"
    with Image.open(io.BytesIO(raw)) as img:
        rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=92, optimize=True)
        return f"{filename_stem}.jpg", buf.getvalue(), "image/jpeg"


def _build_genlook_product(vton_category: str, metadata: dict) -> tuple[str, str]:
    """Tạo title & description giàu ngữ nghĩa từ metadata catalog (giúp Genlook lên phom chuẩn)."""
    region_label = {
        "upper_body": "Upper-body top",
        "lower_body": "Lower-body trousers or skirt",
        "dress": "Full-body dress / one-piece outfit",
    }.get(vton_category, "Garment")
    title = str(metadata.get("productName") or "").strip() or region_label
    parts = [region_label]
    for label, key in (
        ("Subtype", "garmentSubtype"),
        ("Color", "colorName"),
        ("Material", "material"),
        ("Pattern", "pattern"),
        ("Fit", "catalogFit"),
        ("Length", "catalogLength"),
    ):
        val = metadata.get(key)
        if val and str(val).strip():
            parts.append(f"{label}: {str(val).strip()}")
    return title, ". ".join(parts) + "."


def _translate_genlook_error(status_code: int, payload: dict | str | None) -> ServiceError:
    """Chuyển mã lỗi HTTP/JSON của Genlook sang ServiceError chuẩn, không lộ GENLOOK_API_KEY."""
    code = str(payload.get("code") or payload.get("errorCode") or "").upper() if isinstance(payload, dict) else ""
    msg = str(payload.get("message") or "")[:200] if isinstance(payload, dict) else ""
    details = payload.get("details") if isinstance(payload, dict) else None
    LOGGER.warning(
        "VTON provider genlook failed: status=%s code=%s message=%s details=%s",
        status_code,
        code,
        msg,
        details,
    )
    if status_code in (401, 403) or code in {"UNAUTHORIZED", "INVALID_API_KEY", "MISSING_API_KEY"}:
        return ServiceError(
            "VTON_API_AUTH_FAILED",
            "GENLOOK_API_KEY không hợp lệ hoặc đã bị thu hồi.",
            status=503,
            retryable=False,
        )
    if status_code == 402 or code == "INSUFFICIENT_CREDITS":
        return ServiceError(
            "VTON_API_AUTH_FAILED",
            "Tài khoản Genlook đã hết credit; vui lòng nạp thêm trên platform.genlook.app.",
            status=503,
            retryable=False,
        )
    if code == "VALIDATION_FAILED":
        return ServiceError(
            "VTON_REQUEST_INVALID",
            "Payload gửi tới Genlook không hợp lệ; kiểm tra log worker để xem trường bị từ chối.",
            status=502,
            retryable=False,
        )
    if status_code == 400 or code in {
        "NO_PERSON_DETECTED", "PERSON_IMAGE_TOO_SMALL", "PERSON_IMAGE_BLANK", "PERSON_TOO_FAR",
        "PRODUCT_IMAGE_TOO_SMALL", "CONTENT_POLICY_VIOLATION",
        "UNSUPPORTED_IMAGE_TYPE", "PRODUCT_IMAGES_REQUIRED", "CUSTOMER_IMAGE_REQUIRED",
        "INVALID_IMAGE_INPUT", "INVALID_IMAGE",
    }:
        return ServiceError(
            "VTON_GENERATION_REJECTED",
            "Genlook từ chối ảnh đầu vào (không nhận diện được người hoặc định dạng ảnh chưa phù hợp).",
            status=422,
            retryable=False,
        )
    return ServiceError(
        "VTON_PROVIDER_UNAVAILABLE",
        "Dịch vụ Genlook tạm gián đoạn hoặc quá hạn; vui lòng thử lại sau.",
        status=503,
        retryable=True,
    )


def _safe_json(response: httpx.Response) -> dict | str:
    try:
        data = response.json()
        return data if isinstance(data, dict) else str(data)
    except Exception:
        return response.text[:300]


def render_genlook(
    person_path: Path,
    garment_path: Path,
    vton_category: str,
    metadata: dict,
    settings,
    progress_cb: Callable[[int, str, str], None] | None = None,
) -> tuple[Image.Image, dict]:
    """Gọi Genlook Try-On API: upload ảnh khách với crop=false (giữ khung gốc) + multipart cutout sản phẩm."""
    if not settings.has_genlook_key():
        raise ServiceError(
            "VTON_API_KEY_MISSING",
            "Chưa cấu hình GENLOOK_API_KEY hợp lệ trong .env để chạy Genlook.",
            status=503,
            retryable=False,
        )

    api_key = settings.genlook_key_value()
    base = settings.genlook_base_url.rstrip("/")
    parsed_base = urlparse(base)
    if parsed_base.scheme != "https" or not parsed_base.netloc:
        raise ServiceError("VTON_PROVIDER_INVALID", "GENLOOK_BASE_URL phải dùng HTTPS hợp lệ.", 500)

    progress_lock = threading.Lock()
    last_pct = 0

    def report(pct: int, stage: str, msg: str) -> None:
        nonlocal last_pct
        if not progress_cb:
            return
        with progress_lock:
            if pct < last_pct:
                return
            last_pct = pct
            progress_cb(pct, stage, msg)

    headers = {"x-api-key": api_key}
    person_file = _prepare_genlook_upload(person_path, "person")
    garment_file = _prepare_genlook_upload(garment_path, "product")
    title, description = _build_genlook_product(vton_category, metadata)
    timeout_s = float(settings.genlook_timeout_seconds)

    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=False) as client:
            # Bước 1: Upload ảnh khách qua /images/upload để điều khiển crop=false (không bị cắt 4:5!)
            report(30, "UPLOADING", "Đang tải ảnh khách lên Genlook (giữ nguyên khung gốc)...")
            up_resp = client.post(
                f"{base}/images/upload",
                headers=headers,
                files={"file": person_file},
                data={
                    "crop": "true" if settings.genlook_crop else "false",
                    "keepForDays": str(settings.genlook_keep_for_days),
                },
            )
            up_body = _safe_json(up_resp)
            if up_resp.status_code >= 400 or not isinstance(up_body, dict) or not up_body.get("imageId"):
                raise _translate_genlook_error(up_resp.status_code, up_body)
            image_id = str(up_body["imageId"])

            # Bước 2: Gọi /try-on/sync dạng multipart (gửi bytes cutout qua fileKey, không cần public URL)
            report(42, "VTON_RENDER", "Genlook đang mặc trang phục lên ảnh...")
            request_body = {
                "products": [
                    {
                        "title": title,
                        "description": description,
                        "images": [{"source": {"fileKey": "product"}}],
                    }
                ],
                "person": {
                    "image": {"source": {"id": image_id}},
                    "keepHeadCovering": True,
                },
                "output": {
                    "resolution": settings.genlook_resolution,
                    "watermark": settings.genlook_watermark,
                    "aiLabel": settings.genlook_ai_label,
                },
            }
            tryon_headers = dict(headers)
            if metadata.get("sessionId"):
                tryon_headers["Idempotency-Key"] = str(metadata["sessionId"])[:255]

            with _cloud_render_ticker(
                report,
                expected_seconds=12.0,
                stage="VTON_RENDER",
                message="Genlook đang mặc trang phục lên ảnh...",
            ):
                sync_resp = client.post(
                    f"{base}/try-on/sync",
                    headers=tryon_headers,
                    data={"data": json.dumps(request_body, ensure_ascii=False)},
                    files={"product": garment_file},
                )
            sync_body = _safe_json(sync_resp)
            if sync_resp.status_code >= 400 or not isinstance(sync_body, dict):
                raise _translate_genlook_error(sync_resp.status_code, sync_body)

            data = sync_body
            # Bước 3: Nếu trả 202 (chạy quá 90s) thì poll /generations/{generationId}
            if sync_resp.status_code == 202 or data.get("status") in {"PENDING", "PROCESSING"}:
                gen_id = str(data.get("generationId") or "")
                if not gen_id:
                    raise _translate_genlook_error(502, data)
                deadline = time.monotonic() + max(30.0, timeout_s)
                while time.monotonic() < deadline:
                    report(68, "VTON_RENDER", "Đang chờ Genlook hoàn tất xử lý ảnh...")
                    time.sleep(2.0)
                    poll_resp = client.get(f"{base}/generations/{gen_id}", headers=headers)
                    poll_body = _safe_json(poll_resp)
                    if poll_resp.status_code >= 400 or not isinstance(poll_body, dict):
                        raise _translate_genlook_error(poll_resp.status_code, poll_body)
                    status = str(poll_body.get("status") or "").upper()
                    if status == "COMPLETED":
                        data = poll_body
                        break
                    if status == "FAILED":
                        raise _translate_genlook_error(502, poll_body)
                else:
                    raise ServiceError("VTON_PROVIDER_UNAVAILABLE", "Genlook xử lý quá thời gian chờ.", 503, retryable=True)
    except ServiceError:
        raise
    except httpx.HTTPError as exc:
        LOGGER.warning("VTON provider genlook network error: %s", exc)
        raise ServiceError("VTON_PROVIDER_UNAVAILABLE", "Lỗi kết nối mạng tới Genlook; vui lòng thử lại.", 503, retryable=True) from exc

    result_url = data.get("resultImageUrl")
    if not result_url:
        raise ServiceError("VTON_GENERATION_REJECTED", "Genlook không trả về resultImageUrl.", 502, retryable=True)

    report(82, "DOWNLOADING", "Đang tải ảnh kết quả chất lượng cao từ Genlook...")
    image = _download_result_image(
        str(result_url),
        max_bytes=settings.max_upload_bytes,
        timeout=float(settings.vton_download_timeout_seconds),
    )
    return image, {
        "provider": "genlook",
        "endpoint": f"{base}/try-on/sync",
        "generationId": data.get("generationId"),
        "category": vton_category,
        "crop": settings.genlook_crop,
        "resolution": settings.genlook_resolution,
    }


PROVIDERS = {
    "fashn_v16": render_fashn_v16,
    "kling_o1": render_kling_o1,
    "genlook": render_genlook,
}


def render_vton(
    person_path: Path,
    garment_path: Path,
    vton_category: str,
    metadata: dict,
    settings,
    progress_cb: Callable[[int, str, str], None] | None = None,
) -> tuple[Image.Image, dict]:
    """Điều phối tới provider đang chọn trong .env (VTON_PROVIDER)."""
    provider_name = settings.vton_provider
    fn = PROVIDERS.get(provider_name)
    if fn is None:
        raise ServiceError("VTON_PROVIDER_INVALID", f"Provider VTON không được hỗ trợ: {provider_name}", 500)
    return fn(person_path, garment_path, vton_category, metadata, settings, progress_cb=progress_cb)
