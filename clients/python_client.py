"""Client Python có typed DTO cùng contract OpenAPI; cần httpx trong requirements-dev.

Mọi retry submit phải dùng lại cùng sessionId/analysisId; client không tự tạo ID mới.
"""
import httpx

from api.schemas import AssetResponse, JobResponse, ProductRequest, TryOnRequest


class AIClient:
    """Context manager đóng connection và luôn gửi khóa nội bộ qua header."""
    def __init__(self, base_url: str, api_key: str, timeout: float = 15):
        """Tạo client với endpoint/key do caller đọc từ cấu hình môi trường."""
        self.http = httpx.Client(base_url=base_url, headers={"X-Internal-API-Key": api_key}, timeout=timeout, trust_env=False)

    def __enter__(self):
        """Cho phép dùng with AIClient(...) để không rò connection."""
        return self

    def __exit__(self, *args):
        """Đóng transport dù request phát sinh lỗi."""
        self.http.close()

    def upload(self, file_bytes: bytes, role: str, filename: str = "upload.png") -> AssetResponse:
        """Tải ảnh trước rồi dùng assetId trong job, không đưa path laptop tới server."""
        response = self.http.post("/internal/v1/assets", data={"role": role}, files={"file": (filename, file_bytes)})
        response.raise_for_status()
        return AssetResponse.model_validate(response.json())

    def submit_tryon(self, body: TryOnRequest) -> JobResponse:
        """Submit idempotent; status QUEUED không có nghĩa đã tạo ảnh."""
        response = self.http.post("/internal/v1/try-on/jobs", json=body.model_dump(by_alias=True))
        response.raise_for_status()
        return JobResponse.model_validate(response.json())

    def analyze_product(self, body: ProductRequest) -> JobResponse:
        """Submit tác vụ sản phẩm một mẫu/nhiều biến thể."""
        response = self.http.post("/internal/v1/products/analyze", json=body.model_dump(by_alias=True))
        response.raise_for_status()
        return JobResponse.model_validate(response.json())

    def get_job(self, job_id: str) -> JobResponse:
        """Đọc trạng thái kết quả bằng cùng DTO của API."""
        response = self.http.get(f"/internal/v1/jobs/{job_id}")
        response.raise_for_status()
        return JobResponse.model_validate(response.json())
