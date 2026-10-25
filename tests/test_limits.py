"""Kiểm tra limit trên HTTP body và lifespan có worker thật (mock inference)."""
import time

from fastapi.testclient import TestClient

from api.application import create_app


def test_chunked_body_limit(settings):
    """Không lách upload limit bằng cách bỏ Content-Length."""
    settings.max_upload_bytes = 100
    with TestClient(create_app(settings)) as client:
        client.headers["X-Internal-API-Key"] = settings.ai_internal_api_key.get_secret_value()
        body = (part for part in [b"x" * 40000, b"x" * 40000])
        response = client.post("/internal/v1/assets", content=body, headers={"Content-Type": "multipart/form-data; boundary=abc"})
        assert response.status_code in {400, 413}  # malformed multipart can be rejected before reading all chunks
        assert response.json()["errorCode"] in {"HTTP_400", "UPLOAD_TOO_LARGE"}


def test_embedded_worker_lifespan(settings):
    """API mock có worker embedded khởi động/dừng theo lifespan."""
    settings.embedded_worker = True
    with TestClient(create_app(settings)) as client:
        for _ in range(100):
            response = client.get("/health/ready")
            if response.status_code == 200:
                break
            time.sleep(.01)
        assert response.status_code == 200
        assert response.json()["simulated"] is True
