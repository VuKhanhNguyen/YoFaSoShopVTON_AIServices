"""Fixtures chỉ dùng ảnh tổng hợp, không tải model hoặc gửi ảnh khách ra ngoài."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.application import create_app
from api.core.config import Settings

KEY = "test-key-" + "a" * 40


@pytest.fixture
def settings(tmp_path):
    """Mỗi test có DB/asset độc lập và không đọc .env của người dùng."""
    yield Settings(_env_file=None, ai_internal_api_key=KEY, app_env="test", ai_mode="mock", runtime_dir=tmp_path,
                   embedded_worker=False, rate_limit_per_minute=10000)


@pytest.fixture
def app(settings):
    """Factory không cần model/secret production."""
    return create_app(settings)


@pytest.fixture
def client(app):
    """Test lifespan và HTTP giống Uvicorn, không bỏ qua middleware/auth."""
    with TestClient(app) as value:
        value.headers["X-Internal-API-Key"] = KEY
        yield value


@pytest.fixture
def image_bytes():
    """PNG 256x384 có các vùng màu để kiểm tra asset/cutout/mask."""
    image = Image.new("RGB", (256, 384), "white")
    image.paste("red", (60, 80, 200, 310))
    data = io.BytesIO()
    image.save(data, "PNG")
    return data.getvalue()


@pytest.fixture
def upload(client, image_bytes):
    """Helper upload qua HTTP thật trong TestClient; không bỏ qua kiểm tra ảnh."""
    def send(role="garment", data=None, name="anh bat ky.webp"):
        response = client.post("/internal/v1/assets", data={"role": role}, files={"file": (name, data or image_bytes, "image/png")})
        assert response.status_code == 201, response.text
        return response.json()
    return send


@pytest.fixture
def tryon_payload(upload):
    """Payload mặc định đã có ảnh đúng role; kiểm tra model là bước khác."""
    return {"sessionId": "TD001", "productId": "SP001", "variantId": "BT001", "personAssetId": upload("person")["assetId"],
            "garmentAssetId": upload("garment")["assetId"], "vtonCategory": "upper_body", "garmentSubtype": "sling"}
