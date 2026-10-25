"""API functional/security/contract regression; mock không chứng minh chất lượng GPU."""
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from openapi_spec_validator import validate
from PIL import Image

from api.application import create_app
from api.core.config import Settings
from api.core.errors import ServiceError
from api.core.store import Store
from api.services.pipeline import gate_candidate
from api.services.routing import category_for
from workers.runner import Worker

PREFIX = "/internal/v1"


@pytest.mark.parametrize("path", ["/capabilities", "/models", "/config-check", "/jobs", "/jobs/none", "/assets/none", "/assets/none/content", "/try-on/jobs/TD001"])
def test_auth_required(client, path):
    """Mọi GET nội bộ kể cả ảnh private phải chặn khóa thiếu/sai."""
    response = client.get(PREFIX + path, headers={"X-Internal-API-Key": "wrong"})
    assert response.status_code == 401
    assert response.json()["errorCode"] == "UNAUTHORIZED"
    assert response.json()["correlationId"] == response.headers["X-Correlation-ID"]


def test_system_and_health(client):
    """Liveness khác readiness; không nói inference ready khi chưa có worker."""
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    for path in ("capabilities", "models", "config-check"):
        result = client.get(PREFIX + "/" + path)
        assert result.status_code == 200
        assert "test-key" not in result.text
    assert client.get(PREFIX + "/config-check").json()["simulated"] is True


def test_upload_metadata_content_delete(client, upload):
    """Tên tùy ý không thành path; ảnh được chuẩn hóa và download có bảo vệ."""
    asset = upload(name="../../private.png")
    assert asset["width"] == 256 and asset["height"] == 384
    assert client.get(PREFIX + "/assets/" + asset["assetId"]).json() == asset
    content = client.get(asset["contentUrl"])
    assert content.status_code == 200 and content.headers["content-type"].startswith("image/png")
    assert content.headers["cache-control"] == "no-store"
    assert client.delete(PREFIX + "/assets/" + asset["assetId"]).status_code == 204
    assert client.get(asset["contentUrl"]).status_code == 404


@pytest.mark.parametrize("method,path", [("post", "/persons/validate"), ("post", "/routing/preview"), ("post", "/products/analyze"), ("post", "/try-on/jobs"), ("post", "/jobs/missing/cancel"), ("post", "/jobs/missing/retry"), ("delete", "/assets/missing")])
def test_write_auth_required(client, method, path):
    """Các thao tác ghi cũng phải xác thực trước khi tác động queue/storage."""
    response = client.request(method, PREFIX + path, headers={"X-Internal-API-Key": "wrong"}, json={})
    assert response.status_code == 401


def test_exif_removed(client):
    """Chuẩn hóa ảnh khách không giữ EXIF/GPS/comment ngoài ý muốn."""
    source = Image.new("RGB", (256, 256), "white")
    exif = Image.Exif()
    exif[270] = "private-description"
    data = io.BytesIO()
    source.save(data, "JPEG", exif=exif)
    uploaded = client.post(PREFIX + "/assets", data={"role": "person"}, files={"file": ("arbitrary.jpg", data.getvalue())})
    assert uploaded.status_code == 201
    saved = Image.open(io.BytesIO(client.get(uploaded.json()["contentUrl"]).content))
    assert not saved.getexif()


@pytest.mark.parametrize("content", [b"", b"not image", b"<script>alert(1)</script>"])
def test_invalid_upload(client, content):
    """Không tin extension/MIME; ảnh giả bị chặn trước khi ghi storage."""
    response = client.post(PREFIX + "/assets", data={"role": "person"}, files={"file": ("image.png", content, "image/png")})
    assert response.status_code == 422
    assert response.json()["errorCode"] == "IMAGE_INVALID"


def test_upload_limits_and_invalid_role(client, settings, image_bytes):
    """Giới hạn byte/pixel/cạnh và role, không cần load detector."""
    small = io.BytesIO()
    Image.new("RGB", (32, 32)).save(small, "PNG")
    assert client.post(PREFIX + "/assets", data={"role": "person"}, files={"file": ("x", small.getvalue())}).json()["errorCode"] == "IMAGE_TOO_SMALL"
    assert client.post(PREFIX + "/assets", data={"role": "result"}, files={"file": ("x", image_bytes)}).status_code == 422
    settings.max_image_pixels = 100
    assert client.post(PREFIX + "/assets", data={"role": "person"}, files={"file": ("x", image_bytes)}).status_code == 413
    settings.max_upload_bytes = 10
    assert client.post(PREFIX + "/assets", data={"role": "person"}, files={"file": ("x", image_bytes)}).status_code == 413


def test_validate_person_route(client, upload):
    """Mock frame rõ ràng; role sai hoặc thiếu ID không được coi như người thật."""
    person = upload("person")
    result = client.post(PREFIX + "/persons/validate", json={"assetId": person["assetId"], "vtonCategory": "upper_body"})
    assert result.json()["simulated"] and result.json()["accepted"]
    garment = upload()
    assert client.post(PREFIX + "/persons/validate", json={"assetId": garment["assetId"]}).status_code == 422
    assert client.post(PREFIX + "/persons/validate", json={"assetId": "missing"}).status_code == 404


@pytest.mark.parametrize("category,subtype,length,frame,accepted,model", [
    ("upper_body", "sling", "UNKNOWN", "half_body", True, "hd"),
    ("lower_body", "trousers", "LONG", "half_body", False, "dc"),
    ("lower_body", "shorts", "SHORT", "three_quarter", True, "dc"),
    ("lower_body", "skirt", "UNKNOWN", "half_body", False, "dc"),
    ("dress", "sling_dress", "LONG", "full_body", True, "dc"),
    ("dress", "sling_dress", "SHORT", "invalid", False, "dc"),
])
def test_routing(client, category, subtype, length, frame, accepted, model):
    """Bảo vệ rule đồ dài/category từ tài liệu thay vì chỉ test happy path áo."""
    response = client.post(PREFIX + "/routing/preview", json={"vtonCategory": category, "garmentSubtype": subtype, "garmentLength": length, "personFrame": frame})
    assert response.status_code == 200
    assert response.json()["accepted"] is accepted
    assert response.json()["modelType"] == model


def test_category_mismatch_and_unknown_fields(client, tryon_payload):
    """Không fallback category, không chấp nhận URL/path/field ngoài hợp đồng."""
    assert client.post(PREFIX + "/routing/preview", json={"vtonCategory": "dress", "garmentSubtype": "shorts", "personFrame": "full_body"}).status_code == 422
    assert client.post(PREFIX + "/try-on/jobs", json={**tryon_payload, "personImageUrl": "http://169.254.169.254"}).status_code == 422
    assert client.post(PREFIX + "/try-on/jobs", json={**tryon_payload, "sessionId": "../secret"}).status_code == 422
    assert client.post(PREFIX + "/try-on/jobs", json={**tryon_payload, "samples": 10}).status_code == 422


def test_job_end_to_end_mock(client, app, settings, tryon_payload):
    """Submit/poll/result/replay; output mock là placeholder có nhãn, không ảnh gốc."""
    first = client.post(PREFIX + "/try-on/jobs", json=tryon_payload)
    assert first.status_code == 202
    job = first.json()
    assert job["status"] == "QUEUED" and job["simulated"]
    assert client.post(PREFIX + "/try-on/jobs", json=tryon_payload).json()["jobId"] == job["jobId"]
    assert client.post(PREFIX + "/try-on/jobs", json={**tryon_payload, "seed": 99}).status_code == 409
    assert client.post(PREFIX + f"/jobs/{job['jobId']}/retry").status_code == 409
    Worker(settings).run_one()
    final = client.get(PREFIX + f"/jobs/{job['jobId']}").json()
    assert final["status"] == "COMPLETED", final
    assert final["result"]["simulated"]
    assert final["result"]["quality"]["evaluated"] is False
    source = client.get(PREFIX + f"/assets/{tryon_payload['personAssetId']}").json()
    assert final["result"]["resultAsset"]["sha256"] != source["sha256"]
    assert client.get(final["result"]["resultAsset"]["contentUrl"]).status_code == 200
    assert client.get(PREFIX + "/try-on/jobs/TD001").json()["jobId"] == job["jobId"]
    assert client.get(PREFIX + "/jobs?status=COMPLETED&limit=1&offset=0").json()["total"] == 1
    assert client.delete(PREFIX + "/assets/" + tryon_payload["personAssetId"]).status_code == 409
    assert client.post(PREFIX + f"/jobs/{job['jobId']}/cancel").status_code == 409


def test_product_analysis_assets(client, upload, settings):
    """Mỗi ảnh biến thể -> một cutout và mask cùng kích thước ảnh gốc, fit chung."""
    ids = [upload()["assetId"] for _ in range(2)]
    payload = {"analysisId": "ANA1", "productId": "SP1", "productAssetIds": ids, "categoryOverride": "dress"}
    job = client.post(PREFIX + "/products/analyze", json=payload)
    assert job.status_code == 202, job.text
    Worker(settings).run_one()
    result = client.get(PREFIX + "/jobs/" + job.json()["jobId"]).json()
    assert result["status"] == "COMPLETED", result
    assert len(result["result"]["variants"]) == 2
    for variant in result["result"]["variants"]:
        assert variant["vtonCategory"] == "dress"
        rgba = Image.open(io.BytesIO(client.get(variant["cutout"]["contentUrl"]).content))
        mask = Image.open(io.BytesIO(client.get(variant["mask"]["contentUrl"]).content))
        assert rgba.mode == "RGBA" and rgba.getchannel("A").getextrema() == (0, 255)
        assert mask.size == (256, 384) and set(np.unique(mask)) == {0, 255}
    assert client.post(PREFIX + "/products/analyze", json={**payload, "productAssetIds": [ids[0], ids[0]]}).status_code == 422


def test_cancel_retry_queue_and_persistence(client, settings, tryon_payload):
    """Queue đầy, cancel idempotent và DB còn sau tạo application mới."""
    settings.max_queue_size = 1
    job = client.post(PREFIX + "/try-on/jobs", json=tryon_payload).json()
    assert client.post(PREFIX + "/try-on/jobs", json={**tryon_payload, "sessionId": "TD2"}).status_code == 429
    assert client.post(PREFIX + f"/jobs/{job['jobId']}/cancel").json()["status"] == "CANCELLED"
    assert client.post(PREFIX + f"/jobs/{job['jobId']}/cancel").status_code == 200
    assert Store(settings.runtime_dir).job(job["jobId"])["status"] == "CANCELLED"
    assert client.get(PREFIX + "/jobs/not-found").status_code == 404
    assert client.post(PREFIX + "/jobs/not-found/retry").status_code == 404


def test_transient_retry(client, settings, tryon_payload):
    """Retry chỉ lỗi tạm và không vượt attempts; transaction không enqueue trùng."""
    row = client.post(PREFIX + "/try-on/jobs", json=tryon_payload).json()
    store = Store(settings.runtime_dir)
    claimed = store.claim()
    store.finish(claimed["id"], error={"errorCode": "JOB_TIMEOUT", "retryable": True})
    assert client.post(PREFIX + f"/jobs/{row['jobId']}/retry").json()["status"] == "QUEUED"
    store.claim()
    store.finish(claimed["id"], error={"errorCode": "JOB_TIMEOUT", "retryable": True})
    assert client.post(PREFIX + f"/jobs/{row['jobId']}/retry").status_code == 409


def test_frame_fail_is_not_mock_success(client, settings, tryon_payload):
    """Giả lập input frame sai để test pipeline dừng trước tạo ảnh."""
    settings.mock_person_frame = "invalid"
    job = client.post(PREFIX + "/try-on/jobs", json=tryon_payload).json()
    Worker(settings).run_one()
    result = client.get(PREFIX + "/jobs/" + job["jobId"]).json()
    assert result["status"] == "FAILED"
    assert result["error"]["errorCode"] == "PERSON_FRAME_INVALID"
    assert result["result"] is None


def test_rate_limit(settings):
    """Rate limit có 429 và các header để caller biết khi nào retry."""
    settings.rate_limit_per_minute = 2
    with TestClient(create_app(settings)) as c:
        c.headers["X-Internal-API-Key"] = settings.ai_internal_api_key.get_secret_value()
        assert c.get(PREFIX + "/capabilities").status_code == 200
        assert c.get(PREFIX + "/capabilities").status_code == 200
        response = c.get(PREFIX + "/capabilities")
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) > 0
        assert response.headers["X-RateLimit-Remaining"] == "0"


def test_security_config(settings):
    """Không cho chạy production/mock, không khởi động với khóa mẫu."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, ai_internal_api_key="CHANGE_ME")
    with pytest.raises(ValueError):
        Settings(_env_file=None, ai_internal_api_key="k" * 40, app_env="production", ai_mode="mock")
    with pytest.raises(ValueError):
        Settings(_env_file=None, ai_internal_api_key="k" * 40, ai_mode="real", embedded_worker=True)


def test_concurrent_idempotency(client, settings, tryon_payload):
    """20 caller cùng submit qua DB chỉ tạo một job (không đo throughput inference)."""
    client.post(PREFIX + "/try-on/jobs", json=tryon_payload)
    store = Store(settings.runtime_dir)
    payload = __import__("api.schemas", fromlist=["TryOnRequest"]).TryOnRequest(**tryon_payload).model_dump(by_alias=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(lambda _: store.submit("TRY_ON", "TD001", payload, [tryon_payload["personAssetId"], tryon_payload["garmentAssetId"]], "mock", 20), range(20)))
    assert len({row["id"] for row in rows}) == 1


@pytest.mark.parametrize("category,subtype,outside,body,limb,expected", [
    ("upper_body", "sling", .9, .9, .2, False),
    ("dress", "sling_dress", .9, .1, .9, False),
    ("lower_body", "shorts", .2, .9, .9, False),
    ("upper_body", "long_sleeved_shirt", .9, .9, .1, True),
    ("dress", "long_sleeved_dress", .9, .9, .1, True),
])
def test_artifact_gate(category, subtype, outside, body, limb, expected):
    """Candidate màu khớp nhưng tay hỏng phải fail; dài tay được che tay hợp lệ."""
    assert gate_candidate(category, subtype, outside, body, limb, 3) is expected
    assert not gate_candidate(category, subtype, outside, body, limb, 0)


def test_labels():
    """Chuẩn hóa nhiều cách đặt nhãn DeepFashion2, không nhầm vùng."""
    assert category_for("short sleeve top") == "upper_body"
    with pytest.raises(ServiceError):
        category_for("unknown")


def test_openapi_valid_and_frozen(client):
    """Validator đầy đủ OpenAPI 3.1, không chỉ parse JSON/YAML."""
    schema = client.get("/openapi.json").json()
    validate(schema)
    frozen = Path("contracts/openapi.json")
    assert frozen.exists(), "Chạy scripts/export_contract.py trước"
    assert schema == json.loads(frozen.read_text(encoding="utf-8"))
    for path, operations in schema["paths"].items():
        if path.startswith(PREFIX):
            for operation in operations.values():
                assert operation.get("security")
