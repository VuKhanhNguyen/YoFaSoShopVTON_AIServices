"""Kiểm thử Cloud VTON Provider Adapters (FASHN v1.6 & Kling O1) không gọi mạng thật."""
import io
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from api.application import create_app
from api.core.errors import ServiceError
from api.core.store import Store
from api.services.pipeline import Pipeline, prepare_garment_canvas
from api.services.vton_providers import (
    _download_result_image,
    render_vton,
    resolve_fashn_category,
    resolve_garment_photo_type,
)


def _png_bytes(size=(256, 384), color=(180, 210, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class _FakeFalClient:
    """Stub fal_client ghi lại tham số subscribe để verify contract không tốn credit."""

    class InProgress:
        def __init__(self, logs=None):
            self.logs = logs or []

    def __init__(self, response=None, error=None):
        self.uploaded = []
        self.calls = []
        self.response = response or {"images": [{"url": "https://cdn.fashn.ai/fake/output_0.png"}]}
        self.error = error

    def upload_file(self, path: str) -> str:
        self.uploaded.append(path)
        return f"https://fal.media/files/upload_{len(self.uploaded)}.png"

    def subscribe(self, endpoint: str, *, arguments: dict, with_logs: bool = False, on_queue_update=None):
        self.calls.append((endpoint, arguments))
        if on_queue_update:
            on_queue_update(self.InProgress([{"message": "rendering"}]))
        if self.error:
            raise self.error
        return self.response


def test_fashn_category_and_photo_type_mapping():
    assert resolve_fashn_category("upper_body") == "tops"
    assert resolve_fashn_category("lower_body") == "bottoms"
    assert resolve_fashn_category("dress") == "one-pieces"
    with pytest.raises(ServiceError) as exc:
        resolve_fashn_category("unknown")
    assert exc.value.code == "GARMENT_CATEGORY_UNKNOWN"

    # auto + cutout -> flat-lay; auto + raw -> auto; explicit override giữ nguyên
    assert resolve_garment_photo_type("auto", has_cutout=True) == "flat-lay"
    assert resolve_garment_photo_type("auto", has_cutout=False) == "auto"
    assert resolve_garment_photo_type("flat-lay", has_cutout=False) == "flat-lay"
    assert resolve_garment_photo_type("model", has_cutout=True) == "model"


def test_prepare_garment_canvas_handles_cutout_and_raw_images():
    cutout = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for y in range(4, 96):
        for x in range(4, 96):
            cutout.putpixel((x, y), (242, 220, 225, 255))
    rgb, bg, mask = prepare_garment_canvas(cutout)
    assert bg == (255, 255, 255)
    assert rgb.getpixel((0, 0)) == (255, 255, 255)
    assert mask is not None and mask.shape == (rgb.height, rgb.width)
    ys, _ = np.where(mask)
    assert (ys.max() - ys.min()) / rgb.height <= 0.78

    raw = Image.new("RGB", (200, 300), (200, 180, 170))
    rgb_raw, _, mask_raw = prepare_garment_canvas(raw)
    assert rgb_raw.size == (200, 300)
    assert mask_raw is None


def test_render_fashn_locks_num_samples_and_passes_expected_schema(settings, tmp_path, monkeypatch):
    settings.fal_key = "real-fal-test-key"
    fake_fal = _FakeFalClient()
    monkeypatch.setitem(sys.modules, "fal_client", fake_fal)

    person_p = tmp_path / "person.png"
    garment_p = tmp_path / "garment.png"
    Image.new("RGB", (256, 384), "white").save(person_p)
    Image.new("RGB", (256, 384), "pink").save(garment_p)

    stages = []
    with patch("api.services.vton_providers._download_result_image", return_value=Image.new("RGB", (864, 1296), "pink")):
        img, meta = render_vton(
            person_p,
            garment_p,
            "dress",
            {"has_cutout": True, "seed": 77, "samples": 4},  # legacy samples=4 không được nhân phí FASHN!
            settings,
            progress_cb=lambda pct, st, msg: stages.append((pct, st)),
        )

    assert img.size == (864, 1296)
    assert meta["provider"] == "fashn_v16"
    assert meta["category"] == "one-pieces"
    assert meta["garmentPhotoType"] == "flat-lay"
    assert meta["numSamples"] == 1
    assert len(fake_fal.calls) == 1
    endpoint, args = fake_fal.calls[0]
    assert endpoint == "fal-ai/fashn/tryon/v1.6"
    assert args["category"] == "one-pieces"
    assert args["mode"] == "quality"
    assert args["garment_photo_type"] == "flat-lay"
    assert args["num_samples"] == 1
    assert args["segmentation_free"] is True
    assert args["seed"] == 77
    assert any(stage == "VTON_RENDER" for _, stage in stages)


def test_switch_provider_to_kling_o1_via_config_only(settings, tmp_path, monkeypatch):
    settings.fal_key = "real-fal-test-key"
    settings.vton_provider = "kling_o1"
    fake_fal = _FakeFalClient({"images": [{"url": "https://cdn.fal.ai/kling/out.png"}]})
    monkeypatch.setitem(sys.modules, "fal_client", fake_fal)

    person_p = tmp_path / "person.png"
    garment_p = tmp_path / "garment.png"
    Image.new("RGB", (256, 384), "white").save(person_p)
    Image.new("RGB", (256, 384), "blue").save(garment_p)

    with patch("api.services.vton_providers._download_result_image", return_value=Image.new("RGB", (768, 1024), "blue")):
        img, meta = render_vton(
            person_p,
            garment_p,
            "upper_body",
            {"productName": "Áo polo nam", "colorName": "Xanh navy", "material": "Cotton"},
            settings,
        )

    assert img.size == (768, 1024)
    assert meta["provider"] == "kling_o1"
    endpoint, args = fake_fal.calls[0]
    assert endpoint == "fal-ai/kling-image/o1"
    assert len(args["image_urls"]) == 2
    assert "Áo polo nam" in args["prompt"]


def test_provider_errors_and_download_ssrf_guard(settings, tmp_path, monkeypatch):
    person_p = tmp_path / "person.png"
    garment_p = tmp_path / "garment.png"
    Image.new("RGB", (256, 384), "white").save(person_p)
    Image.new("RGB", (256, 384), "white").save(garment_p)

    # 1. Thiếu FAL_KEY -> 503 non-retryable
    settings.fal_key = None
    with pytest.raises(ServiceError) as exc:
        render_vton(person_p, garment_p, "upper_body", {}, settings)
    assert exc.value.code == "VTON_API_KEY_MISSING" and not exc.value.retryable

    # 2. Placeholder CHANGE_ME -> cũng bị chặn
    settings.fal_key = "CHANGE_ME_FAL_API_KEY"
    assert settings.has_fal_key() is False

    # 3. Auth error -> VTON_API_AUTH_FAILED không lộ key
    settings.fal_key = "secret-key-never-leak"
    monkeypatch.setitem(sys.modules, "fal_client", _FakeFalClient(error=RuntimeError("401 Unauthorized secret-key-never-leak")))
    with pytest.raises(ServiceError) as auth_exc:
        render_vton(person_p, garment_p, "upper_body", {}, settings)
    assert auth_exc.value.code == "VTON_API_AUTH_FAILED"
    assert "secret-key-never-leak" not in auth_exc.value.message

    # 4. Moderation / safety refusal -> 422 non-retryable
    monkeypatch.setitem(sys.modules, "fal_client", _FakeFalClient(error=RuntimeError("Blocked by content moderation policy")))
    with pytest.raises(ServiceError) as mod_exc:
        render_vton(person_p, garment_p, "upper_body", {}, settings)
    assert mod_exc.value.code == "VTON_GENERATION_REJECTED" and not mod_exc.value.retryable

    # 5. Non-HTTPS URL bị từ chối ở lớp download
    with pytest.raises(ServiceError) as url_exc:
        _download_result_image("http://127.0.0.1/internal.png", max_bytes=10_000_000)
    assert url_exc.value.code == "VTON_RESULT_INVALID"


def test_pipeline_real_tryon_uses_fashn_and_sets_evaluated_quality(settings, monkeypatch):
    """End-to-end Pipeline.tryon() chế độ real: YOLO validation -> FASHN -> lưu asset PNG, không gọi OOTD/VLM."""
    settings.ai_mode = "real"
    settings.fal_key = "real-fal-test-key"
    pipeline = Pipeline(settings)

    person_asset = pipeline.assets.upload(_png_bytes((300, 500), (230, 210, 200)), "person")
    # Cutout garment với alpha trong suốt
    cutout = Image.new("RGBA", (256, 384), (0, 0, 0, 0))
    for y in range(40, 340):
        for x in range(40, 216):
            cutout.putpixel((x, y), (245, 195, 205, 255))
    buf = io.BytesIO()
    cutout.save(buf, "PNG")
    garment_asset = pipeline.assets.upload(buf.getvalue(), "garment")

    # Mock YOLO pose validation thành công -> full_body; đảm bảo VLM KHÔNG bị gọi
    monkeypatch.setattr(
        pipeline.vision,
        "validate_person",
        lambda img: {"accepted": True, "personFrame": "full_body", "simulated": False, "warnings": []},
    )
    monkeypatch.setattr(pipeline.vlm, "load", lambda: pytest.fail("VLM không được gọi trong VTON!"))

    fake_fal = _FakeFalClient()
    monkeypatch.setitem(sys.modules, "fal_client", fake_fal)

    payload = {
        "sessionId": "TD_FASHN_01",
        "productId": "SP001",
        "variantId": "BT001",
        "personAssetId": person_asset["assetId"],
        "garmentAssetId": garment_asset["assetId"],
        "vtonCategory": "dress",
        "garmentSubtype": "long_sleeved_dress",
        "garmentLength": "LONG",
        "seed": 42,
        "samples": 4,
    }
    job = pipeline.store.submit(
        "TRY_ON",
        "TD_FASHN_01",
        payload,
        [person_asset["assetId"], garment_asset["assetId"]],
        "real",
        20,
    )

    with patch("api.services.vton_providers._download_result_image", return_value=Image.new("RGB", (864, 1296), (245, 195, 205))):
        result = pipeline.run(job)

    assert result["simulated"] is False
    assert result["quality"]["evaluated"] is True
    assert result["quality"]["provider"] == "fashn_v16"
    assert result["provider"]["category"] == "one-pieces"
    assert result["provider"]["garmentPhotoType"] == "flat-lay"
    assert result["route"]["personFrame"] == "full_body"
    assert result["resultAsset"]["width"] == 864
    assert result["resultAsset"]["height"] == 1296
    # Thư mục tạm jobs/{job_id} đã được dọn sạch sau khi lưu asset
    assert not (settings.runtime_dir / "jobs" / job["id"]).exists()


def test_real_health_ready_does_not_require_ootdiffusion(settings, tmp_path):
    """Chế độ real chỉ cần DeepFashion2 + YOLO Pose + FAL_KEY + heartbeat (không cần OOTD)."""
    settings.ai_mode = "real"
    df = tmp_path / "deepfashion.pt"
    pose = tmp_path / "pose.pt"
    df.write_bytes(b"ok")
    pose.write_bytes(b"ok")
    settings.deepfashion_weights = df
    settings.pose_weights = pose
    Store(settings).heartbeat(time.time(), "real", write=True)

    # Thiếu FAL_KEY -> 503
    settings.fal_key = None
    with TestClient(create_app(settings)) as c:
        c.headers["X-Internal-API-Key"] = settings.ai_internal_api_key.get_secret_value()
        assert c.get("/health/ready").status_code == 503

    # Có FAL_KEY -> 200 ready mà không hề có OOTDiffusion, và không bao giờ lộ key
    settings.fal_key = SecretStr("fal-secret-do-not-expose")
    with TestClient(create_app(settings)) as c:
        c.headers["X-Internal-API-Key"] = settings.ai_internal_api_key.get_secret_value()
        ready = c.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["vtonProvider"] == "fashn_v16"
        models = c.get("/internal/v1/models")
        assert models.status_code == 200
        assert "fal-secret-do-not-expose" not in models.text
        assert all("OOTDiffusion" not in m["name"] for m in models.json()["models"])
        cfg = c.get("/internal/v1/config-check")
        assert cfg.status_code == 200 and cfg.json()["vtonReady"] is True
        assert "fal-secret-do-not-expose" not in cfg.text


def test_switch_provider_to_genlook_keeps_crop_off_and_uploads_local_cutout(settings, tmp_path):
    """VTON_PROVIDER=genlook upload ảnh khách với crop=false, gửi cutout multipart, không cần public URL."""
    import json

    import httpx

    settings.vton_provider = "genlook"
    settings.genlook_api_key = SecretStr("gk_test_secret_never_leak")
    settings.genlook_crop = False

    person_p = tmp_path / "person.png"
    garment_p = tmp_path / "garment.png"
    Image.new("RGB", (300, 500), "white").save(person_p)
    Image.new("RGB", (256, 384), "beige").save(garment_p)

    captured = []

    class _FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    class _FakeHttpxClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, *, headers=None, files=None, data=None):
            captured.append(("POST", url, headers, list((files or {}).keys()), data))
            if url.endswith("/images/upload"):
                return _FakeResponse(200, {"imageId": "ephemeral/customer/ttl-1d/test-id.png"})
            return _FakeResponse(
                200,
                {
                    "generationId": "gen_sync_001",
                    "status": "COMPLETED",
                    "resultImageUrl": "https://storage.googleapis.com/genlook/out.png",
                },
            )

        def get(self, url, *, headers=None):
            captured.append(("GET", url, headers, [], None))
            return _FakeResponse(200, {"status": "COMPLETED", "resultImageUrl": "https://storage.googleapis.com/genlook/out.png"})

    with (
        patch.object(httpx, "Client", _FakeHttpxClient),
        patch("api.services.vton_providers._download_result_image", return_value=Image.new("RGB", (900, 1200), "beige")),
    ):
        img, meta = render_vton(
            person_p,
            garment_p,
            "dress",
            {
                "sessionId": "TD_GENLOOK_01",
                "productName": "Đầm sơ mi xếp tầng kem",
                "colorName": "Kem",
                "material": "Cotton",
                "catalogFit": "SUONG_VUA",
            },
            settings,
        )

    assert img.size == (900, 1200)
    assert meta["provider"] == "genlook"
    assert meta["crop"] is False
    assert meta["generationId"] == "gen_sync_001"
    assert len(captured) == 2

    # Verify bước 1: upload customer photo với crop="false" (bắt buộc để giữ khung ảnh gốc!)
    _, up_url, up_headers, up_files, up_data = captured[0]
    assert up_url == "https://api.genlook.app/tryon/v1/images/upload"
    assert up_headers["x-api-key"] == "gk_test_secret_never_leak"
    assert up_files == ["file"]
    assert up_data["crop"] == "false"

    # Verify bước 2: /try-on/sync multipart với person.source.id + product fileKey + Idempotency-Key
    _, sync_url, sync_headers, sync_files, sync_data = captured[1]
    assert sync_url == "https://api.genlook.app/tryon/v1/try-on/sync"
    assert sync_headers["Idempotency-Key"] == "TD_GENLOOK_01"
    assert sync_files == ["product"]
    body = json.loads(sync_data["data"])
    assert "keepHeadCovering" not in body
    assert body["person"] == {
        "image": {"source": {"id": "ephemeral/customer/ttl-1d/test-id.png"}},
        "keepHeadCovering": True,
    }
    assert body["products"][0]["images"] == [{"source": {"fileKey": "product"}}]
    assert body["products"][0]["title"] == "Đầm sơ mi xếp tầng kem"
    assert body["output"] == {"resolution": "default", "watermark": False, "aiLabel": False}


def test_genlook_202_polling_errors_and_health_ready(settings, tmp_path):
    """Kiểm tra nhánh 202 polling, ánh xạ lỗi 402/400 và /health/ready khi chọn VTON_PROVIDER=genlook."""
    import json

    import httpx

    settings.vton_provider = "genlook"
    settings.fal_key = None  # Không cần FAL_KEY khi dùng genlook!
    person_p = tmp_path / "person.png"
    garment_p = tmp_path / "garment.png"
    Image.new("RGB", (256, 384), "white").save(person_p)
    Image.new("RGB", (256, 384), "blue").save(garment_p)

    # 1. Thiếu GENLOOK_API_KEY -> VTON_API_KEY_MISSING
    settings.genlook_api_key = None
    with pytest.raises(ServiceError) as missing_exc:
        render_vton(person_p, garment_p, "upper_body", {}, settings)
    assert missing_exc.value.code == "VTON_API_KEY_MISSING"

    # 2. 202 -> poll /generations/{id} -> COMPLETED
    settings.genlook_api_key = SecretStr("gk_poll_secret")
    poll_calls = []

    class _FakeResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    class _PollingClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            if url.endswith("/images/upload"):
                return _FakeResp(200, {"imageId": "ephemeral/img-202"})
            return _FakeResp(202, {"generationId": "gen_202", "status": "PROCESSING"})

        def get(self, url, **kwargs):
            poll_calls.append(url)
            if len(poll_calls) == 1:
                return _FakeResp(200, {"generationId": "gen_202", "status": "PROCESSING"})
            return _FakeResp(
                200,
                {"generationId": "gen_202", "status": "COMPLETED", "resultImageUrl": "https://storage.googleapis.com/g/ok.png"},
            )

    with (
        patch.object(httpx, "Client", _PollingClient),
        patch("api.services.vton_providers.time.sleep", lambda _: None),
        patch("api.services.vton_providers._download_result_image", return_value=Image.new("RGB", (800, 1200), "blue")),
    ):
        img, meta = render_vton(person_p, garment_p, "upper_body", {}, settings)
    assert img.size == (800, 1200) and meta["generationId"] == "gen_202"
    assert len(poll_calls) == 2 and poll_calls[0].endswith("/generations/gen_202")

    # 3. 402 INSUFFICIENT_CREDITS -> VTON_API_AUTH_FAILED không lộ key
    class _ErrorClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            if url.endswith("/images/upload"):
                return _FakeResp(200, {"imageId": "ephemeral/img-err"})
            return _FakeResp(402, {"code": "INSUFFICIENT_CREDITS", "message": "Top up gk_poll_secret"})

    with patch.object(httpx, "Client", _ErrorClient):
        with pytest.raises(ServiceError) as credit_exc:
            render_vton(person_p, garment_p, "upper_body", {}, settings)
    assert credit_exc.value.code == "VTON_API_AUTH_FAILED"
    assert "gk_poll_secret" not in credit_exc.value.message

    # 4. /health/ready chế độ real với genlook chỉ cần GENLOOK_API_KEY (không cần FAL_KEY)
    settings.ai_mode = "real"
    df = tmp_path / "df.pt"
    pose = tmp_path / "pose.pt"
    df.write_bytes(b"ok")
    pose.write_bytes(b"ok")
    settings.deepfashion_weights = df
    settings.pose_weights = pose
    Store(settings).heartbeat(time.time(), "real", write=True)
    with TestClient(create_app(settings)) as c:
        c.headers["X-Internal-API-Key"] = settings.ai_internal_api_key.get_secret_value()
        ready = c.get("/health/ready")
        assert ready.status_code == 200 and ready.json()["vtonProvider"] == "genlook"
        cfg = c.get("/internal/v1/config-check")
        assert cfg.status_code == 200 and cfg.json()["vtonReady"] is True and cfg.json()["genlookCrop"] is False
        assert "gk_poll_secret" not in cfg.text



def test_resolve_vision_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    """Nếu CUDA lỗi (vd. Error 304) khiến torch.cuda.is_available()=False, phải fallback sang cpu."""
    import types
    from api.services.vision import _resolve_vision_device

    fake_torch_off = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch_off)
    assert _resolve_vision_device("cuda") == "cpu"
    assert _resolve_vision_device("0") == "cpu"
    assert _resolve_vision_device("cpu") == "cpu"

    fake_torch_on = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch_on)
    assert _resolve_vision_device("cuda") == "cuda"
    assert _resolve_vision_device("0") == "0"


def test_fashn_progress_ticker_advances_monotonically_and_stops(settings, tmp_path, monkeypatch):
    """Tiến trình VTON_RENDER tăng đều trong khi chờ FASHN và không vượt ngưỡng trước khi tải ảnh."""
    settings.fal_key = "real-fal-test-key"
    settings.fashn_mode = "performance"

    class _SlowFalClient(_FakeFalClient):
        def subscribe(self, endpoint: str, *, arguments: dict, with_logs: bool = False, on_queue_update=None):
            self.calls.append((endpoint, arguments))
            if on_queue_update:
                on_queue_update(self.InProgress([]))
            time.sleep(1.1)
            return self.response

    monkeypatch.setitem(sys.modules, "fal_client", _SlowFalClient())
    person_p = tmp_path / "person.png"
    garment_p = tmp_path / "garment.png"
    Image.new("RGB", (256, 384), "white").save(person_p)
    Image.new("RGB", (256, 384), "pink").save(garment_p)

    events = []
    with patch("api.services.vton_providers._download_result_image", return_value=Image.new("RGB", (864, 1296), "pink")):
        render_vton(person_p, garment_p, "upper_body", {}, settings, progress_cb=lambda pct, st, msg: events.append((pct, st)))

    pcts = [p for p, _ in events]
    assert pcts == sorted(pcts)
    render_pcts = [p for p, st in events if st == "VTON_RENDER"]
    assert len(render_pcts) >= 2 and max(render_pcts) <= 80
    assert events[-1] == (84, "DOWNLOADING")


