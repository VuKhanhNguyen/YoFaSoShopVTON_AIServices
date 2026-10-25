"""Lỗi worker, restart, lease và retention không yêu cầu GPU thật."""
import threading
import time

import pytest

from api.core.errors import ServiceError
from api.core.store import Store
from api.services.pipeline import Pipeline
from api.services.vision import Vision
from scripts.cleanup_runtime import cleanup
from workers.runner import Worker, error_data, worker_lock


def slow_child(config, connection):
    """Process cố tình chờ để chứng minh watchdog terminate trước job tiếp."""
    time.sleep(10)


def test_real_watchdog_terminates_child(settings):
    """Test cơ chế timeout bằng process thật nhưng không nạp GPU/model."""
    settings.ai_mode = "real"
    settings.job_timeout_seconds = 1
    worker = Worker(settings, child_target=slow_child)
    try:
        with pytest.raises(ServiceError) as exc:
            worker.execute({"id": "timeout-fixture"})
        assert exc.value.code == "JOB_TIMEOUT"
        assert worker.process is None and worker.connection is None
    finally:
        worker.shutdown_child()


def test_worker_restart_recovery(settings, client, tryon_payload):
    """Job active sau crash phải FAILED/RETRYABLE thay vì treo PROCESSING mãi."""
    row = client.post("/internal/v1/try-on/jobs", json=tryon_payload).json()
    store = Store(settings.runtime_dir)
    store.claim()
    worker = Worker(settings)
    thread = threading.Thread(target=worker.serve)
    thread.start()
    try:
        for _ in range(100):
            job = store.job(row["jobId"])
            if job["status"] == "FAILED":
                break
            time.sleep(.01)
        assert store.public_job(job)["error"]["errorCode"] == "WORKER_INTERRUPTED"
        assert client.get("/health/ready").status_code == 200
        with pytest.raises(RuntimeError):
            with worker_lock(settings.runtime_dir):
                pass
    finally:
        worker.stop.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert client.get("/health/ready").status_code == 503


def test_real_dependency_missing(settings):
    """Chưa có dependency/file thì fail rõ; không quay về mock thành công."""
    settings.ai_mode = "real"
    settings.deepfashion_weights = settings.runtime_dir / "missing.pt"
    with pytest.raises(ServiceError) as exc:
        Vision(settings).load()
    assert exc.value.code == "MODEL_NOT_READY"


def test_mode_mismatch(settings):
    """Worker real/mock không được xử lý nhầm job của chế độ khác."""
    with pytest.raises(ServiceError):
        Pipeline(settings).run({"payload": "{}", "simulated": 0})


def test_retention(settings, client, upload, tryon_payload):
    """Chỉ terminal job và asset hết hạn được xóa; ảnh input job queued được giữ."""
    unused = upload()["assetId"]
    job = client.post("/internal/v1/try-on/jobs", json=tryon_payload).json()
    with Store(settings.runtime_dir).connection() as db:
        db.execute("UPDATE assets SET created_at='2000-01-01T00:00:00+00:00'")
    preview = cleanup(settings, apply=False)
    assert preview["unreferencedExpiredAssets"] == 1
    assert client.get(f"/internal/v1/assets/{unused}").status_code == 200
    cleanup(settings, apply=True)
    assert client.get(f"/internal/v1/assets/{unused}").status_code == 404
    assert client.get(f"/internal/v1/assets/{tryon_payload['personAssetId']}").status_code == 200
    client.post(f"/internal/v1/jobs/{job['jobId']}/cancel")
    with Store(settings.runtime_dir).connection() as db:
        db.execute("UPDATE jobs SET updated_at='2000-01-01T00:00:00+00:00'")
    cleanup(settings, apply=True)
    assert client.get(f"/internal/v1/jobs/{job['jobId']}").status_code == 404


def test_worker_unexpected_error_is_sanitized(settings, client, tryon_payload, monkeypatch):
    """Không lộ path/secret của exception bất ngờ trong response job."""
    row = client.post("/internal/v1/try-on/jobs", json=tryon_payload).json()
    worker = Worker(settings)

    def fail(job):
        """Tiêm lỗi để kiểm thử error boundary, không phải giả lập test thành công."""
        raise RuntimeError("private-key-internal-path")

    monkeypatch.setattr(worker, "execute", fail)
    worker.run_one()
    response = client.get("/internal/v1/jobs/" + row["jobId"])
    assert response.json()["error"]["errorCode"] == "INFERENCE_FAILED"
    assert "private-key" not in response.text
    assert not error_data(RuntimeError("secret"))["retryable"]
