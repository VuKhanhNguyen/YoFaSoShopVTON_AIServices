"""Chạy Uvicorn thật trên localhost, test HTTP + mock job và đo tải nhẹ.

Không dùng ảnh khách, không cần GPU, không coi số đo này là SLA production.
"""
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Dựng server độc lập, kiểm tra HTTP, lưu report và luôn dừng process test."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    runtime = ROOT / "data" / ("runtime-http-" + uuid4().hex)
    runtime.mkdir(parents=True)
    fixtures = ROOT / "data/test-fixtures"
    fixtures.mkdir(exist_ok=True)
    for role, color in (("person", "lightblue"), ("garment", "salmon")):
        image = Image.new("RGB", (256, 384), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((55, 65, 200, 325), fill=color)
        draw.text((10, 12), "MOCK FIXTURE - " + role, fill="black")
        image.save(fixtures / f"{role}-demo.png")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(32)
    env = {**os.environ, "AI_INTERNAL_API_KEY": key, "AI_MODE": "mock", "APP_ENV": "test", "EMBEDDED_WORKER": "true", "RUNTIME_DIR": str(runtime), "RATE_LIMIT_PER_MINUTE": "2000"}
    log = (runtime / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, env=env,
                               stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    start = time.monotonic()
    statuses = []
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"X-Internal-API-Key": key}, timeout=20, trust_env=False) as client:
            for _ in range(100):
                try:
                    if client.get("/health/ready").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if process.poll() is not None:
                    raise RuntimeError("Server test không khởi động; xem server.log")
                time.sleep(.1)
            else:
                raise RuntimeError("Timeout chờ server test")
            prefix = "/internal/v1"
            assert client.get(prefix + "/models", headers={"X-Internal-API-Key": "wrong"}).status_code == 401
            ids = {}
            for role in ("person", "garment"):
                response = client.post(prefix + "/assets", data={"role": role}, files={"file": ("ten bat ky.png", (fixtures / f"{role}-demo.png").read_bytes(), "image/png")})
                assert response.status_code == 201, response.text
                ids[role] = response.json()["assetId"]
            validate = client.post(prefix + "/persons/validate", json={"assetId": ids["person"]})
            assert validate.status_code == 200 and validate.json()["simulated"]
            payload = {"sessionId": "HTTP001", "productId": "SP1", "variantId": "BT1", "personAssetId": ids["person"], "garmentAssetId": ids["garment"], "vtonCategory": "upper_body"}
            submitted = client.post(prefix + "/try-on/jobs", json=payload)
            assert submitted.status_code == 202, submitted.text
            replay = client.post(prefix + "/try-on/jobs", json=payload)
            assert replay.json()["jobId"] == submitted.json()["jobId"]
            for _ in range(100):
                completed = client.get(prefix + "/jobs/" + submitted.json()["jobId"]).json()
                if completed["status"] in {"COMPLETED", "FAILED"}:
                    break
                time.sleep(.05)
            assert completed["status"] == "COMPLETED", completed
            assert completed["result"]["simulated"]
            assert client.get(completed["result"]["resultAsset"]["contentUrl"]).status_code == 200

            def timed_request(_):
                """Đo latency từng request, không lấy batch time chia concurrency."""
                begin = time.perf_counter()
                response = client.get(prefix + "/capabilities")
                response.read()
                return response.status_code, (time.perf_counter() - begin) * 1000

            load_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=10) as pool:
                samples = list(pool.map(timed_request, range(100)))
            statuses = [item[0] for item in samples]
            durations = sorted(item[1] for item in samples)
            assert all(status == 200 for status in statuses)
            report = {"timestamp": datetime.now(timezone.utc).isoformat(), "mode": "mock", "simulated": True,
                      "httpFunctional": "PASS", "authentication": "PASS", "idempotency": "PASS", "jobLifecycle": "PASS",
                      "load": {"requests": 100, "concurrency": 10, "errors": statuses.count(500), "p50Ms": round(durations[49], 2), "p95Ms": round(durations[94], 2), "maxMs": round(durations[-1], 2), "elapsedSeconds": round(time.perf_counter() - load_start, 3)},
                      "elapsedSeconds": round(time.monotonic() - start, 2), "limitations": ["Synthetic images/mock; no GPU inference", "100 requests on local machine, not a production SLA or 10x-traffic capacity proof"]}
            (ROOT / "reports").mkdir(exist_ok=True)
            (ROOT / "reports/http_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == "__main__":
    main()
