"""Xuất OpenAPI và Postman collection từ cùng ứng dụng, không load model."""
import json
import secrets
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.application import create_app  # noqa: E402
from api.core.config import Settings  # noqa: E402


def main():
    """Lưu contract đã validate và collection không chứa khóa thật."""
    from openapi_spec_validator import validate
    settings = Settings(_env_file=None, ai_internal_api_key=secrets.token_urlsafe(32),
                        runtime_dir=ROOT / "data/runtime-contract")
    schema = create_app(settings).openapi()
    validate(schema)
    (ROOT / "contracts").mkdir(exist_ok=True)
    (ROOT / "contracts/openapi.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    samples = {
        "/internal/v1/persons/validate": {"assetId": "{{personAssetId}}", "vtonCategory": "upper_body"},
        "/internal/v1/routing/preview": {"vtonCategory": "dress", "personFrame": "full_body", "garmentSubtype": "sling_dress", "garmentLength": "LONG"},
        "/internal/v1/products/analyze": {"analysisId": "{{analysisId}}", "productId": "SP00000001", "productAssetIds": ["{{garmentAssetId}}"], "referenceAssetIds": [], "categoryOverride": "upper_body"},
        "/internal/v1/try-on/jobs": {"sessionId": "{{sessionId}}", "productId": "SP00000001", "variantId": "BT00000001", "personAssetId": "{{personAssetId}}", "garmentAssetId": "{{garmentAssetId}}", "vtonCategory": "upper_body", "garmentSubtype": "sling", "garmentLength": "UNKNOWN", "seed": 42, "steps": 20, "samples": 1},
    }
    requests = []
    for path, ops in schema["paths"].items():
        for method, operation in ops.items():
            if method not in {"get", "post", "delete"}:
                continue
            rendered = path.replace("{asset_id}", "{{assetId}}").replace("{job_id}", "{{jobId}}").replace("{session_id}", "{{sessionId}}")
            headers = [{"key": "X-Correlation-ID", "value": "postman-demo"}]
            if path.startswith("/internal/"):
                headers.append({"key": "X-Internal-API-Key", "value": "{{aiApiKey}}"})
            request = {"method": method.upper(), "header": headers, "url": "{{aiBaseUrl}}" + rendered}
            if path == "/internal/v1/assets" and method == "post":
                request["body"] = {"mode": "formdata", "formdata": [{"key": "role", "value": "garment", "type": "text"}, {"key": "file", "type": "file", "src": ""}]}
            if path in samples:
                request["body"] = {"mode": "raw", "raw": json.dumps(samples[path], ensure_ascii=False, indent=2), "options": {"raw": {"language": "json"}}}
                headers.append({"key": "Content-Type", "value": "application/json"})
            script = ["pm.test('Không có lỗi server ngoài dự kiến', function () { pm.expect(pm.response.code).to.be.below(500); });"]
            if path == "/internal/v1/assets" and method == "post":
                script += ["if (pm.response.code === 201) { const a = pm.response.json(); pm.environment.set('assetId', a.assetId); if (a.role === 'person') pm.environment.set('personAssetId', a.assetId); if (a.role === 'garment') pm.environment.set('garmentAssetId', a.assetId); }"]
            if path in {"/internal/v1/config-check", "/internal/v1/capabilities"}:
                script.append("if(pm.response.code===200)pm.test('Queue uses manifest',()=>pm.expect(pm.response.json().queueBackend).to.equal('file-manifest'));")
            if path == "/internal/v1/jobs" and method == "get":
                request["url"] += "?status=QUEUED&limit=20&offset=0"
            if path in {"/internal/v1/products/analyze", "/internal/v1/try-on/jobs"}:
                script += ["if (pm.response.code === 202) pm.environment.set('jobId', pm.response.json().jobId);"]
            requests.append({"name": method.upper() + " " + path, "request": request, "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": script}}]})
    directory = ROOT / "postman"
    directory.mkdir(exist_ok=True)
    collection = {"info": {"description": "File-manifest runtime on a persistent single-host volume; no SQL/SQLite. Mock is simulated.", "name": "YoFaSoShop AI v1", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"}, "item": requests}
    (directory / "YoFaSoShop_AI.postman_collection.json").write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")
    environment = {"name": "YoFaSoShop local MOCK", "values": [{"key": key, "value": value, "enabled": True, "type": "secret" if key == "aiApiKey" else "default"} for key, value in {"aiBaseUrl": "http://127.0.0.1:8000", "aiApiKey": "", "assetId": "", "personAssetId": "", "garmentAssetId": "", "jobId": "", "sessionId": "TD_POSTMAN_001", "analysisId": "ANA_POSTMAN_001"}.items()], "_postman_variable_scope": "environment"}
    (directory / "YoFaSoShop_local.postman_environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    from export_reference import export_reference
    export_reference()
    print(f"OpenAPI validated; exported {len(requests)} requests. Không có khóa thật trong collection.")


if __name__ == "__main__":
    main()
