from pathlib import Path
import json


def export_reference():
    """Đồng bộ hướng dẫn từng API từ OpenAPI và Postman, không kết nối service."""
    repo = Path(__file__).resolve().parents[2]
    ai = repo / "YoFaSoShop_AIService"
    guide = repo / "YoFaSoShop_Docs/extra-info/apiTestingGuide.md"
    schema = json.loads((ai / "contracts/openapi.json").read_text("utf-8"))
    collection = json.loads((ai / "postman/YoFaSoShop_AI.postman_collection.json").read_text("utf-8"))
    sections = []
    for item in collection["item"]:
        req = item["request"]
        url = req["url"]
        method = req["method"]
        url = url if isinstance(url, str) else url["raw"]
        path = url.split("}}", 1)[1].split("?", 1)[0]
        matched = None
        for route, ops in schema["paths"].items():
            import re

            expression = re.sub(r"\{[^}]+\}", r"[^/]+", route)
            # Collection có {{id}}, regex route chỉ cần giữ nguyên mỗi segment.
            if re.fullmatch(expression, path) and method.lower() in ops:
                matched = (route, ops[method.lower()])
                break
        if not matched:
            raise ValueError(url)
        route, op = matched
        text = f"### AI-{len(sections) + 1:02d}. {method} {route}\n\n" + op.get("description", op.get("summary", "")) + "\n\n"
        text += (
            "Postman URL: `"
            + url
            + "`. "
            + (
                "Public, không cần key."
                if route.startswith("/health/")
                else "Header `X-Internal-API-Key: {{aiApiKey}}`; không dùng JWT khách thay cho key."
            )
            + "\n\n"
        )
        text += "Success: " + ", ".join(code for code in op["responses"] if code.startswith("2")) + ". "
        if "{asset_id}" in route:
            text += "assetId là UUID lấy từ POST /assets, không phải object key R2 trong SQL. "
        if "{job_id}" in route:
            text += "jobId lấy từ response submit job; khác business sessionId. "
        if "{session_id}" in route:
            text += "sessionId là business ID đã gửi khi tạo try-on, giúp tra lại sau timeout. "
        text += "\n"
        parameters = op.get("parameters", [])
        if parameters:
            text += (
                "\nParams: "
                + ", ".join("`" + p["name"] + "` (" + ("bắt buộc" if p.get("required") else "tùy chọn") + ")" for p in parameters)
                + ".\n"
            )
        if "body" in req:
            if req["body"]["mode"] == "raw":
                text += "\nBody → raw JSON (thay biến sau upload):\n\n```json\n" + req["body"]["raw"] + "\n```\n"
            else:
                text += "\nBody → form-data: role Text `person`/`garment`/`reference`, file File chọn ảnh thật. Postman tự tạo boundary. Muốn tạo VTON phải upload riêng person và garment để có đúng role; muốn ảnh mẫu chọn reference.\n"
        if route == "/internal/v1/jobs":
            text += "\nBật query status (QUEUED/PROCESSING/COMPLETED/FAILED/CANCELLED), limit 1..100, offset>=0 trong Params nếu muốn lọc. GET không chạy model.\n"
        if route == "/health/ready":
            text += (
                "\n503 khi worker chưa heartbeat/thiếu checkpoint; live=ok không thay cho ready hoặc nghiệm thu inference thật.\n"
            )
        if route.endswith("/cancel"):
            text += "\nChỉ queued được hủy; processing trả 409. Không giả vờ đã dừng GPU khi HTTP timeout.\n"
        if route.endswith("/retry"):
            text += "\nChỉ FAILED có error.retryable và chưa hết max attempts; sai điều kiện trả 409.\n"
        sections.append(text)
    s = guide.read_text("utf-8")
    marker = "<!-- AI_REFERENCE_BEGIN -->"
    end = "<!-- AI_REFERENCE_END -->"
    if marker in s:
        s = s.split(marker, 1)[0] + marker + "\n\n" + "\n".join(sections) + "\n" + end + s.split(end, 1)[1]
    else:
        s += "\n\n## 10. Phụ lục từng API FastAPI / Postman\n\n" + marker + "\n\n" + "\n".join(sections) + "\n" + end + "\n"
    guide.write_text(s, "utf-8")
    print(f"Wrote {len(sections)} AI endpoint instructions from contract/Postman.")


if __name__ == "__main__":
    export_reference()
