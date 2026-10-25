# Kết quả kiểm thử AI Service — 25/10/2026

- Python thực thi: 3.10.21, venv cách ly `.venv-api`.
- Pytest: **63 test đạt**, 0 failures/errors; JUnit: `pytest.xml`.
- Statement coverage HTTP application: **95.4%**.
- Tổng statement coverage báo cáo: **72.8%**; bao gồm code real chưa chạy và script extract_weights cũ. Primitive snapshot/model bên thứ ba không tính trong tỷ lệ này.
- OpenAPI 3.1 validator: PASS; snapshot khớp app, 18 operations.
- Postman collection: 18 request từ cùng schema; environment không chứa khóa thật.
- Ruff (code ứng dụng/worker/client/script mới/tests): PASS.
- `pip check`: PASS, không có dependency conflict trong venv.
- Compose mock và override real: config --quiet PASS.
- Git LFS: git-lfs/3.7.1; chưa có `.git` để xác minh hook/index/remote hoặc push.

## HTTP Uvicorn thật trên localhost (MOCK)

- Auth, upload, validate, idempotency, submit/poll/result: PASS.
- Requests metadata: 100, concurrency 10, errors 0.
- p50 26.88 ms; p95 **39.14 ms**; max 49.76 ms.
- Dữ liệu ảnh tổng hợp, simulated=true. Đây không phải thời gian inference GPU hoặc benchmark capacity production.

## Nội dung test thêm

Input ảnh hỏng/quá nhỏ/quá byte/pixel; multipart/chunked limit; không giữ EXIF;
API key và asset private; fields/URL/path không cho phép; category/frame;
mask/cutout theo biến thể; idempotency atomic đồng thời; cancel/retry/queue đầy;
job persistence/restart; worker singleton; child process timeout/terminate;
retention không xóa queued input; CPU mask/color và letterbox/restore.

## Chưa xác minh

- Chưa chạy OOTDiffusion/YOLO checkpoint thật trên GPU, chưa đo VRAM hoặc model accuracy/fairness.
- Chưa build/run Docker image: docker CLI có nhưng Docker Engine Linux chưa chạy (named pipe không tồn tại).
- Chưa chạy BE/FE, S3 adapter, TLS triển khai hoặc scale nhiều host. SQLite queue chỉ dùng một local runtime.
- Test mô phỏng output detector chỉ xác minh rule; không phải chứng minh chất lượng model.
- Có cảnh báo dependency về coverage C tracer bị policy Windows chặn (đã chạy Python tracer) và deprecation AnyIO của Starlette; không có test fail.
