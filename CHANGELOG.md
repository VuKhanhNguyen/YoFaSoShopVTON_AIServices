# API changelog

## 1.0.0 — 2026-10-25

- Hợp đồng nội bộ đầu tiên: 18 operations, camelCase, API key và correlation ID.
- Asset private, ảnh hợp lệ/giới hạn upload, category/frame routing.
- Catalog analysis và VTON async jobs, SQLite durable queue, idempotency,
  poll, cancel/retry có policy, worker singleton + watchdog process.
- Mock development có cờ simulated; production cấm mock. Local OOTDiffusion
  adapter và primitive từ notebook đã tách, nhưng chưa nghiệm thu GPU.
- OpenAPI snapshot, Postman collection/environment, typed Python client và tests.

Chưa có consumer production. Thay đổi hợp đồng về sau cần phân loại additive/
breaking, cập nhật OpenAPI/test/docs, không âm thầm đổi field/type/status.
