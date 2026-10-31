# Hợp đồng AI nội bộ v1 — 25/10/2026

> **Cập nhật 30/10/2026:** AI runtime đã chuyển sang MySQL, thêm API địa chỉ JSON trong RAM và sửa role mapping theo tên. Làm theo [hướng dẫn migration, địa chỉ và Postman](../../YoFaSoShop_Docs/extra-info/migrationAndLocationGuide.md) trước khi chạy. Source mới chưa được nghiệm thu bằng build/HTTP/MySQL/GPU; các báo cáo PASS trước đây là kết quả baseline cũ.

Thiết kế trước khi hiện thực; OpenAPI được xuất vào `contracts/openapi.json`
và đối chiếu bằng test để ngăn thay đổi hợp đồng ngoài ý muốn.

- Prefix `/internal/v1`; JSON camelCase; timestamp UTC ISO 8601.
- `X-Internal-API-Key` bắt buộc với mọi endpoint nội bộ và asset.
- Không xử lý đơn hàng, đăng nhập khách hay tính điểm thẩm mỹ nghiệp vụ tại AI.
- Chỉ nhận asset ID đã upload; không nhận URL/path tùy ý (tránh SSRF/path traversal).
- `POST /assets` multipart `file` + `role=person|garment|reference`.
- `GET /assets/{assetId}` metadata; `GET .../content` file có bảo vệ;
  `DELETE ...` chỉ asset không bị job tham chiếu.
- `POST /persons/validate`: assetId, category/subtype/length tùy chọn.
- `POST /routing/preview`: category, subtype, length, frame; không cần model.
- `POST /products/analyze`: analysisId, productId, productAssetIds,
  referenceAssetIds, categoryOverride tùy chọn; trả job 202.
- `POST /try-on/jobs`: sessionId, productId, variantId, personAssetId,
  garmentAssetId, vtonCategory, garmentSubtype, garmentLength, seed, steps, samples.
- `GET /jobs`: limit/offset/status; `GET /jobs/{jobId}`; `POST .../cancel`;
  `POST .../retry` chỉ job lỗi cho phép retry; `GET /try-on/jobs/{sessionId}`.
- `GET /capabilities`, `/models`, `/config-check`; `/health/live`, `/health/ready`.
- POST job dùng ID nghiệp vụ để chống lặp (atomic MySQL transaction): cùng ID
  và payload trả job cũ; payload khác 409. Job queued trước khi trả 202.
- Trạng thái QUEUED/PROCESSING/COMPLETED/FAILED/CANCELLED; mapping MySQL do BE quản lý.
- Lỗi chung: errorCode/message/details/correlationId/retryable, status HTTP đúng.
- Rate limit có Limit/Remaining/Reset và Retry-After; riêng giới hạn queue.
- MySQL bền vững + worker một process là cấu hình đầu tiên; API/worker dùng chung runtime ảnh.
  Claim atomic và khóa worker bảo đảm một job GPU. Không có task in-memory giả bền.
- API và worker là hai tiến trình; worker có thể chạy embedded chỉ với mock để test.
- Chế độ mock chỉ development/test, mọi output có simulated=true và cảnh báo,
  không trả ảnh người gốc như ảnh VTON thật. Production từ chối mock.
- Chế độ real lazy-load YOLO/OOTDiffusion; thiếu dependency/checkpoint thì lỗi rõ,
  không tự fallback sang mock hoặc sai category.
- So sánh chất lượng candidate trước khi publish; all-fail -> FAILED.
- Retention chỉ xóa asset/job đã hết hạn và không đang xử lý; không có xóa kho hàng.

Các endpoint được thêm trong bản đầu chưa có consumer production. Những thay đổi
sau này phải cập nhật contract/changelog và test; không đổi âm thầm kiểu dữ liệu.
