"""Giới hạn cả request chunked trước khi multipart parser ghi nhiều dữ liệu tạm."""
import json
from uuid import uuid4


class BodyTooLarge(Exception):
    """Tín hiệu nội bộ cho ASGI receive khi vượt giới hạn wire bytes."""


class BodyLimitMiddleware:
    """ASGI middleware đếm bytes kể cả khi caller bỏ Content-Length."""
    def __init__(self, app, max_bytes):
        """Giữ app phía sau và giới hạn tổng body, kể cả multipart overhead."""
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        """Chặn ở receive; 413 có cùng error contract và không lộ dữ liệu request."""
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0
        started = False

        async def limited_receive():
            """Tổng body gồm multipart overhead, không chỉ một file."""
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                raise BodyTooLarge()
            return message

        async def observed_send(message):
            """Không cố gửi status mới nếu app đã bắt đầu response."""
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, observed_send)
        except BodyTooLarge:
            if started:
                raise
            correlation = scope.get("state", {}).get("correlation_id", uuid4().hex)
            body = json.dumps({"errorCode": "UPLOAD_TOO_LARGE", "message": "Request vượt dung lượng cho phép.", "details": {}, "retryable": False, "correlationId": correlation}).encode()
            await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store"), (b"x-correlation-id", correlation.encode())]})
            await send({"type": "http.response.body", "body": body})
