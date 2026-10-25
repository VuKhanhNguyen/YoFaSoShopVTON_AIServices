"""ASGI unit test độc lập bảo vệ body chunked, không phụ thuộc multipart parser."""
import asyncio
import json

from api.core.body_limit import BodyLimitMiddleware


def test_body_limit_stream():
    """App đọc quá giới hạn phải nhận 413 theo error contract."""
    messages = iter([{"type": "http.request", "body": b"1234", "more_body": True}, {"type": "http.request", "body": b"5678", "more_body": False}])
    output = []

    async def receive():
        """Giả lập hai network chunks, không fake response thành công."""
        return next(messages)

    async def send(message):
        """Thu response để kiểm tra status và mã lỗi."""
        output.append(message)

    async def reader(scope, receive, send):
        """Đọc input giống parser phía sau middleware."""
        await receive()
        await receive()

    asyncio.run(BodyLimitMiddleware(reader, 5)({"type": "http", "state": {}}, receive, send))
    assert output[0]["status"] == 413
    assert json.loads(output[1]["body"])["errorCode"] == "UPLOAD_TOO_LARGE"
