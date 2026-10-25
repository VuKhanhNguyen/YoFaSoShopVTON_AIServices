"""Lỗi có mã ổn định và thông báo tiếng Việt cho BE/Postman."""


class ServiceError(Exception):
    """Lỗi dự kiến; details không chứa token, path server hoặc traceback."""

    def __init__(self, code: str, message: str, status: int = 422, *, retryable: bool = False, details=None):
        """Tách mã máy đọc được khỏi thông báo người dùng; không thêm secret vào details."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable
        self.details = details or {}

