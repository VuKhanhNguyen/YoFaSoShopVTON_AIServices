"""Đọc .env một lần tại biên ứng dụng; mọi đường dẫn triển khai đổi ở đây."""
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Cấu hình API/worker; không có khóa thật được ghi cứng trong source."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", env_file_encoding="utf-8")
    app_env: Literal["development", "test", "production"] = "development"
    ai_mode: Literal["mock", "real"] = "mock"
    ai_internal_api_key: SecretStr
    runtime_dir: Path = ROOT / "data/runtime"
    ootd_root: Path = ROOT / "api/models/ootdiffusion"
    deepfashion_weights: Path = ROOT / "weights/deepfashion2_yolov8s-seg.pt"
    pose_weights: Path = ROOT / "weights/yolo26s-pose.pt"
    vision_device: str = "cpu"
    hf_home: Path = ROOT / ".cache/huggingface"
    hf_token: SecretStr | None = None
    embedded_worker: bool = False
    worker_poll_seconds: float = 0.25
    job_timeout_seconds: int = 300
    worker_lease_seconds: int = 30
    max_job_attempts: int = 2
    max_queue_size: int = 20
    rate_limit_per_minute: int = 120
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    min_image_side: int = 128
    max_variants: int = 8
    retention_hours: int = 24
    mock_person_frame: Literal["full_body", "three_quarter", "half_body", "invalid"] = "full_body"
    mock_category: Literal["upper_body", "lower_body", "dress"] = "upper_body"
    expose_docs: bool = True

    @model_validator(mode="after")
    def validate_deployment(self):
        """Chặn cấu hình dễ làm lộ khóa, giả lập production hoặc worker GPU trong API."""
        key = self.ai_internal_api_key.get_secret_value()
        if len(key) < 32 or key.startswith("CHANGE_ME"):
            raise ValueError("Đặt AI_INTERNAL_API_KEY ngẫu nhiên tối thiểu 32 ký tự trong .env")
        if self.app_env == "production" and self.ai_mode == "mock":
            raise ValueError("Production không được dùng AI_MODE=mock")
        if self.embedded_worker and self.ai_mode != "mock":
            raise ValueError("EMBEDDED_WORKER chỉ dùng cho mock; model thật chạy worker riêng")
        for field in ("job_timeout_seconds", "worker_lease_seconds", "max_queue_size", "rate_limit_per_minute", "max_upload_bytes", "max_image_pixels", "min_image_side", "retention_hours", "max_variants", "max_job_attempts"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} phải lớn hơn 0")
        if self.worker_poll_seconds <= 0:
            raise ValueError("worker_poll_seconds phải lớn hơn 0")
        for field in ("runtime_dir", "ootd_root", "deepfashion_weights", "pose_weights", "hf_home"):
            path = getattr(self, field).expanduser()
            setattr(self, field, (ROOT / path).resolve() if not path.is_absolute() else path.resolve())
        return self

