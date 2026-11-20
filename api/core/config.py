"""Đọc .env một lần tại biên ứng dụng; mọi đường dẫn triển khai đổi ở đây."""
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ràng buộc chống phân mảnh bộ nhớ VRAM theo VLM_and_SAM.md
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Giới hạn thread BLAS/OpenMP/OpenCV/Rayon về 1 để tránh chạm cgroup pids.max (vd. 256 trên Vast.ai)
# gây lỗi pthread_create EAGAIN -> CUDA Error 304 / device busy khi chạy kèm uvicorn + worker.
for _thread_env in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "RAYON_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Cấu hình API/worker; không có khóa thật được ghi cứng trong source."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", env_file_encoding="utf-8")
    app_env: Literal["development", "test", "production"] = "development"
    ai_mode: Literal["mock", "real"] = "mock"
    ai_internal_api_key: SecretStr
    runtime_dir: Path = ROOT / "data/runtime"
    deepfashion_weights: Path = ROOT / "weights/deepfashion2_yolov8s-seg.pt"
    pose_weights: Path = ROOT / "weights/yolo26s-pose.pt"
    sam2_weights: Path = ROOT / "weights/sam2/sam2.1_l.pt"
    vlm_model_dir: Path = ROOT / "api/models/vlm/Qwen3-VL-4B-Instruct"
    enable_sam2: bool = True
    enable_vlm: bool = True
    vision_device: str = "cpu"
    hf_home: Path = ROOT / ".cache/huggingface"
    hf_token: SecretStr | None = None
    # Cloud VTON Provider (fashn_v16 | genlook | kling_o1 — đổi 1 dòng VTON_PROVIDER trong .env)
    vton_provider: Literal["fashn_v16", "kling_o1", "genlook"] = "fashn_v16"
    fal_key: SecretStr | None = None
    fashn_endpoint: str = "fal-ai/fashn/tryon/v1.6"
    kling_endpoint: str = "fal-ai/kling-image/o1"
    fashn_mode: Literal["performance", "balanced", "quality"] = "quality"
    fashn_garment_photo_type: Literal["auto", "flat-lay", "model"] = "auto"
    fashn_moderation_level: Literal["none", "permissive", "conservative"] = "permissive"
    fashn_segmentation_free: bool = True
    fashn_num_samples: int = Field(default=1, ge=1, le=4)
    # Genlook Try-On API (https://api.genlook.app/tryon/v1)
    genlook_api_key: SecretStr | None = None
    genlook_base_url: str = "https://api.genlook.app/tryon/v1"
    genlook_crop: bool = False
    genlook_resolution: Literal["default", "2k"] = "default"
    genlook_watermark: bool = False
    genlook_ai_label: bool = False
    genlook_keep_for_days: int = 1
    genlook_timeout_seconds: int = 120
    vton_download_timeout_seconds: int = 60
    embedded_worker: bool = False
    worker_poll_seconds: float = 0.25
    job_timeout_seconds: int = 300
    vton_job_timeout_seconds: int = 180
    worker_lease_seconds: int = 30
    max_job_attempts: int = 2
    max_queue_size: int = 20
    rate_limit_per_minute: int = 120
    max_upload_bytes: int = 60 * 1024 * 1024
    max_image_pixels: int = 60_000_000
    min_image_side: int = 128
    max_variants: int = 8
    retention_hours: int = 24
    mock_person_frame: Literal["full_body", "three_quarter", "half_body", "invalid"] = "full_body"
    mock_category: Literal["upper_body", "lower_body", "dress"] = "upper_body"
    expose_docs: bool = True

    def fal_key_value(self) -> str:
        """Lấy giá trị FAL_KEY dạng chuỗi đã trim, hỗ trợ cả SecretStr lẫn str."""
        if not self.fal_key:
            return ""
        raw = self.fal_key.get_secret_value() if isinstance(self.fal_key, SecretStr) else str(self.fal_key)
        return raw.strip()

    def has_fal_key(self) -> bool:
        """Kiểm tra FAL_KEY đã được điền thật (không rỗng và không phải placeholder)."""
        value = self.fal_key_value()
        return bool(value) and not value.startswith("CHANGE_ME")

    def genlook_key_value(self) -> str:
        """Lấy giá trị GENLOOK_API_KEY dạng chuỗi đã trim, hỗ trợ cả SecretStr lẫn str."""
        if not self.genlook_api_key:
            return ""
        raw = self.genlook_api_key.get_secret_value() if isinstance(self.genlook_api_key, SecretStr) else str(self.genlook_api_key)
        return raw.strip()

    def has_genlook_key(self) -> bool:
        """Kiểm tra GENLOOK_API_KEY đã được điền thật (không rỗng và không phải placeholder)."""
        value = self.genlook_key_value()
        return bool(value) and not value.startswith("CHANGE_ME")

    def has_vton_credentials(self) -> bool:
        """Kiểm tra API key tương ứng với VTON_PROVIDER đang chọn."""
        if self.vton_provider == "genlook":
            return self.has_genlook_key()
        return self.has_fal_key()

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
        for field in (
            "job_timeout_seconds", "vton_job_timeout_seconds", "vton_download_timeout_seconds",
            "genlook_timeout_seconds", "worker_lease_seconds", "max_queue_size",
            "rate_limit_per_minute", "max_upload_bytes", "max_image_pixels",
            "min_image_side", "retention_hours", "max_variants", "max_job_attempts",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} phải lớn hơn 0")
        if self.worker_poll_seconds <= 0:
            raise ValueError("worker_poll_seconds phải lớn hơn 0")
        if self.genlook_keep_for_days not in (1, 3, 7):
            raise ValueError("genlook_keep_for_days phải là 1, 3 hoặc 7")
        for field in ("runtime_dir", "deepfashion_weights", "pose_weights", "hf_home", "sam2_weights", "vlm_model_dir"):
            path = getattr(self, field).expanduser()
            setattr(self, field, (ROOT / path).resolve() if not path.is_absolute() else path.resolve())
        return self
