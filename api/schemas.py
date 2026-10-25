"""Hợp đồng camelCase: AI trả đặc trưng, BE quyết định điểm/kho/quyền người dùng."""
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic.alias_generators import to_camel

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
Category = Literal["upper_body", "lower_body", "dress"]
Frame = Literal["full_body", "three_quarter", "half_body", "invalid"]
Length = Literal["SHORT", "MIDI", "LONG", "UNKNOWN"]


class DTO(BaseModel):
    """Tên field Python snake_case, giao tiếp JSON camelCase, từ chối field viết nhầm."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class Status(str, Enum):
    """Vòng đời kỹ thuật; Backend ánh xạ sang enum PhienThuDoAo."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RouteRequest(DTO):
    """Preview rule không load model; metadata phải lấy từ catalog đã duyệt."""
    vton_category: Category
    person_frame: Frame
    garment_subtype: str | None = Field(default=None, max_length=64)
    garment_length: Length = "UNKNOWN"


class PersonRequest(DTO):
    """Kiểm tra ảnh người trước khi cấp lượt diffusion."""
    asset_id: Identifier
    vton_category: Category | None = None
    garment_subtype: str | None = Field(default=None, max_length=64)
    garment_length: Length = "UNKNOWN"


class ProductRequest(DTO):
    """Một lần phân tích: một mẫu sản phẩm, nhiều ảnh biến thể và ảnh mẫu."""
    analysis_id: Identifier
    product_id: Identifier
    product_asset_ids: list[Identifier] = Field(min_length=1, max_length=8)
    reference_asset_ids: list[Identifier] = Field(default_factory=list, max_length=8)
    category_override: Category | None = None


class TryOnRequest(DTO):
    """Tạo job VTON idempotent theo sessionId do Backend cấp."""
    session_id: Identifier
    product_id: Identifier
    variant_id: Identifier
    person_asset_id: Identifier
    garment_asset_id: Identifier
    vton_category: Category
    garment_subtype: str | None = Field(default=None, max_length=64)
    garment_length: Length = "UNKNOWN"
    seed: int = Field(default=42, ge=0, le=2147483647)
    steps: int = Field(default=20, ge=10, le=50)
    samples: int = Field(default=4, ge=1, le=4)


class ErrorResponse(DTO):
    """Một cấu trúc lỗi duy nhất cho toàn bộ HTTP API."""
    error_code: str
    message: str
    details: dict = Field(default_factory=dict)
    correlation_id: str
    retryable: bool = False


class AssetResponse(DTO):
    """Metadata an toàn, không lộ đường dẫn file nội bộ."""
    asset_id: str
    role: str
    width: int
    height: int
    bytes: int
    sha256: str
    created_at: str
    content_url: str
    simulated: bool = False


class JobResponse(DTO):
    """Snapshot job, kết quả chỉ có khi hoàn tất và mọi output có cờ simulated."""
    job_id: str
    kind: str
    business_id: str
    status: Status
    created_at: str
    updated_at: str
    attempts: int
    simulated: bool
    result: dict | None = None
    error: dict | None = None

