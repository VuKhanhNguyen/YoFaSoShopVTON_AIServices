"""Kho ảnh private trên local volume, chuẩn hóa PNG và bảo vệ path/kích thước."""
import hashlib
import io
import warnings
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from api.core.errors import ServiceError
from api.core.store import now


class Assets:
    """Không lưu tên/path người dùng, không public StaticFiles chứa ảnh khách."""

    def __init__(self, settings, store):
        """Đặt kho ảnh bên trong runtime đã cấu hình, không nhận path từ caller."""
        self.settings, self.store = settings, store
        self.root = store.root / "assets"
        self.root.mkdir(exist_ok=True)

    def path(self, asset_id: str):
        """Sinh path chỉ từ ID đã tồn tại, không nối chuỗi path tùy ý."""
        self.store.asset(asset_id)
        return self.root / f"{asset_id}.png"

    def open(self, asset_id: str, roles: set | None = None):
        """Đọc ảnh theo role; file thiếu trên volume được báo lỗi có kiểm soát."""
        row = self.store.asset(asset_id)
        if roles and row["role"] not in roles:
            raise ServiceError("ASSET_ROLE_MISMATCH", "Ảnh không đúng vai trò yêu cầu.")
        try:
            with Image.open(self.root / f"{asset_id}.png") as image:
                return image.copy()
        except OSError as exc:
            raise ServiceError("ASSET_UNAVAILABLE", "File ảnh không còn trên storage.", 503) from exc

    def upload(self, data: bytes, role: str) -> dict:
        """Giới hạn byte/pixel trước decode; chỉ chấp nhận PNG/JPEG/WEBP thật."""
        if len(data) > self.settings.max_upload_bytes:
            raise ServiceError("UPLOAD_TOO_LARGE", "Ảnh vượt dung lượng cho phép.", 413)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as probe:
                    if probe.format not in {"PNG", "JPEG", "WEBP"} or getattr(probe, "n_frames", 1) != 1:
                        raise ServiceError("IMAGE_FORMAT_UNSUPPORTED", "Chỉ nhận PNG/JPEG/WEBP tĩnh.", 415)
                    if probe.width * probe.height > self.settings.max_image_pixels:
                        raise ServiceError("IMAGE_TOO_LARGE", "Ảnh có quá nhiều pixel.", 413)
                    if min(probe.size) < self.settings.min_image_side:
                        raise ServiceError("IMAGE_TOO_SMALL", f"Ảnh quá nhỏ; mỗi cạnh cần tối thiểu {self.settings.min_image_side} pixel.")
                    probe.verify()
                with Image.open(io.BytesIO(data)) as source:
                    image = ImageOps.exif_transpose(source).convert("RGBA" if "A" in source.getbands() or "transparency" in source.info else "RGB")
                    image.load()
        except ServiceError:
            raise
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ServiceError("IMAGE_INVALID", "File không phải ảnh hợp lệ hoặc dữ liệu ảnh bị hỏng.") from exc
        return self.save(image, role)

    def save(self, image, role: str, simulated: bool = False, job_id: str | None = None) -> dict:
        """Ghi PNG không EXIF, atomic replace; giữ output asset với job trong DB."""
        asset_id = uuid4().hex
        buffer = io.BytesIO()
        clean = image.copy()
        clean.info.clear()  # Không giữ EXIF/GPS/comment/profile từ ảnh khách.
        clean.save(buffer, "PNG")
        data = buffer.getvalue()
        path = self.root / f"{asset_id}.png"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        try:
            with self.store.connection() as db:
                db.execute("INSERT INTO assets VALUES(?,?,?,?,?,?,?,?)", (asset_id, role, image.width, image.height, len(data), hashlib.sha256(data).hexdigest(), now(), int(simulated)))
                if job_id:
                    db.execute("INSERT INTO job_assets VALUES(?,?)", (job_id, asset_id))
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return self.store.public_asset(self.store.asset(asset_id))

    def delete(self, asset_id: str):
        """Không xóa ảnh được input/output job tham chiếu, kể cả job đã hoàn tất."""
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM assets WHERE id=?", (asset_id,)).fetchone():
                raise ServiceError("ASSET_NOT_FOUND", "Không tìm thấy ảnh.", 404)
            if db.execute("SELECT 1 FROM job_assets WHERE asset_id=?", (asset_id,)).fetchone():
                raise ServiceError("ASSET_IN_USE", "Ảnh đang thuộc một job; dùng retention sau khi hết hạn.", 409)
            db.execute("DELETE FROM assets WHERE id=?", (asset_id,))
            (self.root / f"{asset_id}.png").unlink(missing_ok=True)
