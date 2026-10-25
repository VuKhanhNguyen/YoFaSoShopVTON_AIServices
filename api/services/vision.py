"""Adapter YOLO lazy-load; API mock không cần torch/ultralytics để khởi động."""
import threading


from api.core.errors import ServiceError
from api.services.routing import category_for, normalize_label

VISION_LOCK = threading.RLock()


class FashionProxy:
    """Chuẩn hóa tên nhãn checkpoint mà không đổi thuật toán notebook."""
    def __init__(self, model):
        """Bọc một model YOLO để chuẩn hóa nhãn cho helper notebook."""
        self.model = model

    def predict(self, *args, **kwargs):
        """Chuyển short sleeve top thành short_sleeved_shirt trước joint resolver."""
        outputs = self.model.predict(*args, **kwargs)
        for item in outputs:
            item.names = {k: normalize_label(v) for k, v in item.names.items()}
        return outputs


class Vision:
    """Giữ một bản model detector trong process; mọi inference được serialize."""
    def __init__(self, settings):
        """Giữ cấu hình; chưa import thư viện AI nặng lúc API khởi động."""
        self.settings = settings
        self.fashion = self.pose = self.primitives = None

    def load(self):
        """Không tự tải file theo tên; thiếu checkpoint/dependency báo lỗi rõ."""
        with VISION_LOCK:
            if self.fashion is not None:
                return
            if not self.settings.deepfashion_weights.is_file() or not self.settings.pose_weights.is_file():
                raise ServiceError("MODEL_NOT_READY", "Thiếu checkpoint DeepFashion2 hoặc YOLO Pose; kiểm tra .env.", 503)
            try:
                from ultralytics import YOLO
                from api.vision import product_primitives as primitives
            except ImportError as exc:
                raise ServiceError("MODEL_DEPENDENCY_MISSING", "Chưa cài thư viện vision trong môi trường đang chạy.", 503) from exc
            self.fashion = FashionProxy(YOLO(str(self.settings.deepfashion_weights)))
            self.pose = YOLO(str(self.settings.pose_weights))
            self.primitives = primitives
            primitives.fashion_model, primitives.pose_model = self.fashion, self.pose
            primitives.DEVICE = self.settings.vision_device

    def validate_person(self, image) -> dict:
        """Đòi người confidence>=.35, diện tích>=8%, >=5 điểm và thấy vai/hông."""
        if self.settings.ai_mode == "mock":
            return {"accepted": self.settings.mock_person_frame != "invalid", "personFrame": self.settings.mock_person_frame,
                    "simulated": True, "warnings": ["MOCK: frame từ cấu hình test; chưa kiểm tra người bằng model."]}
        with VISION_LOCK:
            self.load()
            pose = self.primitives.best_pose(image.convert("RGB"))
        if pose is None:
            raise ServiceError("PERSON_NOT_FOUND", "Không nhận diện được người; hãy tải ảnh người rõ vai và hông.")
        conf = pose["conf"]
        visible = conf >= 0.35
        box = pose["box"]
        area = (box[2] - box[0]) * (box[3] - box[1]) / (image.width * image.height)
        core = bool(visible[[5, 6]].any() and visible[[11, 12]].any())
        if pose["confidence"] < .35 or area < .08 or visible.sum() < 5 or not core:
            raise ServiceError("PERSON_FRAME_INVALID", "Ảnh cần thấy rõ vai, hông và ít nhất 5 khớp; người không quá nhỏ.")
        frame = "full_body" if visible[[13, 14]].any() and visible[[15, 16]].any() else "three_quarter" if visible[[13, 14]].any() else "half_body"
        return {"accepted": True, "personFrame": frame, "simulated": False, "warnings": []}

    def garment(self, image, category=None):
        """Trả detection+mask thật; category không khớp phải chặn trước diffusion."""
        with VISION_LOCK:
            self.load()
            detections = self.primitives.fashion_candidates(image.convert("RGB"))
            if category:
                detections = [d for d in detections if category_for(d["label"]) == category]
            best = self.primitives.choose_primary_garment(detections)
            if best is None:
                raise ServiceError("GARMENT_NOT_FOUND", "Không tìm thấy trang phục đúng vùng yêu cầu.")
            mask = self.primitives.detection_mask(best, image.size, image=image.convert("RGB"), refine_background=True)
        return best, mask
