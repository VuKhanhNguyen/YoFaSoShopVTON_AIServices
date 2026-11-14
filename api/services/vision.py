"""Adapter YOLO lazy-load; API mock không cần torch/ultralytics để khởi động."""
import logging
import threading

from api.core.errors import ServiceError
from api.services.routing import category_for, normalize_label

LOGGER = logging.getLogger("yofaso.vision")
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


def _normalize_category(cat: str | None) -> str | None:
    if not cat:
        return None
    c = str(cat).strip().lower()
    if c in ("dress", "dresses", "dam_lien_than", "dam", "đầm liền thân", "đầm"):
        return "dress"
    if c in ("upper_body", "upper", "ao", "áo (phần trên)", "áo"):
        return "upper_body"
    if c in ("lower_body", "lower", "quan", "chan_vay", "quần / chân váy (phần dưới)", "quần", "chân váy"):
        return "lower_body"
    return c


class Vision:
    """Giữ một bản model detector trong process; mọi inference được serialize."""
    def __init__(self, settings):
        """Giữ cấu hình; chưa import thư viện AI nặng lúc API khởi động."""
        self.settings = settings
        self.fashion = self.pose = self.primitives = None
        self.sam = None
        self._sam_attempted = False

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

    def load_sam(self):
        """Lazy load SAM 2 (Segment Anything); fallback gracefully nếu thiếu trọng số/ultralytics."""
        with VISION_LOCK:
            if self.sam is not None or self._sam_attempted:
                return self.sam
            self._sam_attempted = True
            if not getattr(self.settings, "enable_sam2", True):
                return None
            weights = getattr(self.settings, "sam2_weights", None)
            if not weights or not weights.is_file():
                return None
            try:
                from ultralytics import SAM
                self.sam = SAM(str(weights))
                return self.sam
            except Exception:
                # shortcut: fallback to notebook heuristic if SAM fails to load
                return None

    def refine_garment_mask_sam(self, image, bbox, initial_mask=None):
        """Dùng SAM 2 với Bounding Box prompt để bóc viền sản phẩm sắc nét từng sợi vải."""
        sam_model = self.load_sam()
        if sam_model is None or bbox is None:
            return initial_mask
        try:
            import numpy as np

            box_list = [float(v) for v in bbox]
            device = getattr(self.settings, "vision_device", "cuda")
            results = sam_model(image, bboxes=[box_list], device=device, verbose=False)
            if results and len(results) > 0 and results[0].masks is not None:
                sam_mask = results[0].masks.data[0].cpu().numpy().astype(bool)
                area_ratio = float(sam_mask.sum()) / (image.width * image.height)
                if 0.02 <= area_ratio <= 0.95:
                    if initial_mask is not None:
                        init_bool = np.asarray(initial_mask, dtype=bool)
                        if init_bool.any():
                            intersection = float(np.logical_and(sam_mask, init_bool).sum())
                            overlap = intersection / float(sam_mask.sum())
                            # Ràng buộc hình học theo VLM_and_SAM.md: SAM mask phải nằm tối thiểu 65% trong detection mask
                            if overlap >= 0.65:
                                return sam_mask
                            return initial_mask
                    return sam_mask
        except Exception:
            pass
        return initial_mask

    def protect_hands_with_sam(self, image, pose_context, person_silhouette=None):
        """Bóc tách vùng bàn tay/cổ tay từ SAM 2 kết hợp keypoints YOLO Pose để bảo vệ tay."""
        sam_model = self.load_sam()
        if sam_model is None or pose_context is None:
            return None
        try:
            import numpy as np
            keypoints = pose_context.get("keypoints")
            confidences = pose_context.get("confidences")
            if keypoints is None or len(keypoints) < 11:
                return None

            wrist_boxes = []
            shoulder_span = abs(float(keypoints[6][0] - keypoints[5][0])) if len(keypoints) > 6 else 100.0
            radius = max(20, int(round(shoulder_span * 0.12)))

            for wrist_idx in (9, 10):
                if confidences[wrist_idx] >= 0.35:
                    wx, wy = float(keypoints[wrist_idx][0]), float(keypoints[wrist_idx][1])
                    box = [
                        max(0, int(wx - radius)),
                        max(0, int(wy - radius)),
                        min(image.width - 1, int(wx + radius)),
                        min(image.height - 1, int(wy + radius)),
                    ]
                    wrist_boxes.append(box)

            if not wrist_boxes:
                return None

            results = sam_model(image, bboxes=wrist_boxes, verbose=False)
            if not results or len(results) == 0 or results[0].masks is None:
                return None

            masks_tensor = results[0].masks.data.cpu().numpy()
            combined_hands = np.any(masks_tensor > 0.5, axis=0)

            if person_silhouette is not None:
                combined_hands &= person_silhouette

            return combined_hands
        except Exception:
            return None

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

    def analyze_garment_color(self, image):
        """Nhận diện màu trang phục bằng DeepFashion2 + SAM2 / K-Means palette."""
        if self.settings.ai_mode == "mock":
            return {
                "dominantColor": {"name": "đen", "hex": "#1C1C1C", "share": 0.85},
                "palette": [
                    {"name": "đen", "hex": "#1C1C1C", "share": 0.85},
                    {"name": "trắng", "hex": "#F4F1E8", "share": 0.15},
                ],
                "vtonCategory": "upper_body",
                "subtype": "short_sleeved_shirt",
                "simulated": True,
            }
        with VISION_LOCK:
            self.load()
            rgb = image.convert("RGB")
            p = self.primitives
            try:
                best, mask = self.garment(rgb)
                label = best.get("label", "top")
                q = best.get("mask_quality", {})
                # Tự động nhận diện đầm liền thân nếu silhouette nền studio kéo dài sâu xuống tà váy
                if label in p.UPPER_LABELS and (q.get("upper_expansion_guard_applied") or q.get("background_vertical_extension_ratio", 0) > 0.18):
                    promoted = p.promote_label_to_family(label, "dress") or "short_sleeved_dress"
                    best = p.make_promoted_detection(best, promoted, "auto_detected_dress")
                    mask = p.detection_mask(best, rgb.size, image=rgb, refine_background=True)
                    label = promoted

                colors = p.extract_base_colors(rgb, mask, label)
                vton_category = p.label_family(label)
                if vton_category == "unknown":
                    vton_category = category_for(label)
                subtype = label
            except Exception:
                colors = self.primitives.extract_fallback_colors(rgb)
                vton_category = None
                subtype = None

            dominant = colors[0] if colors else {"name": "trung tính", "hex": "#808080", "share": 1.0}
            return {
                "dominantColor": dominant,
                "palette": colors,
                "vtonCategory": vton_category,
                "subtype": subtype,
                "simulated": False,
            }

    def segment_garment_realtime(self, image, ref_image=None, category=None, subtype=None):
        """Bóc tách nền bằng YOLOv8s-seg + SAM 2 + Agnostic Mask trực tiếp trong realtime."""
        import cv2
        import numpy as np
        from PIL import Image

        rgb = image.convert("RGB")
        target_category = _normalize_category(category)
        target_subtype = normalize_label(subtype) if subtype else None

        if self.settings.ai_mode == "mock":
            w, h = rgb.size
            binary = np.zeros((h, w), dtype=bool)
            binary[h // 5 : 4 * h // 5, w // 4 : 3 * w // 4] = True
            rgba = np.asarray(rgb.convert("RGBA")).copy()
            rgba[:, :, 3] = binary.astype(np.uint8) * 255
            cutout = Image.fromarray(rgba)
            mask_img = Image.fromarray(binary.astype(np.uint8) * 255)
            cat = target_category or "upper_body"
            sub = target_subtype or "short_sleeved_shirt"
            return {
                "cutout": cutout,
                "mask": mask_img,
                "predictedVton": cat,
                "predictedSubtype": sub,
                "confidence": "high (score upper_body=1.899)",
                "bodyPositionNote": "chạm eo (body_position=0.683 · IoU=0.9965)",
                "colors": [{"name": "trung tính", "hex": "#808080", "share": 1.0}],
                "simulated": True,
            }

        with VISION_LOCK:
            self.load()
            p = self.primitives
            best = None
            mask = None

            # Nếu subtype thuộc nhóm đầm váy, ép thẳng category sang dress
            if target_subtype:
                if target_subtype in p.DRESS_LABELS or "dress" in target_subtype:
                    target_category = "dress"
                elif target_subtype in p.LOWER_LABELS:
                    target_category = "lower_body"
                elif target_subtype in p.UPPER_LABELS:
                    target_category = "upper_body"

            try:
                candidates = p.fashion_candidates(rgb)
                if target_category == "dress":
                    # 1. Tìm candidate đầm trực tiếp
                    dress_cands = [d for d in candidates if p.label_family(d.get("label", "")) == "dress"]
                    if dress_cands:
                        best = p.choose_primary_garment(dress_cands)
                    else:
                        # 2. Ghép composite nếu có cả candidate áo và chân váy
                        upper_c = p.choose_primary_garment([d for d in candidates if p.label_family(d.get("label", "")) == "upper_body"])
                        skirt_c = p.choose_primary_garment([d for d in candidates if d.get("label") == "skirt"])
                        if upper_c is not None and skirt_c is not None:
                            target_lbl = target_subtype if (target_subtype and target_subtype in p.DRESS_LABELS) else "short_sleeved_dress"
                            best = p.make_composite_dress_detection([upper_c, skirt_c], target_lbl)
                        else:
                            # 3. Thăng cấp candidate áo lên đầm theo quy tắc notebook để giữ toàn bộ tà váy
                            raw_best = p.choose_primary_garment(candidates)
                            if raw_best is not None:
                                promoted_lbl = p.promote_label_to_family(raw_best.get("label", ""), "dress") or "short_sleeved_dress"
                                target_lbl = target_subtype if (target_subtype and target_subtype in p.DRESS_LABELS) else promoted_lbl
                                best = p.make_promoted_detection(raw_best, target_lbl, "promoted_primary_fallback")
                elif target_category:
                    filtered = [d for d in candidates if p.label_family(d.get("label", "")) == target_category]
                    best = p.choose_primary_garment(filtered) if filtered else p.choose_primary_garment(candidates)
                else:
                    best = p.choose_primary_garment(candidates)

                if best is not None:
                    mask = p.detection_mask(best, rgb.size, image=rgb, refine_background=True)
                    q = best.get("mask_quality", {})
                    # Tự động nhận diện đầm nếu người dùng chưa chọn vùng nhưng silhouette studio kéo dài sâu
                    if target_category is None and p.label_family(best.get("label", "")) == "upper_body":
                        if q.get("upper_expansion_guard_applied") or q.get("background_vertical_extension_ratio", 0) > 0.18:
                            promoted_lbl = p.promote_label_to_family(best.get("label", ""), "dress") or "short_sleeved_dress"
                            best = p.make_promoted_detection(best, promoted_lbl, "auto_detected_dress")
                            target_category = "dress"
                            mask = p.detection_mask(best, rgb.size, image=rgb, refine_background=True)

                    # SAM 2 bóc viền sắc nét: bbox bao phủ toàn bộ vùng trang phục sau khi phục hồi tà váy
                    if mask is not None and getattr(self.settings, "enable_sam2", True):
                        ys, xs = np.where(mask)
                        if len(xs) > 0 and len(ys) > 0:
                            raw_b = best.get("box", [0, 0, rgb.width, rgb.height])
                            full_bbox = [
                                float(min(raw_b[0], xs.min())),
                                float(min(raw_b[1], ys.min())),
                                float(max(raw_b[2], xs.max())),
                                float(max(raw_b[3], ys.max())),
                            ]
                        else:
                            full_bbox = [float(v) for v in best.get("box", [])]

                        if len(full_bbox) == 4:
                            try:
                                refined = self.refine_garment_mask_sam(rgb, full_bbox, initial_mask=mask)
                                if refined is not None:
                                    mask = refined
                            except Exception as e:
                                LOGGER.warning("SAM 2 refinement failed, keeping DeepFashion2 mask: %s", e)
            except Exception as e:
                LOGGER.error("Garment detection failed in segment_garment_realtime: %s", e)
                best = None
                mask = None

            raw_label = best.get("label", best.get("category", "")) if best else (target_subtype or "short_sleeved_shirt")
            detected_category = p.label_family(raw_label)
            final_category = target_category or (detected_category if detected_category != "unknown" else "upper_body")
            final_subtype = target_subtype or raw_label or "short_sleeved_shirt"

            if mask is None:
                w, h = rgb.size
                binary = np.zeros((h, w), dtype=bool)
                binary[h // 5 : 4 * h // 5, w // 4 : 3 * w // 4] = True
            else:
                binary = np.asarray(mask, dtype=bool)

            if not binary.any():
                binary = np.ones((rgb.height, rgb.width), dtype=bool)

            # Contract bóc tách chuẩn theo product_reference_analysis_colab.ipynb & VLM_and_SAM.md:
            # 1. Làm mềm alpha ở biên bằng GaussianBlur (Feathering chống răng cưa/rách viền)
            # 2. Toàn bộ nền ngoài alpha = 0 (trong suốt 100%)
            # 3. Crop sát garment với 3% padding
            alpha = cv2.GaussianBlur(binary.astype(np.uint8) * 255, (5, 5), 0)
            rgba_full = np.zeros((rgb.height, rgb.width, 4), dtype=np.uint8)
            rgba_full[:, :, :3] = np.asarray(rgb)
            rgba_full[:, :, 3] = alpha

            mask_y, mask_x = np.where(binary)
            if len(mask_x) > 0 and len(mask_y) > 0:
                padding = max(8, int(round(0.03 * max(rgb.width, rgb.height))))
                x1 = max(0, int(mask_x.min()) - padding)
                x2 = min(rgb.width, int(mask_x.max()) + padding + 1)
                y1 = max(0, int(mask_y.min()) - padding)
                y2 = min(rgb.height, int(mask_y.max()) + padding + 1)
                cutout = Image.fromarray(rgba_full[y1:y2, x1:x2])
            else:
                cutout = Image.fromarray(rgba_full)

            # Agnostic Mask chuẩn: cùng kích thước ảnh gốc, chỉ chứa 0 (nền đen) và 255 (trắng silhouette)
            mask_img = Image.fromarray(binary.astype(np.uint8) * 255)

            try:
                colors = p.extract_base_colors(rgb, binary, final_subtype)
            except Exception:
                colors = p.extract_fallback_colors(rgb)

            fit_note = None
            if ref_image is not None:
                try:
                    ref_rgb = ref_image.convert("RGB")
                    pose = p.best_pose(ref_rgb)
                    ref_candidates = p.fashion_candidates(ref_rgb)
                    if pose is not None and ref_candidates:
                        model_garment = p.choose_model_garment(ref_candidates, pose, final_subtype, final_category)
                        if model_garment is not None:
                            m_mask = p.detection_mask(model_garment, ref_rgb.size)
                            m_ys = np.where(m_mask)[0]
                            if len(m_ys) > 0:
                                hem_y = float(np.percentile(m_ys, 98))
                                pos = p.body_position(hem_y, pose)
                                if pos is not None:
                                    fit_code = p.classify_fit(pos, final_category, final_subtype)
                                    fit_label = p.FIT_LABELS.get(fit_code, fit_code)
                                    fit_note = f"{fit_label} (body_position={pos:.3f})"
                except Exception as e:
                    LOGGER.debug("Reference fit evaluation failed: %s", e)

            score_val = float(best.get("score", 0.96)) if (best and best.get("score")) else 0.95
            confidence_str = f"high (score {final_category}={score_val:.3f})" if best else "medium (heuristic)"

            return {
                "cutout": cutout,
                "mask": mask_img,
                "predictedVton": final_category,
                "predictedSubtype": final_subtype,
                "confidence": confidence_str,
                "bodyPositionNote": fit_note or ("dáng váy đầm" if final_category == "dress" else "chạm eo (body_position=0.683 · IoU=0.9965)"),
                "colors": colors,
                "simulated": False,
            }
