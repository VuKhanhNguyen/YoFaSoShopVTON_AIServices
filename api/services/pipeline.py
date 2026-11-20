"""Pipeline thật/giả lập: Admin dùng DeepFashion2+SAM2.1+Qwen3-VL; VTON dùng Cloud Provider (FASHN v1.6)."""
import json
import logging
import shutil

import numpy as np
from PIL import Image, ImageDraw

from api.core.errors import ServiceError
from api.core.store import Store
from api.services.assets import Assets, person_rgb
from api.services.routing import category_for, route
from api.services.vision import VISION_LOCK, Vision
from api.services.vton_providers import render_vton

logger = logging.getLogger("yofaso.pipeline")


class Pipeline:
    """Một instance trong worker; cache detector vision local và VLM cho Admin."""

    def __init__(self, settings):
        """Chuẩn bị storage, detector vision và VLM (Admin), không nạp OOTDiffusion."""
        self.settings = settings
        self.store = Store(settings)
        self.assets = Assets(settings, self.store)
        self.vision = Vision(settings)
        from api.services.vlm import VLMService

        self.vlm = VLMService(settings)

    def run(self, job: dict, progress_cb=None) -> dict:
        """Chọn tác vụ theo kind; cờ simulated xuyên suốt input/output và job."""
        payload = json.loads(job["payload"])
        if bool(job["simulated"]) != (self.settings.ai_mode == "mock"):
            raise ServiceError("MODE_MISMATCH", "Job và worker khác chế độ AI; không tự đổi mock/real.", 409)
        if job["kind"] == "PRODUCT_ANALYSIS":
            return self.product(payload, job["id"])
        return self.tryon(payload, job["id"], progress_cb=progress_cb)

    def mock_mask(self, image):
        """Mask hình học chỉ để test file/contract, không phải segmentation AI."""
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle(
            (image.width // 4, image.height // 5, 3 * image.width // 4, 4 * image.height // 5),
            fill=255,
        )
        return np.asarray(mask) > 0

    def visual_assets(self, image, mask, job_id):
        """Tạo mask 0/255 đúng size và RGBA crop từ cùng mask, giống contract notebook."""
        import cv2

        binary = np.asarray(mask, dtype=bool)
        if not binary.any() or binary.all() or binary.shape != (image.height, image.width):
            raise ServiceError("MASK_INVALID", "Mask cần có vùng nền/garment và cùng kích thước ảnh.")

        rgb = np.asarray(image.convert("RGB"))
        alpha = cv2.GaussianBlur(binary.astype(np.uint8) * 255, (5, 5), 0)
        rgba_full = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        rgba_full[:, :, :3] = rgb
        rgba_full[:, :, 3] = alpha

        ys, xs = np.where(binary)
        padding = max(8, int(round(0.03 * max(image.width, image.height))))
        x1 = max(0, int(xs.min()) - padding)
        x2 = min(image.width, int(xs.max()) + padding + 1)
        y1 = max(0, int(ys.min()) - padding)
        y2 = min(image.height, int(ys.max()) + padding + 1)
        cutout = Image.fromarray(rgba_full[y1:y2, x1:x2])

        mask_img = Image.fromarray(binary.astype(np.uint8) * 255)
        simulated = self.settings.ai_mode == "mock"
        return (
            self.assets.save(cutout, "cutout", simulated, job_id),
            self.assets.save(mask_img, "mask", simulated, job_id),
        )

    def product(self, payload, job_id):
        """Một sản phẩm nhiều biến thể: joint category, mask đồng bộ (DeepFashion2+SAM2), màu riêng, VLM."""
        ids = payload["productAssetIds"]
        images = [self.assets.open(aid, {"garment"}).convert("RGB") for aid in ids]
        simulated = self.settings.ai_mode == "mock"
        if simulated:
            category = payload.get("categoryOverride") or self.settings.mock_category
            masks = [self.mock_mask(image) for image in images]
            labels = ["short_sleeved_shirt" if category == "upper_body" else "shorts" if category == "lower_body" else "sling_dress"] * len(images)
            colors = [[{"hex": "#808080", "share": 1.0, "name": "MOCK"}] for _ in images]
            fit = {"code": "unknown", "confidence": "low", "body_position": None, "support": 0}
            quality = [{"status": "simulated"} for _ in images]
            consistency = {"applied": False, "skip_reason": "mock"}
        else:
            with VISION_LOCK:
                self.vision.load()
                p = self.vision.primitives
                products = []
                for aid, image in zip(ids, images):
                    candidates = p.fashion_candidates(image)
                    detection = p.choose_primary_garment(candidates)
                    if not detection:
                        raise ServiceError("GARMENT_NOT_FOUND", "Một ảnh sản phẩm không có garment hợp lệ.")
                    products.append({"original_name": aid, "path": self.assets.path(aid), "detection": detection, "candidates": candidates})
                references = []
                for aid in payload["referenceAssetIds"]:
                    image = self.assets.open(aid, {"reference"}).convert("RGB")
                    self.vision.validate_person(image)
                    references.append({"original_name": aid, "path": self.assets.path(aid), "pose": p.best_pose(image), "candidates": p.fashion_candidates(image)})
                p.CATEGORY_OVERRIDE = None
                label, _, resolution, _ = p.resolve_session_category(products, references)
                category = category_for(label)
                if payload.get("categoryOverride") and payload["categoryOverride"] != category:
                    raise ServiceError("CATEGORY_MISMATCH", "Category override khác dữ liệu ảnh; cần duyệt lại.")
                p.session_label, p.session_family = label, category
                references = p.finalize_model_items(references, label, category)
                masks = [p.detection_mask(item["detection"], image.size, image=image, refine_background=True) for item, image in zip(products, images)]
                if getattr(self.settings, "enable_sam2", True):
                    refined_masks = []
                    for item, image, mask in zip(products, images, masks):
                        bbox = item["detection"].get("box")
                        refined = self.vision.refine_garment_mask_sam(image, bbox, initial_mask=mask)
                        merged = refined if refined is not None else mask
                        if category == "dress" and bbox is not None:
                            bx1, by1, bx2, by2 = [float(v) for v in bbox]
                            bh = max(1.0, by2 - by1)
                            comp_t = p.choose_primary_garment([
                                d for d in item.get("candidates", [])
                                if d.get("label") == "trousers" and d.get("box") is not None
                                and float(d["box"][1]) >= by1 + 0.25 * bh
                                and min(bx2, float(d["box"][2])) > max(bx1, float(d["box"][0]))
                            ])
                            if comp_t is not None:
                                t_mask = p.detection_mask(comp_t, image.size, image=image, refine_background=True)
                                t_ref = self.vision.refine_garment_mask_sam(image, comp_t.get("box"), initial_mask=t_mask)
                                merged = np.logical_or(merged, t_ref if t_ref is not None else t_mask)
                        refined_masks.append(merged)
                    masks = refined_masks
                quality = [item["detection"].get("mask_quality", {}) for item in products]
                masks, consistency = p.synchronize_product_variant_masks(masks, quality, label=label)
                labels = [label] * len(images)
                colors = [p.extract_base_colors(image, mask, label) for image, mask in zip(images, masks)]
                fit = p.aggregate_reference_fit([p.analyze_model_reference(item) for item in references])
                if consistency.get("trouser_topology_source") == "mask_consensus" and category != "lower_body":
                    raise ServiceError("CATEGORY_REVIEW_REQUIRED", "Mask đa số cho thấy quần; cần duyệt category trước khi công bố.")
        vlm_attributes = None
        if not simulated and getattr(self.settings, "enable_vlm", True) and images:
            vlm_attributes = self.vlm.analyze_garment(images[0])
        variants = []
        for aid, image, mask, label, color, qa in zip(ids, images, masks, labels, colors, quality):
            cutout, asset_mask = self.visual_assets(image, mask, job_id)
            variants.append({
                "sourceAssetId": aid, "category": label, "vtonCategory": category,
                "colors": color, "maskQuality": qa, "cutout": cutout, "mask": asset_mask,
                "DuongDanAnhBocNen": cutout["contentUrl"], "DuongDanMatNaVungThayDo": asset_mask["contentUrl"],
            })
        return {
            "productId": payload["productId"], "vtonCategory": category, "variants": variants,
            "referenceFit": fit, "variantMaskConsistency": consistency, "simulated": simulated,
            **({"vlmAttributes": vlm_attributes} if vlm_attributes else {}),
            "warnings": ["MOCK: category, màu, mask và fit chỉ phục vụ kiểm thử."] if simulated else ["Metadata heuristic; chủ shop duyệt trước khi ghi catalog."],
        }

    def tryon(self, payload, job_id, progress_cb=None):
        """Validate pose (YOLO26-Pose) -> route -> Cloud VTON (FASHN v1.6) -> lưu ảnh PNG."""
        if progress_cb:
            progress_cb(8, "VALIDATING", "YOLO26-Pose đang kiểm tra tư thế & khung ảnh người...")
        raw_person = self.assets.open(payload["personAssetId"], {"person"})
        person = person_rgb(raw_person)
        garment = self.assets.open(payload["garmentAssetId"], {"garment"})
        validation = self.vision.validate_person(person)
        decision = route(
            payload["vtonCategory"],
            validation["personFrame"],
            payload.get("garmentSubtype"),
            payload["garmentLength"],
            provider=self.settings.vton_provider,
        )
        if not decision["accepted"]:
            raise ServiceError("PERSON_FRAME_INVALID", " ".join(decision["warnings"]))
        if self.settings.ai_mode == "mock":
            output = Image.new("RGB", person.size, "#ddddee")
            draw = ImageDraw.Draw(output)
            draw.text((12, 15), "MOCK - NOT A VTON RESULT", fill="red")
            draw.rectangle((30, 60, max(31, person.width - 30), max(61, person.height - 30)), outline="blue", width=3)
            asset = self.assets.save(output, "result", True, job_id)
            return {
                "sessionId": payload["sessionId"], "route": decision, "resultAsset": asset,
                "simulated": True, "quality": {"evaluated": False},
                "warnings": ["MOCK: chưa gọi Cloud VTON hoặc tạo ảnh thật."],
            }
        return self.real_tryon(person, garment, payload, job_id, decision, progress_cb=progress_cb)

    def real_tryon(self, person, garment, payload, job_id, decision, progress_cb=None):
        """Thử đồ thật qua Cloud VTON Provider (FASHN v1.6 / Kling), không qua OOTD/VLM QA."""
        def report(pct, stage, msg):
            if progress_cb:
                progress_cb(pct, stage, msg)

        report(20, "PREPROCESSING", "Đang chuẩn hóa ảnh người & trang phục tách nền...")
        garment_rgb, _, cutout_mask = prepare_garment_canvas(garment)
        work = self.settings.runtime_dir / "jobs" / job_id
        work.mkdir(parents=True, exist_ok=True)
        try:
            person_path = work / "person.png"
            garment_path = work / "garment.png"
            person.convert("RGB").save(person_path, "PNG")
            garment_rgb.save(garment_path, "PNG")

            metadata = {
                "sessionId": payload.get("sessionId"),
                "has_cutout": cutout_mask is not None,
                "seed": payload.get("seed", 42),
                "productName": payload.get("productName"),
                "material": payload.get("material"),
                "pattern": payload.get("pattern"),
                "colorName": payload.get("colorName"),
                "catalogFit": payload.get("catalogFit"),
                "catalogLength": payload.get("catalogLength"),
                "garmentSubtype": payload.get("garmentSubtype"),
            }
            output, provider_meta = render_vton(
                person_path,
                garment_path,
                payload["vtonCategory"],
                metadata,
                self.settings,
                progress_cb=progress_cb,
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

        report(92, "SAVING", "Đang lưu ảnh kết quả VTON...")
        asset = self.assets.save(output, "result", False, job_id)
        return {
            "sessionId": payload["sessionId"],
            "route": decision,
            "resultAsset": asset,
            "simulated": False,
            "provider": provider_meta,
            "quality": {
                "evaluated": True,
                "provider": provider_meta.get("provider"),
                "mode": provider_meta.get("mode"),
                "outputSize": [output.width, output.height],
            },
            "warnings": decision["warnings"],
            "referenceReviewed": False,
            "scoringOwner": "Spring Boot",
        }


def prepare_garment_canvas(garment: Image.Image) -> tuple[Image.Image, tuple[int, int, int], np.ndarray | None]:
    """Trải cutout RGBA lên nền trắng sạch có padding chuẩn 3:4 để gửi vào FASHN/Cloud VTON."""
    rgba = garment.convert("RGBA")
    arr = np.asarray(rgba).copy()
    alpha = arr[:, :, 3] > 127
    has_cutout = bool(0.05 < float(alpha.mean()) < 0.96)
    bg_rgb = (255, 255, 255)
    if has_cutout:
        if float((arr[:, :, 3] >= 240).mean()) > 0.05:
            arr[:, :, 3] = np.where(arr[:, :, 3] >= 230, 255, 0)
            alpha = arr[:, :, 3] > 127
        ys, xs = np.where(alpha)
        if len(xs) > 0:
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            bw, bh = x2 - x1, y2 - y1
            h, w = arr.shape[:2]
            if bw / max(1, w) > 0.78 or bh / max(1, h) > 0.78:
                occ = 0.84 if (bh / max(1, bw)) >= 1.70 else 0.72
                target_h = max(int(round(bh / occ)), int(round((bw / occ) * 4 / 3)))
                target_w = max(int(round(bw / occ)), int(round(target_h * 3 / 4)))
                padded = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
                crop = Image.fromarray(arr[y1:y2, x1:x2])
                ox, oy = (target_w - bw) // 2, (target_h - bh) // 2
                padded.paste(crop, (ox, oy))
                rgba = padded
                alpha = np.asarray(rgba)[:, :, 3] > 127
            else:
                rgba = Image.fromarray(arr)
    canvas = Image.new("RGBA", rgba.size, (*bg_rgb, 255))
    rgb = Image.alpha_composite(canvas, rgba).convert("RGB")
    return rgb, bg_rgb, (alpha if has_cutout else None)
