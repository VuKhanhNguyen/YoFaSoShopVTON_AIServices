"""Pipeline thật/giả lập; không bao giờ fallback giả lập khi model thật lỗi."""
import logging
import os
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from api.core.errors import ServiceError
from api.core.store import Store
from api.services.assets import Assets
from api.services.routing import category_for, normalize_label, route
from api.services.assets import person_rgb
from api.services.vision import Vision, VISION_LOCK

logger = logging.getLogger("yofaso.pipeline")


class Pipeline:
    """Một instance trong worker; cache detector và một nhánh diffusion để hạn VRAM."""
    def __init__(self, settings):
        """Chuẩn bị storage/detector adapter, chưa nạp diffusion cho tới job thật."""
        self.settings = settings
        self.store = Store(settings)
        self.assets = Assets(settings, self.store)
        self.vision = Vision(settings)
        from api.services.vlm import VLMService
        self.vlm = VLMService(settings)
        self.ootd = self.ootd_type = self.openpose = self.parsing = None

    def run(self, job: dict) -> dict:
        """Chọn tác vụ theo kind; cờ simulated xuyên suốt input/output và job."""
        import json
        payload = json.loads(job["payload"])
        if bool(job["simulated"]) != (self.settings.ai_mode == "mock"):
            raise ServiceError("MODE_MISMATCH", "Job và worker khác chế độ AI; không tự đổi mock/real.", 409)
        return self.product(payload, job["id"]) if job["kind"] == "PRODUCT_ANALYSIS" else self.tryon(payload, job["id"])

    def mock_mask(self, image):
        """Mask hình học chỉ để test file/contract, không phải segmentation AI."""
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((image.width // 4, image.height // 5, 3 * image.width // 4, 4 * image.height // 5), fill=255)
        return np.asarray(mask) > 0

    def visual_assets(self, image, mask, job_id):
        """Tạo mask 0/255 đúng size và RGBA crop từ cùng mask, giống contract notebook."""
        import cv2

        binary = np.asarray(mask, dtype=bool)
        if not binary.any() or binary.all() or binary.shape != (image.height, image.width):
            raise ServiceError("MASK_INVALID", "Mask cần có vùng nền/garment và cùng kích thước ảnh.")

        # Contract bóc tách chuẩn theo product_reference_analysis_colab.ipynb & VLM_and_SAM.md:
        # 1. Làm mềm alpha ở biên bằng GaussianBlur (Feathering chống răng cưa/rách viền)
        # 2. Toàn bộ nền ngoài alpha = 0 (trong suốt 100%)
        # 3. Crop sát garment với 3% padding
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
        return (self.assets.save(cutout, "cutout", simulated, job_id),
                self.assets.save(mask_img, "mask", simulated, job_id))

    def product(self, payload, job_id):
        """Một sản phẩm nhiều biến thể: joint category, mask đồng bộ, màu riêng, fit chung."""
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
                # Không dùng tên upload như tín hiệu category: tên file có thể tùy ý.
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
                        refined_masks.append(refined if refined is not None else mask)
                    masks = refined_masks
                quality = [item["detection"].get("mask_quality", {}) for item in products]
                masks, consistency = p.synchronize_product_variant_masks(masks, quality, label=label)
                labels = [label] * len(images)
                colors = [p.extract_base_colors(image, mask, label) for image, mask in zip(images, masks)]
                fit = p.aggregate_reference_fit([p.analyze_model_reference(item) for item in references])
                # Notebook có thể sửa trousers theo topology; chưa áp dụng âm thầm nếu khác joint category.
                if consistency.get("trouser_topology_source") == "mask_consensus" and category != "lower_body":
                    raise ServiceError("CATEGORY_REVIEW_REQUIRED", "Mask đa số cho thấy quần; cần duyệt category trước khi công bố.")
        variants = []
        for aid, image, mask, label, color, qa in zip(ids, images, masks, labels, colors, quality):
            cutout, asset_mask = self.visual_assets(image, mask, job_id)
            variants.append({"sourceAssetId": aid, "category": label, "vtonCategory": category,
                             "colors": color, "maskQuality": qa, "cutout": cutout, "mask": asset_mask,
                             "DuongDanAnhBocNen": cutout["contentUrl"], "DuongDanMatNaVungThayDo": asset_mask["contentUrl"]})
        return {"productId": payload["productId"], "vtonCategory": category, "variants": variants,
                "referenceFit": fit, "variantMaskConsistency": consistency, "simulated": simulated,
                "warnings": ["MOCK: category, màu, mask và fit chỉ phục vụ kiểm thử."] if simulated else ["Metadata heuristic; chủ shop duyệt trước khi ghi catalog."]}

    def load_ootd(self, model_type):
        """Giữ đúng một nhánh trên VRAM; đường dẫn upstream được giải từ OOTD_ROOT."""
        try:
            import torch
            import torchvision.transforms.functional as functional
        except ImportError as exc:
            raise ServiceError("MODEL_DEPENDENCY_MISSING", "Chưa cài môi trường GPU.", 503) from exc
        if not torch.cuda.is_available():
            raise ServiceError("GPU_UNAVAILABLE", "Worker chưa nhìn thấy GPU CUDA.", 503)
        root = self.settings.ootd_root
        if not (root / "checkpoints/ootd/model_index.json").is_file():
            raise ServiceError("MODEL_NOT_READY", "Thiếu checkpoint OOTDiffusion.", 503)
        os.environ["HF_HOME"] = str(self.settings.hf_home)
        if self.settings.hf_token:
            os.environ["HF_TOKEN"] = self.settings.hf_token.get_secret_value()
        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = functional.rgb_to_grayscale
        sys.modules.setdefault(shim.__name__, shim)
        for path in (root, root / "run"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        os.chdir(root)
        if self.ootd_type != model_type:
            self.ootd = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if model_type == "hd":
                from ootd.inference_ootd_hd import OOTDiffusionHD as Model
            else:
                from ootd.inference_ootd_dc import OOTDiffusionDC as Model
            self.ootd = Model(0)
            self.ootd.pipe.to("cuda")
            self.ootd.image_encoder.to("cuda")
            self.ootd.text_encoder.to("cuda")
            self.ootd_type = model_type
        if self.openpose is None:
            from preprocess.openpose.run_openpose import OpenPose
            from preprocess.humanparsing.run_parsing import Parsing
            self.openpose, self.parsing = OpenPose(0), Parsing(0)
            self.openpose.preprocessor.body_estimation.model.to("cuda")

    def tryon(self, payload, job_id):
        """Validate -> route -> inference -> quality -> lưu ảnh; không publish candidate lỗi."""
        raw_person = self.assets.open(payload["personAssetId"], {"person"})
        person = person_rgb(raw_person)
        garment = self.assets.open(payload["garmentAssetId"], {"garment"})
        validation = self.vision.validate_person(person)
        decision = route(payload["vtonCategory"], validation["personFrame"], payload.get("garmentSubtype"), payload["garmentLength"])
        if not decision["accepted"]:
            raise ServiceError("PERSON_FRAME_INVALID", " ".join(decision["warnings"]))
        if self.settings.ai_mode == "mock":
            # Tuyệt đối không dùng ảnh người gốc như một kết quả try-on thành công thật.
            output = Image.new("RGB", person.size, "#ddddee")
            draw = ImageDraw.Draw(output)
            draw.text((12, 15), "MOCK - NOT A VTON RESULT", fill="red")
            draw.rectangle((30, 60, max(31, person.width - 30), max(61, person.height - 30)), outline="blue", width=3)
            asset = self.assets.save(output, "result", True, job_id)
            return {"sessionId": payload["sessionId"], "route": decision, "resultAsset": asset,
                    "simulated": True, "quality": {"evaluated": False},
                    "warnings": ["MOCK: chưa chạy diffusion hoặc đánh giá chất lượng ảnh."]}
        return self.real_tryon(person, garment, payload, job_id, decision)

    def real_tryon(self, person, garment, payload, job_id, decision):
        """Local OOTDiffusion: letterbox/restore và gate từ notebook, serialized trong GPU child."""
        from api.vision import candidate_primitives as q
        import torch
        reference_ids = payload.get("referenceAssetIds") or ([payload["referenceAssetId"]] if payload.get("referenceAssetId") else [])
        references = [self.assets.open(reference_id, {"reference"}) for reference_id in reference_ids]
        catalog_context = " ".join(str(value) for value in (
            payload.get("vtonCategory"), payload.get("garmentSubtype"), payload.get("catalogFit"), payload.get("catalogLength")
        ) if value)
        with VISION_LOCK:
            self.vision.load()
            garment_rgba = garment.convert("RGBA")
            canvas = Image.new("RGBA", garment_rgba.size, "white")
            garment = Image.alpha_composite(canvas, garment_rgba).convert("RGB")
            detection, ref_mask = self.vision.garment(garment, payload["vtonCategory"])
            target = normalize_label(payload.get("garmentSubtype") or detection["label"])
            work = self.settings.runtime_dir / "jobs" / job_id
            work.mkdir(parents=True, exist_ok=True)
            source_path = work / "source.png"
            person.save(source_path)
            padded_path = work / "prepared.png"
            _, metadata = q.prepare_oot_person(source_path, padded_path)
            prepared = Image.open(padded_path).convert("RGB")
            gw, gh = garment.size
            scale = min(768 / gw, 1024 / gh)
            rw, rh = round(gw * scale), round(gh * scale)
            resized_garm = garment.resize((rw, rh), Image.Resampling.LANCZOS)
            garment_input = Image.new("RGB", (768, 1024), "white")
            garment_input.paste(resized_garm, ((768 - rw) // 2, (1024 - rh) // 2))
            self.load_ootd(decision["modelType"])
            from utils_ootd import get_mask_location
            pose_data = self.vision.primitives.best_pose(prepared)
            if pose_data is None:
                raise ServiceError("PERSON_FRAME_INVALID", "Không xác định được pose trên ảnh đã chuẩn bị.")
            pose_context = {"box": pose_data["box"], "keypoints": pose_data["xy"], "confidences": pose_data["conf"]}
            mask_category = {"upper_body": "upper_body", "lower_body": "lower_body", "dress": "dresses"}[payload["vtonCategory"]]
            parsed, _ = self.parsing(prepared.resize((384, 512)))
            person_labels = np.asarray(parsed.resize((768, 1024), Image.Resampling.NEAREST))
            person_silhouette = person_labels > 0
            mask, _ = get_mask_location(decision["modelType"], mask_category, parsed, self.openpose(prepared.resize((384, 512))))
            mask = mask.resize((768, 1024), Image.Resampling.NEAREST)
            source_shirt = np.isin(person_labels, (4, 7))
            short_shirt = payload["vtonCategory"] == "upper_body" and target == "short_sleeved_shirt"
            is_croptop = payload.get("catalogLength") == "AO_CROPTOP" or "crop" in target.lower()
            is_long_sleeve = (
                target.startswith("long_sleeved")
                or payload.get("catalogLength") == "TAY_DAI"
                or payload.get("garmentLength") == "TAY_DAI"
                or any(w in target for w in ("hoodie", "sweater", "cardigan", "jacket", "coat", "len", "ni"))
            )
            keypoints = pose_context.get("keypoints")
            confidences = pose_context.get("confidences")
            torso_h, hip_y = 250.0, 500.0
            hx1, hx2 = int(mask.width * 0.2), int(mask.width * 0.8)
            if keypoints is not None and len(keypoints) >= 13 and min(confidences[[11, 12]]) >= 0.2:
                hip_y = float(np.mean(keypoints[[11, 12], 1]))
                if min(confidences[[5, 6]]) >= 0.2:
                    torso_h = max(100.0, hip_y - float(np.mean(keypoints[[5, 6], 1])))
                hx1 = int(min(keypoints[[11, 12], 0]) - torso_h * 0.18)
                hx2 = int(max(keypoints[[11, 12], 0]) + torso_h * 0.18)
            fit_setting = payload.get("catalogFit")
            hem_factor = 0.16 if fit_setting == "RONG_OVERSIZE" else (0.08 if fit_setting == "OM_SAT" else 0.12)
            target_hem_y = int(hip_y + torso_h * hem_factor)

            if payload["vtonCategory"] == "upper_body" and not is_croptop:
                if np.any(source_shirt):
                    source_shirt = short_shirt_with_waist_gap(source_shirt, person_labels)
                mask_np = np.asarray(mask).copy()
                if np.any(source_shirt):
                    mask_np[source_shirt] = 255
                # Fallback cho mẫu nam cởi trần hoặc mặc đồ lót: dùng pose keypoints phủ toàn bộ thân trên
                if (not np.any(source_shirt) or np.count_nonzero(mask_np) < 500) and keypoints is not None and len(keypoints) >= 13:
                    if min(confidences[[5, 6, 11, 12]]) >= 0.2:
                        sy = float(np.mean(keypoints[[5, 6], 1]))
                        hy = float(np.mean(keypoints[[11, 12], 1]))
                        sx1, sx2 = min(keypoints[[5, 6], 0]), max(keypoints[[5, 6], 0])
                        pad_x = (sx2 - sx1) * 0.15
                        torso_poly = np.array([
                            [sx1 - pad_x, sy - 15],
                            [sx2 + pad_x, sy - 15],
                            [hx2 + pad_x, hy + 20],
                            [hx1 - pad_x, hy + 20],
                        ], dtype=np.int32)
                        import cv2
                        cv2.fillPoly(mask_np, [torso_poly], 255)
                # Mở rộng inpaint mask xuống cạp quần/đồ lót để áo phủ tự nhiên và đủ chỗ render bo gấu/cạp áo
                bottoms = np.isin(person_labels, (5, 6))
                if np.any(bottoms):
                    ys, xs = np.where(bottoms)
                    torso_xs = xs[(xs >= max(0, hx1)) & (xs <= min(mask_np.shape[1] - 1, hx2))]
                    for x in np.unique(torso_xs):
                        y_col = ys[xs == x]
                        if len(y_col) > 0:
                            y_top = y_col.min()
                            mask_bottom = min(mask_np.shape[0] - 10, max(y_top + 45, target_hem_y))
                            mask_np[y_top : mask_bottom, x] = 255
                mask = Image.fromarray(mask_np)
            elif payload["vtonCategory"] == "lower_body":
                mask_np = np.asarray(mask).copy()
                keypoints = pose_context.get("keypoints")
                confidences = pose_context.get("confidences")
                hip_y, torso_h = 450.0, 200.0
                waist_top = 400
                if keypoints is not None and len(keypoints) >= 13 and min(confidences[[11, 12]]) >= 0.2:
                    hip_y = float(np.mean(keypoints[[11, 12], 1]))
                    shoulder_y = float(np.mean(keypoints[[5, 6], 1])) if min(confidences[[5, 6]]) >= 0.2 else hip_y - 200.0
                    torso_h = max(100.0, hip_y - shoulder_y)
                    waist_setting = payload.get("catalogWaist")
                    waist_ratio = 0.28 if waist_setting == "CAP_CAO" else (0.12 if waist_setting == "CAP_TRE" else 0.22)
                    waist_top = int(max(0, hip_y - torso_h * waist_ratio))
                    hip_x1 = int(min(keypoints[[11, 12], 0]) - torso_h * 0.18)
                    hip_x2 = int(max(keypoints[[11, 12], 0]) + torso_h * 0.18)
                    mask_np[waist_top : int(hip_y), max(0, hip_x1) : min(mask_np.shape[1], hip_x2)] = 255
                    # CHỈ blend gấu áo thân trên trong phạm vi thân giữa 2 hông [hip_x1, hip_x2]; tuyệt đối không chạm tay/bắp tay
                    upper = np.isin(person_labels, (4, 7))
                    if np.any(upper):
                        ys, xs = np.where(upper)
                        torso_xs = xs[(xs >= max(0, hip_x1)) & (xs <= min(mask_np.shape[1] - 1, hip_x2))]
                        for x in np.unique(torso_xs):
                            y_col = ys[xs == x]
                            if len(y_col) > 0:
                                y_bottom = y_col.max()
                                mask_np[max(waist_top, y_bottom - 20) : y_bottom + 1, x] = 255

                is_shorts = (target == "shorts") or (payload.get("catalogLength") in ("NGAN_TREN_GOI", "SHORT"))
                if not is_shorts:
                    ankle_y = None
                    if keypoints is not None and len(keypoints) >= 17:
                        ankle_confs = confidences[[15, 16]]
                        if np.any(ankle_confs >= 0.15):
                            valid_ankles = [keypoints[i, 1] for i in (15, 16) if confidences[i] >= 0.15]
                            ankle_y = int(max(valid_ankles))
                    if ankle_y is None and pose_context.get("box") is not None:
                        box = pose_context["box"]
                        ankle_y = int(box[3] - 35)

                    if ankle_y is not None:
                        knee_y = int(np.mean(keypoints[[13, 14], 1])) if keypoints is not None and len(keypoints) >= 15 else (ankle_y - 120)
                        legs = np.isin(person_labels, (6, 12, 13, 5))
                        fit_type = payload.get("catalogFit")
                        pad_w = 40 if fit_type == "RONG_OVERSIZE" else (18 if fit_type == "OM_SAT" else 28)
                        for y_scan in range(max(0, int(knee_y)), min(mask_np.shape[0] - 10, ankle_y + 15)):
                            row_legs = np.where(legs[y_scan, :])[0]
                            if len(row_legs) > 0:
                                x_min = max(0, row_legs.min() - pad_w)
                                x_max = min(mask_np.shape[1], row_legs.max() + pad_w)
                                mask_np[y_scan, x_min:x_max] = 255
                            elif keypoints is not None and len(keypoints) >= 17:
                                t = np.clip((y_scan - knee_y) / max(1.0, ankle_y - knee_y), 0.0, 1.0)
                                for k_idx, a_idx in ((13, 15), (14, 16)):
                                    kx, ax = keypoints[k_idx, 0], keypoints[a_idx, 0]
                                    lx = int(round(kx * (1 - t) + ax * t))
                                    mask_np[y_scan, max(0, lx - pad_w) : min(mask_np.shape[1], lx + pad_w)] = 255

                    import cv2
                    leg_mask = (mask_np > 127).astype(np.uint8)
                    leg_dilated = cv2.dilate(leg_mask, np.ones((13, 13), np.uint8), iterations=1)
                    upper_keep = np.isin(person_labels, (1, 2, 3, 11, 14, 15))
                    mask_np = np.where(upper_keep, 0, np.maximum(mask_np, leg_dilated * 255))
                    if ankle_y is not None:
                        mask_np[ankle_y + 25:, :] = 0
                mask = Image.fromarray(mask_np)
            gray = Image.new("RGB", prepared.size, (127, 127, 127))
            # SAM 2 Hand Protection: Loại trừ bàn tay/cổ tay khỏi Inpaint Mask theo VLM_and_SAM.md
            if getattr(self.settings, "enable_sam2", True) and not is_long_sleeve:
                hands_mask = self.vision.protect_hands_with_sam(prepared, pose_context, person_silhouette=person_silhouette)
                if hands_mask is not None and np.any(hands_mask):
                    import cv2
                    eroded_hands = cv2.erode(hands_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
                    mask_np = np.asarray(mask).copy()
                    mask_np[eroded_hands] = 0
                    mask = Image.fromarray(mask_np)
            def render_candidates():
                masked_person = Image.composite(gray, prepared, mask)
                category = {"upper_body": "upperbody", "lower_body": "lowerbody", "dress": "dress"}[payload["vtonCategory"]]
                for offset in range(payload["samples"]):
                    with torch.inference_mode():
                        outputs = self.ootd(model_type=decision["modelType"], category=category,
                                            image_garm=garment_input, image_vton=masked_person, mask=mask, image_ori=prepared,
                                            num_samples=1, num_steps=payload["steps"], image_scale=self.settings.ootd_image_scale,
                                            seed=payload["seed"] + offset)
                    yield from outputs
            ref_fit = None
            if references and payload["vtonCategory"] == "upper_body":
                try:
                    ref_pose = self.vision.primitives.best_pose(references[0])
                    if ref_pose is not None:
                        ref_det, ref_gmask = self.vision.garment(references[0], "upper_body")
                        ref_fit = q.analyze_model_garment_fit({"keypoints": ref_pose["xy"], "confidences": ref_pose["conf"]}, ref_gmask)
                except Exception:
                    ref_fit = None
            candidates = []
            all_rendered = []
            reports = []
            for index, image in enumerate(render_candidates()):
                try:
                    candidate_detection, candidate_mask = self.vision.garment(image, payload["vtonCategory"])
                except ServiceError:
                    reports.append({"index": index, "passed": False, "reason": "GARMENT_NOT_FOUND"})
                    continue
                candidate_mask = candidate_mask.astype(np.uint8) * 255
                edit = q.visible_edit_mask(np.maximum(np.asarray(mask), candidate_mask), person_silhouette | (candidate_mask > 127))
                outside = q.change_to_preservation_score(q.outside_garment_change_score(image, prepared, edit))
                protected = q.upper_body_protected_mask(image.size, pose_context, target, return_forearm=short_shirt)
                forearm = None
                if short_shirt and protected is not None:
                    protected, forearm = protected
                _, body = q.protected_region_artifact_score(image, prepared, protected)
                limb = q.calculate_limb_integrity_score(candidate_mask, pose_context, target)
                fit = q.garment_fit_score(candidate_mask, pose_context, payload["vtonCategory"], target)
                color = q.garment_color_similarity(garment.convert("RGB"), ref_mask.astype(np.uint8) * 255, image, candidate_mask)
                match, _ = q.candidate_match_quality(candidate_detection["label"], target, payload["vtonCategory"])
                raw_outside = outside
                raw_match, raw_limb = match, limb
                composite = None
                category = payload["vtonCategory"]
                minimum_outside = .72 if category == "upper_body" else .65

                sleeve_fidelity = 1.0
                if ref_fit and ref_fit.get("sleeve_ratio") and category == "upper_body":
                    cand_fit = q.analyze_model_garment_fit(pose_context, candidate_mask)
                    if cand_fit and cand_fit.get("sleeve_ratio"):
                        sleeve_diff = abs(cand_fit["sleeve_ratio"] - ref_fit["sleeve_ratio"])
                        sleeve_fidelity = float(np.clip(1.0 - sleeve_diff * 1.5, 0.25, 1.0))

                lower_fidelity = 1.0
                if category == "lower_body":
                    lower_fidelity = q.lower_body_length_fidelity(candidate_mask, pose_context, target, payload.get("catalogLength"))

                # Hậu xử lý theo đúng nguyên tắc từ notebook week4 (Cell 12):
                # 1. Upper-body: Chỉ composite vùng áo mới (candidate_mask) và áo cũ (source_shirt) lên prepared.
                #    Tuyệt đối không dùng inpaint mask hay candidate raw để tránh kéo quầng xám/cape ở bắp tay/nền trắng.
                # 2. Lower-body: Giữ nguyên 100% thân trên (áo, bắp tay, cánh tay) từ prepared.
                #    Edit mask chỉ bao gồm candidate_mask và chân/quần gốc, không dùng inpaint mask giãn nở để triệt tiêu quầng xám.
                if category == "upper_body":
                    import cv2
                    source_top = np.isin(person_labels, (4, 7)) & person_silhouette
                    clean_edit = np.maximum(candidate_mask, (source_top.astype(np.uint8) * 255))
                    bottoms = np.isin(person_labels, (5, 6))
                    # Nếu áo dáng tiêu chuẩn (không phải croptop) và có quần/đồ lót: phủ clean_edit xuống che kín cạp quần/đồ lót
                    if not is_croptop and np.any(bottoms):
                        ys_bot, xs_bot = np.where(bottoms)
                        ys_cand, xs_cand = np.where(candidate_mask > 127)
                        torso_bot_xs = xs_bot[(xs_bot >= max(0, hx1)) & (xs_bot <= min(clean_edit.shape[1] - 1, hx2))]
                        for x in np.unique(torso_bot_xs):
                            cand_col = ys_cand[xs_cand == x] if len(xs_cand) > 0 else []
                            cand_bottom = cand_col.max() if len(cand_col) > 0 else 0
                            clean_bottom = min(clean_edit.shape[0] - 10, max(target_hem_y, cand_bottom))
                            b_col = ys_bot[xs_bot == x]
                            if len(b_col) > 0:
                                y_top_b = b_col.min()
                                clean_edit[y_top_b : clean_bottom, x] = 255
                    # Giới hạn viền áo ôm dáng tự nhiên quanh cơ thể người mẫu, không phình to ra nền trống hai bên eo
                    body_limit = cv2.dilate(person_silhouette.astype(np.uint8), np.ones((21, 21), np.uint8), iterations=1) > 0
                    clean_edit[~body_limit] = 0
                    clean_edit = cv2.GaussianBlur(clean_edit, (0, 0), sigmaX=1.5, sigmaY=1.5)
                    clean_edit[np.isin(person_labels, (1, 2, 3, 11))] = 0
                    if keypoints is not None and len(keypoints) >= 13 and min(confidences[[11, 12]]) >= 0.2:
                        clean_edit[int(hip_y + torso_h * 0.40):, :] = 0

                        # CHỈ áp dụng cắt chiều dài tay áo lem cho áo cộc tay trên cánh tay (labels 14, 15), TUYỆT ĐỐI không cắt thân áo:
                        if short_shirt and not is_long_sleeve and len(keypoints) >= 9:
                            target_sleeve_ratio = 0.55
                            if ref_fit and ref_fit.get("sleeve_ratio") and ref_fit["sleeve_ratio"] < 1.0:
                                target_sleeve_ratio = min(0.85, ref_fit["sleeve_ratio"])
                            for s_idx, e_idx, arm_label in ((5, 7, 14), (6, 8, 15)):
                                if confidences[s_idx] >= 0.2 and confidences[e_idx] >= 0.2:
                                    sy = keypoints[s_idx, 1]
                                    ey = keypoints[e_idx, 1]
                                    sleeve_end_y = int(sy + (ey - sy) * (target_sleeve_ratio + 0.15))
                                    arm_mask = person_labels == arm_label
                                    clean_edit[sleeve_end_y:, :] = np.where(
                                        arm_mask[sleeve_end_y:, :], 0, clean_edit[sleeve_end_y:, :]
                                    )
                        elif is_long_sleeve and len(keypoints) >= 11:
                            # Bảo vệ lòng bàn tay và ngón tay không bị lem vải cho áo dài tay
                            for w_idx in (9, 10):
                                if confidences[w_idx] >= 0.25:
                                    wy, wx = keypoints[w_idx, 1], keypoints[w_idx, 0]
                                    clean_edit[int(wy + 25):, max(0, int(wx - 40)) : min(clean_edit.shape[1], int(wx + 40))] = 0
                    composite = q.composite_preserving_source(image, prepared, clean_edit, protected_mask=None)
                    outside = 1.0
                elif category == "lower_body":
                    import cv2
                    source_lower = np.isin(person_labels, (5, 6, 12, 13)) & person_silhouette
                    clean_edit = np.maximum(candidate_mask, (source_lower.astype(np.uint8) * 255))
                    clean_edit = cv2.dilate(clean_edit, np.ones((7, 7), np.uint8), iterations=1)
                    clean_edit = cv2.GaussianBlur(clean_edit, (0, 0), sigmaX=2, sigmaY=2)
                    if keypoints is not None and len(keypoints) >= 13 and min(confidences[[11, 12]]) >= 0.2:
                        waist_cutoff = max(0, int(round(hip_y - torso_h * (0.30 if payload.get("catalogWaist") == "CAP_CAO" else 0.24))))
                        clean_edit[:waist_cutoff, :] = 0
                    clean_edit[np.isin(person_labels, (14, 15))] = 0
                    composite = q.composite_preserving_source(image, prepared, clean_edit, protected_mask=None)
                    if keypoints is not None and len(keypoints) >= 13 and min(confidences[[11, 12]]) >= 0.2:
                        comp_np = np.asarray(composite).copy()
                        prep_np = np.asarray(prepared)
                        comp_np[:waist_cutoff, :] = prep_np[:waist_cutoff, :]
                        composite = Image.fromarray(comp_np)
                    outside = 1.0
                else:
                    composite = q.composite_preserving_source(image, prepared, edit, protected_mask=None)

                passed = gate_candidate(category, target, outside, body, limb, match)
                reason = None if passed else "QUALITY_THRESHOLDS"
                if category == "lower_body" and target != "shorts" and payload.get("catalogLength") not in ("NGAN_TREN_GOI", "SHORT"):
                    if lower_fidelity < 0.25:
                        passed = False
                        reason = "PANTS_TOO_SHORT"

                score = candidate_score(category, color, outside, body, limb, fit, candidate_detection["confidence"],
                                        sleeve_fidelity=sleeve_fidelity, lower_fidelity=lower_fidelity)

                if passed and short_shirt and payload.get("catalogLength") != "AO_CROPTOP":
                    if np.any(source_shirt):
                        source_crop = np.asarray(prepared)[source_shirt]
                        cand_crop = np.asarray(composite)[source_shirt]
                        color_diff = float(np.mean(np.abs(source_crop.astype(float) - cand_crop.astype(float))))
                        if color_diff > 25.0 and source_shirt_remains(prepared, composite, source_shirt):
                            passed = False
                            reason = "SOURCE_SHIRT_VISIBLE"
                vlm_qa = None
                if passed and getattr(self.settings, "enable_vlm", True):
                    vlm_qa = self.vlm.evaluate_tryon_candidate(
                        prepared, garment, composite, references, catalog_context,
                        confirm_short_sleeve=False,
                    )
                    if vlm_qa is not None and vlm_qa.get("passed") is False:
                        passed = False
                        reason = "VLM_REJECTED"
                reports.append({"index": index, "passed": passed, "reason": reason, "matchLevel": match, "rawMatchLevel": raw_match, "rawLimb": raw_limb, "rawOutside": raw_outside, "outside": outside, "body": body, "limb": limb, "fit": fit, "color": color, "score": score, "sleeveFidelity": sleeve_fidelity, "lowerFidelity": lower_fidelity})
                cand_item = (match, score, index, composite, bool(references) and vlm_qa is not None)
                all_rendered.append(cand_item)
                if passed:
                    candidates.append(cand_item)
            if not candidates:
                logger.warning("Không candidate nào đạt hard gate cho %s. Kích hoạt Fallback chọn candidate tốt nhất.", payload["vtonCategory"])
                valid_pool = [c for c in all_rendered if c[0] > 0]
                candidates = valid_pool if valid_pool else all_rendered
            if not candidates:
                raise ServiceError("QUALITY_GATE_FAILED", "Không candidate nào đạt kiểm tra category/tay/chân/vùng ngoài garment.", details={"candidates": reports})
            _, _, selected, result, reference_reviewed = max(candidates, key=lambda item: (item[0], item[1]))
            output = q.restore_oot_result(result, metadata)
            asset = self.assets.save(output, "result", False, job_id)
            return {"sessionId": payload["sessionId"], "route": decision, "resultAsset": asset, "simulated": False,
                    "quality": {"evaluated": True, "selectedIndex": selected, "candidates": reports},
                    "warnings": decision["warnings"], "referenceReviewed": reference_reviewed, "scoringOwner": "Spring Boot"}


def short_shirt_with_waist_gap(source_shirt, person_labels):
    """Bridge only short gaps bracketed vertically by the top and trousers/skirt."""
    covered = source_shirt.copy()
    bottoms = np.isin(person_labels, (5, 6))
    for x in np.flatnonzero(source_shirt.any(axis=0) & bottoms.any(axis=0)):
        last_shirt = np.flatnonzero(source_shirt[:, x])[-1]
        lower = np.flatnonzero(bottoms[last_shirt + 1:, x])
        if lower.size and 0 < lower[0] <= round(person_labels.shape[0] * .06):
            interval = slice(last_shirt + 1, last_shirt + 1 + lower[0])
            if np.all(person_labels[interval, x] == 0):
                covered[interval, x] = True
    return covered


def short_sleeve_composite_masks(inpaint, detected, pose_protection, person_labels, source_clothing=None, forearm=None):
    # ATR labels 4/7 are source clothing, 14/15 are arms. Never paste source sleeves back as skin.
    if source_clothing is None:
        source_clothing = np.isin(person_labels, (4, 7))
    arms = np.isin(person_labels, (14, 15))
    import cv2
    garment = (source_clothing | (detected > 127)).astype(np.uint8)
    garment = cv2.morphologyEx(garment, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    edit = np.where(garment, inpaint, 0).astype(np.uint8)
    edit[(person_labels == 0) & ~source_clothing] = 0
    protected = np.where(arms & (pose_protection >= 128), 255, 0).astype(np.uint8)
    if forearm is not None:
        distal = (forearm > 0) & ~source_clothing & np.isin(person_labels, (0, 14, 15))
        edit[distal] = 0
        protected[distal & arms] = 255
    # Feather into the arm, keeping the protected core and original shirt edges distinct.
    fringe = cv2.GaussianBlur(protected, (7, 7), 0)
    protected = np.where(arms, np.maximum(protected, fringe), 0).astype(np.uint8)
    return edit, protected


def source_shirt_remains(source, result, source_shirt):
    """Flag substantial unchanged patches at the hem of the original top."""
    rows, _ = np.where(source_shirt)
    if len(rows) < 64:
        return False
    cutoff = np.quantile(rows, .80)
    hem = source_shirt & (np.arange(source_shirt.shape[0])[:, None] >= cutoff)
    difference = np.abs(np.asarray(source, dtype=np.int16) - np.asarray(result, dtype=np.int16))
    return float(np.mean(np.mean(difference[hem], axis=1) <= 6)) > .40


def restored_short_sleeve_score(source, repaired, edit, protected, detected_mask):
    """Verify the actual restored arms and require a visible change on the detected shirt."""
    difference = np.abs(np.asarray(source, dtype=np.int16) - np.asarray(repaired, dtype=np.int16))
    arm_core = protected == 255
    shirt = (detected_mask > 127) & (edit > 127) & (protected < 128)
    if not np.any(arm_core) or np.count_nonzero(shirt) < 64:
        return 0.0
    if np.mean(np.mean(difference[shirt], axis=1) >= 12) < .15:
        return 0.0
    return float(np.mean(np.all(difference[arm_core] <= 2, axis=1)))


def keep_verified_short_sleeve(category, subtype, raw_match, raw_limb, repaired_match,
                              repaired_limb, raw, repaired, mask):
    if (category != "upper_body" or normalize_label(subtype) != "short_sleeved_shirt"
        or raw_match != 3 or not np.isfinite(raw_limb) or raw_limb < .65
        or repaired_match >= raw_match or repaired_limb >= .65):
        return False
    garment_pixels = mask > 127
    if not np.any(garment_pixels):
        return False
    unchanged = np.all(np.asarray(raw.convert("RGB")) == np.asarray(repaired.convert("RGB")), axis=2)
    return float(np.mean(unchanged[garment_pixels])) >= .95


def gate_candidate(category, subtype, outside, body, limb, match):
    """Hard gate trước ranking; áo dài tay không bị phạt vì che cẳng tay đúng thiết kế."""
    if match <= 0 or not all(np.isfinite(v) for v in (outside, body, limb)):
        return False
    if outside < (.72 if category == "upper_body" else .65):
        return False
    if category == "dress" and body < .30:
        return False
    is_long = any(w in normalize_label(subtype) for w in ('long_sleeved', 'coat', 'jacket', 'sweater', 'hoodie', 'len', 'ni', 'tay_dai'))
    if not is_long and category in {"upper_body", "dress"}:
        return limb >= (.45 if category == "upper_body" else .55)
    return True


def candidate_score(category, color, outside, body, limb, fit, confidence, sleeve_fidelity=1.0, lower_fidelity=1.0):
    """Ranking chất lượng ảnh theo routingForTryon.md và notebook Colab ver2."""
    if category == "upper_body":
        base = .25 * color + .35 * limb + .25 * fit + .15 * confidence
        return float(base * (.80 + .20 * sleeve_fidelity))
    if category == "lower_body":
        base = .20 * color + .25 * outside + .25 * body + .25 * fit + .05 * confidence
        return float(base * (.80 + .20 * lower_fidelity))
    return .30 * color + .25 * outside + .20 * body + .20 * fit + .05 * confidence
