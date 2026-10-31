"""Pipeline thật/giả lập; không bao giờ fallback giả lập khi model thật lỗi."""
import os
import sys
import types

import numpy as np
from PIL import Image, ImageDraw

from api.core.errors import ServiceError
from api.core.store import Store
from api.services.assets import Assets
from api.services.routing import category_for, normalize_label, route
from api.services.vision import Vision, VISION_LOCK


class Pipeline:
    """Một instance trong worker; cache detector và một nhánh diffusion để hạn VRAM."""
    def __init__(self, settings):
        """Chuẩn bị storage/detector adapter, chưa nạp diffusion cho tới job thật."""
        self.settings = settings
        self.store = Store(settings)
        self.assets = Assets(settings, self.store)
        self.vision = Vision(settings)
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
        binary = np.asarray(mask, dtype=bool)
        if not binary.any() or binary.all() or binary.shape != (image.height, image.width):
            raise ServiceError("MASK_INVALID", "Mask cần có vùng nền/garment và cùng kích thước ảnh.")
        rgba = np.asarray(image.convert("RGBA")).copy()
        rgba[:, :, 3] = binary.astype(np.uint8) * 255
        ys, xs = np.where(binary)
        cutout = Image.fromarray(rgba).crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        # Chừa biên trong suốt để PNG luôn thể hiện được alpha khi garment hình chữ nhật.
        padded = Image.new("RGBA", (cutout.width + 4, cutout.height + 4))
        padded.paste(cutout, (2, 2))
        simulated = self.settings.ai_mode == "mock"
        return (self.assets.save(padded, "cutout", simulated, job_id),
                self.assets.save(Image.fromarray(binary.astype(np.uint8) * 255), "mask", simulated, job_id))

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
        person = self.assets.open(payload["personAssetId"], {"person"}).convert("RGB")
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
        with VISION_LOCK:
            self.vision.load()
            detection, ref_mask = self.vision.garment(garment.convert("RGB"), payload["vtonCategory"])
            target = normalize_label(payload.get("garmentSubtype") or detection["label"])
            work = self.settings.runtime_dir / "jobs" / job_id
            work.mkdir(parents=True, exist_ok=True)
            source_path = work / "source.png"
            person.save(source_path)
            padded_path = work / "prepared.png"
            _, metadata = q.prepare_oot_person(source_path, padded_path)
            prepared = Image.open(padded_path).convert("RGB")
            garment = garment.convert("RGBA")
            canvas = Image.new("RGBA", garment.size, "white")
            garment_input = Image.alpha_composite(canvas, garment).convert("RGB").resize((768, 1024))
            self.load_ootd(decision["modelType"])
            from utils_ootd import get_mask_location
            pose_data = self.vision.primitives.best_pose(prepared)
            if pose_data is None:
                raise ServiceError("PERSON_FRAME_INVALID", "Không xác định được pose trên ảnh đã chuẩn bị.")
            pose_context = {"box": pose_data["box"], "keypoints": pose_data["xy"], "confidences": pose_data["conf"]}
            mask_category = {"upper_body": "upper_body", "lower_body": "lower_body", "dress": "dresses"}[payload["vtonCategory"]]
            parsed, _ = self.parsing(prepared.resize((384, 512)))
            mask, gray = get_mask_location(decision["modelType"], mask_category, parsed, self.openpose(prepared.resize((384, 512))))
            mask = mask.resize((768, 1024), Image.Resampling.NEAREST)
            gray = gray.resize((768, 1024), Image.Resampling.NEAREST)
            with torch.inference_mode():
                outputs = self.ootd(model_type=decision["modelType"], category={"upper_body": "upperbody", "lower_body": "lowerbody", "dress": "dress"}[payload["vtonCategory"]],
                                    image_garm=garment_input, image_vton=Image.composite(gray, prepared, mask), mask=mask, image_ori=prepared,
                                    num_samples=payload["samples"], num_steps=payload["steps"], image_scale=self.settings.ootd_image_scale, seed=payload["seed"])
            candidates = []
            reports = []
            for index, image in enumerate(outputs):
                try:
                    candidate_detection, candidate_mask = self.vision.garment(image, payload["vtonCategory"])
                except ServiceError:
                    reports.append({"index": index, "passed": False, "reason": "GARMENT_NOT_FOUND"})
                    continue
                candidate_mask = candidate_mask.astype(np.uint8) * 255
                edit = np.maximum(np.asarray(mask), candidate_mask)
                outside = q.change_to_preservation_score(q.outside_garment_change_score(image, prepared, edit))
                protected = q.upper_body_protected_mask(image.size, pose_context, target)
                _, body = q.protected_region_artifact_score(image, prepared, protected)
                limb = q.calculate_limb_integrity_score(candidate_mask, pose_context, target)
                fit = q.garment_fit_score(candidate_mask, pose_context, payload["vtonCategory"], target)
                color = q.garment_color_similarity(garment.convert("RGB"), ref_mask.astype(np.uint8) * 255, image, candidate_mask)
                match, _ = q.candidate_match_quality(candidate_detection["label"], target, payload["vtonCategory"])
                passed = gate_candidate(payload["vtonCategory"], target, outside, body, limb, match)
                score = candidate_score(payload["vtonCategory"], color, outside, body, limb, fit, candidate_detection["confidence"])
                reports.append({"index": index, "passed": passed, "matchLevel": match, "outside": outside, "body": body, "limb": limb, "fit": fit, "color": color, "score": score})
                if passed:
                    composite = q.composite_preserving_source(image, prepared, edit, protected)
                    candidates.append((match, score, index, composite))
            if not candidates:
                raise ServiceError("QUALITY_GATE_FAILED", "Không candidate nào đạt kiểm tra category/tay/chân/vùng ngoài garment.", details={"candidates": reports})
            _, _, selected, result = max(candidates, key=lambda item: (item[0], item[1]))
            asset = self.assets.save(q.restore_oot_result(result, metadata), "result", False, job_id)
            return {"sessionId": payload["sessionId"], "route": decision, "resultAsset": asset, "simulated": False,
                    "quality": {"evaluated": True, "selectedIndex": selected, "candidates": reports},
                    "warnings": decision["warnings"], "scoringOwner": "Spring Boot"}


def gate_candidate(category, subtype, outside, body, limb, match):
    """Hard gate trước ranking; áo dài tay không bị phạt vì che cẳng tay đúng thiết kế."""
    if match <= 0 or not all(np.isfinite(v) for v in (outside, body, limb)):
        return False
    if outside < (.72 if category == "upper_body" else .65):
        return False
    if category == "dress" and body < .30:
        return False
    if not normalize_label(subtype).startswith("long_sleeved") and category in {"upper_body", "dress"}:
        return limb >= (.65 if category == "upper_body" else .75)
    return True


def candidate_score(category, color, outside, body, limb, fit, confidence):
    """Ranking chất lượng ảnh theo routingForTryon.md, khác điểm thẩm mỹ 40/40/20."""
    if category == "upper_body":
        return .25 * color + .35 * limb + .25 * fit + .15 * confidence
    if category == "lower_body":
        return .20 * color + .25 * outside + .25 * body + .25 * fit + .05 * confidence
    return .30 * color + .25 * outside + .20 * body + .20 * fit + .05 * confidence
