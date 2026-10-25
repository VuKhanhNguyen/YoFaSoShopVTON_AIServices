"""Regression CPU với ảnh/mask tổng hợp; không thay thế test checkpoint thực tế."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from api.core.errors import ServiceError
from api.services.vision import Vision
from api.vision import candidate_primitives as q
from api.vision import product_primitives as p


@pytest.mark.parametrize("label", ["sling", "shorts", "trousers", "skirt", "sling_dress"])
def test_mask_cleanup_category_cpu(label):
    """Mask cleanup không phụ thuộc tên ảnh/áo cố định, không mất kích thước ảnh."""
    image = Image.new("RGB", (256, 384), "white")
    image.paste("red", (70, 70, 190, 300))
    raw = np.zeros((384, 256), dtype=bool)
    raw[70:300, 70:190] = True
    mask, quality = p.refine_garment_mask(raw, label, image=image, refine_background=True)
    assert mask.shape == raw.shape and mask.any() and not mask.all()
    assert isinstance(quality, dict)
    colors = p.extract_base_colors(image, mask, label)
    assert colors and all(0 < c["share"] <= 1 for c in colors)


def test_letterbox_restore_preserves_size(tmp_path):
    """Người ngang/dọc không bị đổi output size do canvas 768x1024 của model."""
    original = Image.new("RGB", (300, 700), "red")
    original.save(tmp_path / "person.png")
    _, metadata = q.prepare_oot_person(tmp_path / "person.png", tmp_path / "prepared.png")
    with Image.open(tmp_path / "prepared.png") as prepared:
        restored = q.restore_oot_result(prepared, metadata)
    assert restored.size == original.size


@pytest.mark.parametrize("landmarks,frame", [([5, 6, 11, 12, 13, 14, 15, 16], "full_body"), ([5, 6, 11, 12, 13], "three_quarter"), ([0, 5, 6, 11, 12], "half_body")])
def test_real_person_rule_without_checkpoint(settings, landmarks, frame):
    """Tiêm output detector để kiểm tra rule thật, không gọi đây là model accuracy."""
    settings.ai_mode = "real"
    vision = Vision(settings)
    vision.fashion = object()  # chỉ bỏ bước nạp model; dùng chính rule production
    scores = np.zeros(17)
    scores[landmarks] = .9
    pose = {"conf": scores, "box": np.array([0, 0, 256, 384]), "confidence": .9}
    vision.primitives = SimpleNamespace(best_pose=lambda _: pose)
    result = vision.validate_person(Image.new("RGB", (256, 384)))
    assert result["personFrame"] == frame and not result["simulated"]
    pose["confidence"] = .1
    with pytest.raises(ServiceError):
        vision.validate_person(Image.new("RGB", (256, 384)))
