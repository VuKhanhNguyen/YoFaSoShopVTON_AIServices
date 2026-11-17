"""Quy tắc ranking là hàm thuần, không cần MySQL/GPU."""
import pytest
import numpy as np
from PIL import Image
from api.services.pipeline import candidate_score, gate_candidate, keep_verified_short_sleeve, restored_short_sleeve_score, short_sleeve_composite_masks, source_shirt_remains
from api.vision import candidate_primitives as q


def test_room_background_respects_letterbox_and_keeps_person_pixels():
    room = Image.new("RGB", (12, 8), "blue")
    person = Image.new("RGB", (4, 6), "red")
    parsed = np.zeros((8, 8), np.uint8)
    parsed[1:7, 3:5] = 4
    metadata = {"target_size": (8, 8), "content_box": (2, 1, 6, 7)}
    output = q.composite_room_background(person, Image.fromarray(parsed), metadata, room)
    assert output.size == person.size
    assert output.getpixel((0, 3)) == (0, 0, 255)
    assert output.getpixel((1, 3)) == (255, 0, 0)


def test_forearm_strip_is_removed_without_erasing_source_shirt_region():
    labels = np.array([[4, 14, 0, 14]], np.uint8)
    full = np.full(labels.shape, 255, np.uint8)
    distal = np.array([[255, 255, 255, 0]], np.uint8)
    edit, protected = short_sleeve_composite_masks(
        full, full, np.zeros_like(full), labels, forearm=distal,
    )
    assert edit.tolist() == [[255, 0, 0, 255]]
    assert protected[0, 1] == 255
    assert protected[0, 0] == 0
from api.services.pipeline import short_shirt_with_waist_gap


def test_waist_gap_preserves_pants_and_excludes_open_background():
    labels = np.zeros((100, 5), np.uint8)
    labels[20:50, 1:4] = 4
    labels[54:70, 2] = 6
    labels[60:70, 3] = 6
    covered = short_shirt_with_waist_gap(labels == 4, labels)
    assert covered[50:54, 2].all()
    assert not covered[54, 2]
    assert not covered[50:, 1].any()
    assert not covered[50:, 3].any()


def test_source_shirt_hem_must_be_replaced():
    source = Image.new("RGB", (30, 30), "white")
    shirt = np.zeros((30, 30), bool)
    shirt[2:25, 5:25] = True
    partial = source.copy()
    partial.paste("blue", (5, 2, 25, 18))
    assert source_shirt_remains(source, partial, shirt)
    partial.paste("blue", (5, 18, 25, 25))
    assert not source_shirt_remains(source, partial, shirt)


def test_short_sleeve_restores_skin_without_pasting_original_sleeve_or_background():
    source = Image.new("RGB", (3, 1), "white")
    candidate = Image.new("RGB", source.size, "blue")
    broad_mask = np.full((1, 3), 255, np.uint8)
    labels = np.array([[0, 4, 14]], np.uint8)
    edit, protected = short_sleeve_composite_masks(broad_mask, broad_mask, broad_mask, labels)
    output = q.composite_preserving_source(candidate, source, edit, protected)
    assert list(output.getdata()) == [(255, 255, 255), (0, 0, 255), (255, 255, 255)]


def test_public_composite_preserves_background_inside_inpaint_area():
    source = Image.new("RGB", (8, 8), "white")
    candidate = Image.new("RGB", source.size, "gray")
    inpaint = np.full((8, 8), 255, np.uint8)
    silhouette = np.zeros((8, 8), bool)
    silhouette[2:6, 2:6] = True
    output = q.composite_preserving_source(candidate, source, q.visible_edit_mask(inpaint, silhouette))
    assert output.getpixel((1, 2)) == source.getpixel((1, 2))
    assert output.getpixel((2, 2)) == candidate.getpixel((2, 2))


@pytest.mark.parametrize("category,expected", [("upper_body", .35), ("lower_body", 0), ("dress", 0)])
def test_ranking_limb_weight(category, expected):
    assert candidate_score(category, 0, 0, 0, 1, 0, 0) == pytest.approx(expected)


def test_long_sleeve_alias_does_not_trigger_short_sleeve_limb_gate():
    assert gate_candidate("dress", "long sleeve dress", .9, .9, .1, 3)
    assert not gate_candidate("dress", "sling dress", .9, .9, .1, 3)


def test_repair_recovers_original_pixels_without_erasing_garment():
    source = Image.new("RGB", (8, 8), (245, 245, 245))
    candidate = Image.new("RGB", (8, 8), (235, 235, 235))
    candidate.paste((20, 40, 80), (2, 2, 6, 6))
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    def passes(image, limb=.922):
        outside = q.change_to_preservation_score(q.outside_garment_change_score(image, source, mask))
        return gate_candidate("upper_body", "short_sleeved_shirt", outside, .003, limb, 3)
    assert not passes(candidate)
    repaired = q.composite_preserving_source(candidate, source, mask)
    assert repaired.getpixel((0, 0)) == source.getpixel((0, 0))
    assert repaired.getpixel((3, 3)) == candidate.getpixel((3, 3))
    assert passes(repaired)
    assert not passes(repaired, limb=.1)


def test_repair_keeps_detected_sleeve_beyond_inpaint_mask():
    source = Image.new("RGB", (8, 8), "white")
    candidate = Image.new("RGB", (8, 8), "blue")
    inpaint = np.zeros((8, 8), dtype=np.uint8)
    inpaint[2:6, 3:5] = 255
    garment = inpaint.copy()
    garment[2:4, 1:7] = 255
    protected = np.zeros_like(inpaint)
    protected[3, 1] = 255
    output = q.composite_preserving_source(candidate, source, np.maximum(inpaint, garment), protected)
    assert output.getpixel((1, 2)) == candidate.getpixel((1, 2))
    assert output.getpixel((1, 3)) == source.getpixel((1, 3))
    assert output.getpixel((0, 0)) == source.getpixel((0, 0))


def test_short_sleeve_redetection_can_disagree_only_if_verified_pixels_survive():
    raw = Image.new("RGB", (10, 10), "blue")
    mask = np.zeros((10, 10), np.uint8)
    mask[2:8, 2:8] = 255
    repaired = q.composite_preserving_source(raw, Image.new("RGB", raw.size, "white"), mask)
    args = ("upper_body", "short_sleeved_shirt", 3, .898, 1, 0)
    assert keep_verified_short_sleeve(*args, raw, repaired, mask)
    repaired.paste("white", (2, 2, 3, 8))
    assert not keep_verified_short_sleeve(*args, raw, repaired, mask)
    assert not keep_verified_short_sleeve("upper_body", "short_sleeved_shirt", 3, 0, 1, 0, raw, raw, mask)


def test_short_sleeve_recovery_requires_visible_shirt_and_original_arm_pixels():
    source = Image.new("RGB", (20, 20), "white")
    raw = Image.new("RGB", source.size, "blue")
    edit = np.full((20, 20), 255, np.uint8)
    protected = np.zeros_like(edit)
    protected[:, :5] = 255
    repaired = q.composite_preserving_source(raw, source, edit, protected)
    assert restored_short_sleeve_score(source, repaired, edit, protected, edit) == 1
    assert restored_short_sleeve_score(source, raw, edit, protected, edit) == 0
    assert restored_short_sleeve_score(source, source, edit, protected, edit) == 0


def test_oversized_short_sleeve_cuff_preserved_by_protected_mask():
    """Bảo vệ tay không lấn vào bắp tay/khuỷu tay nơi cổ tay áo lửng/ngắn tay dáng rộng phủ qua."""
    keypoints = np.zeros((17, 2), dtype=float)
    # Vai trái/phải: 5, 6
    keypoints[5] = [50, 100]
    keypoints[6] = [150, 100]
    # Khuỷu tay: 7, 8
    keypoints[7] = [50, 200]
    keypoints[8] = [150, 200]
    # Cổ tay: 9, 10
    keypoints[9] = [50, 300]
    keypoints[10] = [150, 300]
    pose_context = {
        "box": [40, 80, 160, 320],
        "keypoints": keypoints,
        "confidences": np.full(17, 0.9, dtype=float),
    }
    protected, forearm = q.upper_body_protected_mask((200, 400), pose_context, "short_sleeved_shirt", return_forearm=True)
    assert protected is not None
    # 50% bắp tay (y=150) phải không bị đánh dấu protected để giữ trọn cổ tay áo rộng/rơi vai
    assert protected[150, 50] == 0
    # Cẳng tay phía dưới (y=260) được bảo vệ để giữ da thật của mẫu/khách
    assert protected[260, 50] >= 200
    # Khuỷu tay (y=200) không bị forearm mask cắt cụt viền gấu tay áo
    assert forearm[200, 50] == 0
    assert forearm[260, 50] == 255


def test_long_sleeve_fit_ratio_and_coverage():
    """Áo dài tay đo tỷ lệ từ vai tới cổ tay (gần 2.0x arm span) không bị cap cụt ở 1.5."""
    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[5] = [50, 100]  # vai trái
    keypoints[7] = [50, 200]  # khuỷu tay trái (span = 100)
    keypoints[9] = [50, 300]  # cổ tay trái
    pose_context = {
        "keypoints": keypoints,
        "confidences": np.full(17, 0.9, dtype=float),
    }
    # Garment mask phủ từ vai y=100 xuống tận gần cổ tay y=290
    mask = np.zeros((400, 200), dtype=np.uint8)
    mask[100:295, 30:70] = 255
    res = q.analyze_model_garment_fit(pose_context, mask)
    assert res is not None
    # sleeve_length = 294 - 100 = 194. arm_span = 100. ratio = 1.94 > 1.5!
    assert res["sleeve_ratio"] > 1.8


def test_gate_candidate_accepts_all_long_sleeve_variants():
    """Tất cả các biến thể áo dài tay (hoodie, sweater, coat, tay_dai) không bị gate limb chặn."""
    for subtype in ["hoodie", "sweater", "aolen", "aoni", "long_sleeved_shirt", "tay_dai"]:
        assert gate_candidate("upper_body", subtype, outside=0.85, body=0.8, limb=0.4, match=3)


