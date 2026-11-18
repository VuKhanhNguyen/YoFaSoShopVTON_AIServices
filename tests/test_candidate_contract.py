"""Quy tắc ranking là hàm thuần, không cần MySQL/GPU."""
import pytest
import numpy as np
from PIL import Image
from api.services.pipeline import (
    candidate_score,
    classify_upper_style,
    expand_long_sleeve_arm_mask,
    format_catalog_context,
    gate_candidate,
    keep_verified_short_sleeve,
    outerwear_inner_v_mask,
    prepare_garment_canvas,
    restored_short_sleeve_score,
    short_sleeve_composite_masks,
    source_shirt_remains,
    trim_short_sleeve_bleed,
    upper_body_limit_mask,
    vlm_category_only_false_positive,
    widen_upper_torso_mask,
)
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


def test_upper_body_limit_mask_bridges_waist_arm_gap_without_ballooning():
    silhouette = np.zeros((100, 120), dtype=bool)
    silhouette[10:90, 15:30] = True   # right arm
    silhouette[10:90, 48:72] = True   # torso (18px gap each side)
    silhouette[10:90, 90:105] = True  # left arm
    limit = upper_body_limit_mask(silhouette)
    assert limit[50, 39]  # gap between right arm and torso is bridged
    assert limit[50, 81]  # gap between torso and left arm is bridged
    assert not limit[50, 2]  # far outer background stays excluded


def test_prepare_garment_canvas_keeps_pure_white_and_pads_tight_cutout():
    tight_polo = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for y in range(4, 96):
        for x in range(4, 96):
            tight_polo.putpixel((x, y), (242, 242, 242, 255))
    # Anti-aliased slate border pixel should be stripped on hard cutouts
    tight_polo.putpixel((4, 50), (70, 95, 120, 180))
    rgb_w, bg_w, mask_w = prepare_garment_canvas(tight_polo)
    assert bg_w == (255, 255, 255)
    assert rgb_w.getpixel((0, 0)) == (255, 255, 255)
    assert mask_w is not None and mask_w.shape == (rgb_w.height, rgb_w.width)
    # Padded out so the 92x92 tight garment now spans ~72% rather than >90% edge-to-edge
    ys, xs = np.where(mask_w)
    assert (ys.max() - ys.min()) / rgb_w.height <= 0.76


def test_trim_short_sleeve_bleed_preserves_elbow_cuff_and_armpit_drape():
    labels = np.zeros((280, 200), np.uint8)
    labels[60:260, 40:55] = 14   # left arm (image x=40..55)
    labels[60:200, 75:125] = 4   # source shirt torso
    clean_edit = np.zeros((280, 200), np.uint8)
    clean_edit[60:200, 75:125] = 255  # torso down to waist
    clean_edit[60:170, 34:76] = 255   # drop-shoulder cuff near elbow + armpit drape (x=55..75)
    clean_edit[170:215, 40:55] = 255  # abnormal bleed down distal forearm
    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[5] = [48, 60]   # shoulder
    keypoints[7] = [48, 160]  # elbow
    keypoints[9] = [48, 260]  # wrist -> forearm_cutoff = 160 + 25 = 185
    confs = np.full(17, 0.9, dtype=float)

    trimmed = trim_short_sleeve_bleed(clean_edit, labels, keypoints, confs, source_top=(labels == 4))
    assert trimmed[162, 48] == 255  # cuff near elbow stays intact
    assert trimmed[162, 65] == 255  # armpit drape between arm and torso NOT sliced horizontally!
    assert trimmed[200, 48] == 0    # distal forearm bleed still cleared
    assert trimmed[180, 90] == 255  # torso hem untouched


def test_format_catalog_context_humanizes_deepfashion_subtypes():
    ctx = format_catalog_context({
        "productName": "Áo polo nam trơn bo kẻ",
        "colorName": "Trắng",
        "vtonCategory": "upper_body",
        "garmentSubtype": "short_sleeved_shirt",
    })
    assert "short_sleeved_shirt" not in ctx
    assert "polo shirt" in ctx
    vest_ctx = format_catalog_context({"garmentSubtype": "vest"})
    assert "sleeveless" in vest_ctx


def test_vlm_category_only_false_positive_handles_short_sleeved_vs_real_defects():
    taxonomy_hallucination = {"passed": False, "score": 0.0, "reason": "The image does not show a short-sleeved top."}
    assert vlm_category_only_false_positive(taxonomy_hallucination, match=3, limb=1.0) is True
    assert vlm_category_only_false_positive(taxonomy_hallucination, match=1, limb=1.0) is False

    real_defect = {"passed": False, "score": 0.2, "reason": "The garment has a large gray stain across the chest."}
    assert vlm_category_only_false_positive(real_defect, match=3, limb=1.0) is False


def test_garment_color_similarity_prefers_clean_cream_over_gray_cast():
    mask = np.full((60, 60), 255, dtype=np.uint8)
    cream_ref = Image.new("RGB", (60, 60), (234, 220, 204))
    clean_cand = Image.new("RGB", (60, 60), (218, 209, 194))
    gray_cand = Image.new("RGB", (60, 60), (142, 142, 142))
    assert q.garment_color_similarity(cream_ref, mask, clean_cand, mask) > q.garment_color_similarity(cream_ref, mask, gray_cand, mask) + 0.1


def test_harmonize_light_garment_tone_lifts_white_and_preserves_dark_collar():
    mask = np.full((60, 60), 255, dtype=np.uint8)
    ref = Image.new("RGB", (60, 60), (242, 242, 242))
    cand = Image.new("RGB", (60, 60), (182, 174, 163))
    # Dark navy collar patch should stay dark while pale body shifts closer to crisp white
    for y in range(10, 22):
        for x in range(20, 40):
            cand.putpixel((x, y), (25, 36, 56))
    out = q.harmonize_light_garment_tone(cand, mask, ref, mask)
    body_px = out.getpixel((30, 40))
    collar_px = out.getpixel((30, 15))
    assert min(body_px) >= 210
    assert max(collar_px) <= 65

    # Dark garment must remain completely untouched
    dark_ref = Image.new("RGB", (60, 60), (40, 42, 46))
    untouched = q.harmonize_light_garment_tone(cand, mask, dark_ref, mask)
    assert untouched.getpixel((30, 40)) == (182, 174, 163)


def test_harmonize_sleeveless_arm_skin_matches_real_arm_and_preserves_tank():
    labels = np.zeros((120, 120), dtype=np.uint8)
    labels[20:110, 15:40] = 14   # real bare arm region
    labels[20:95, 42:82] = 4     # old shirt + short sleeve covered y=20..55 on arm overlap
    source_top = np.zeros((120, 120), dtype=bool)
    source_top[20:95, 42:82] = True
    source_top[20:55, 15:42] = True  # old short sleeve covered upper bicep down to y=55
    cand_mask = np.zeros((120, 120), dtype=np.uint8)
    cand_mask[20:95, 46:78] = 255    # sleeveless tank top torso only

    # Person has fair warm skin on bare lower arm; OOTD generated darker muted skin on upper bicep
    source_img = Image.new("RGB", (120, 120), (225, 192, 170))
    generated_img = Image.new("RGB", (120, 120), (225, 192, 170))
    arr_g = np.asarray(generated_img).copy()
    arr_g[20:60, 15:45] = (182, 150, 134)  # off-tone generated upper bicep
    arr_g[20:95, 46:78] = (165, 130, 140)  # mauve sleeveless tank top fabric
    generated_img = Image.fromarray(arr_g)

    out, exposed_arm = q.harmonize_sleeveless_arm_skin(generated_img, source_img, cand_mask, labels, source_top)
    assert exposed_arm is not None
    # Exposed arm includes overlap beyond y=55 cuff seam so cuff crease blends smoothly
    assert bool(exposed_arm[57, 28])
    # Tank top fabric untouched, while generated bicep skin shifts closer to real lower-arm skin
    assert out.getpixel((60, 60)) == (165, 130, 140)
    assert abs(out.getpixel((28, 35))[0] - 225) < abs(182 - 225)


def test_classify_upper_style_distinguishes_sleeveless_long_and_layered_blazer():
    # Sleeveless tank top -> sleeveless, NOT outerwear
    assert classify_upper_style("vest", {"productName": "AIRISM COTTON ÁO THUN KHÔNG TAY"}) == (True, False, False, False)
    # Crewneck gile / cardigan -> no V-lapel cutout
    assert classify_upper_style("long_sleeved_shirt", {"productName": "ÁO CARDIGAN SLIMME NỮ CỔ TRÒN ĐÓNG CÚC"}) == (False, True, True, False)
    # Tailored blazer (even if catalog subtype stored as long_sleeved_shirt) -> long sleeve + layered outerwear V-opening
    assert classify_upper_style("long_sleeved_shirt", {"productName": "ÁO BLAZER NAM DỆT JACQUARD FORM FITTED"}) == (False, True, True, True)
    # Sleeveless lapel gile -> both sleeveless arm handling AND V-lapel preservation
    assert classify_upper_style("vest", {"productName": "ÁO GILE NỮ VẢI TUYTXI CỔ HAI VE LIỀN"}) == (True, False, True, True)


def test_expand_long_sleeve_arm_mask_covers_outer_forearm_and_protects_hand():
    labels = np.zeros((320, 200), dtype=np.uint8)
    labels[60:240, 32:56] = 14   # muscular left arm (width 24px)
    mask_np = np.zeros((320, 200), dtype=np.uint8)
    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[5], keypoints[6] = [44, 60], [150, 60]
    keypoints[7] = [44, 150]
    keypoints[9] = [44, 240]
    keypoints[11], keypoints[12] = [65, 220], [135, 220]
    confs = np.full(17, 0.9, dtype=float)

    updated, dilated = expand_long_sleeve_arm_mask(mask_np, labels, keypoints, confs, fit_setting="SUONG_VUA")
    # Outer edge + 8px past muscular arm is covered for long sleeve
    assert updated[180, 26] == 255
    assert dilated[180, 56] == 255
    # Fingers well below wrist stay protected
    assert updated[290, 44] == 0


def test_widen_upper_torso_mask_and_outerwear_inner_v():
    labels = np.zeros((320, 240), dtype=np.uint8)
    labels[60:230, 85:155] = 4   # source shirt torso (narrow waist)
    labels[60:250, 35:55] = 14   # left arm with 30px background gap to torso (x=55..85)
    labels[60:250, 185:205] = 15 # right arm
    labels[10:55, 105:135] = 11  # head/neck
    mask_np = np.where(labels == 4, 255, 0).astype(np.uint8)
    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[5], keypoints[6] = [65, 65], [175, 65]
    keypoints[11], keypoints[12] = [90, 210], [150, 210]
    confs = np.full(17, 0.9, dtype=float)

    widened = widen_upper_torso_mask(mask_np, labels, keypoints, confs, "SUONG_VUA", target_hem_y=230, is_long_sleeve=True)
    # Waist background gap (x=72, y=170) is bridged so sweatshirt/blazer won't pinch inward
    assert widened[170, 72] == 255

    # Chest V-opening exposes user's inner shirt at sternum while keeping blazer body/lapels masked
    inner_v = outerwear_inner_v_mask(labels, keypoints, confs, source_shirt=(labels == 4))
    assert inner_v is not None
    assert bool(inner_v[85, 120])       # center chest collar/sternum preserved
    assert not bool(inner_v[170, 120])  # below top button covered by buttoned blazer
    assert not bool(inner_v[85, 70])    # shoulder seam covered by blazer



