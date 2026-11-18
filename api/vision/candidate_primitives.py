"""Snapshot hình học/màu/gate từ notebook VTON cell 12; không gọi Space."""
import cv2
import numpy as np
from PIL import Image, ImageOps


def prepare_oot_person(source_path, output_path, target_size=(768, 1024)):
    """Bước prepare_oot_person: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    image = Image.open(source_path).convert('RGB')
    (original_width, original_height) = image.size
    (target_width, target_height) = target_size
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    corners = [image.getpixel((0, 0)), image.getpixel((original_width - 1, 0)), image.getpixel((0, original_height - 1)), image.getpixel((original_width - 1, original_height - 1))]
    background_color = tuple((sum((pixel[channel] for pixel in corners)) // len(corners) for channel in range(3)))
    canvas = Image.new('RGB', target_size, background_color)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    canvas.save(output_path, 'PNG')
    metadata = {'original_size': (original_width, original_height), 'content_box': (offset_x, offset_y, offset_x + resized_width, offset_y + resized_height), 'target_size': target_size}
    return (output_path, metadata)

def restore_oot_result(output_image, metadata):
    """Bước restore_oot_result: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    output_image = output_image.convert('RGB')
    if output_image.size != metadata['target_size']:
        output_image = output_image.resize(metadata['target_size'], Image.Resampling.LANCZOS)
    output_image = output_image.crop(metadata['content_box'])
    return output_image.resize(metadata['original_size'], Image.Resampling.LANCZOS)

def composite_room_background(person, parsed, metadata, room):
    foreground = Image.fromarray((np.asarray(parsed) > 0).astype(np.uint8) * 255)
    foreground = foreground.resize(metadata["target_size"], Image.Resampling.BILINEAR)
    foreground = foreground.crop(metadata["content_box"]).resize(person.size, Image.Resampling.BILINEAR)
    if foreground.getextrema()[1] == 0:
        raise ValueError("Không nhận diện được người để ghép nền phòng.")
    background = ImageOps.fit(room.convert("RGB"), person.size, Image.Resampling.LANCZOS)
    return Image.composite(person, background, foreground)


def garment_edit_mask(source_mask, candidate_mask):
    """Hợp mask đồ cũ/mới và làm mềm biên để giới hạn vùng được phép thay đổi."""
    available_masks = [mask for mask in (source_mask, candidate_mask) if mask is not None]
    if not available_masks:
        return None
    edit_mask = np.maximum.reduce(available_masks).astype(np.uint8)
    edit_mask = cv2.dilate(edit_mask, np.ones((25, 25), np.uint8), iterations=1)
    return cv2.GaussianBlur(edit_mask, (0, 0), sigmaX=6, sigmaY=6)

def outside_garment_change_score(candidate_image, source_image, edit_mask):
    """Điểm càng thấp thì mặt, da, chân và nền ngoài vùng thử đồ càng được giữ nguyên."""
    if edit_mask is None:
        return float('inf')
    candidate_np = np.asarray(candidate_image.convert('RGB'), dtype=np.float32)
    source_np = np.asarray(source_image.convert('RGB'), dtype=np.float32)
    outside = edit_mask < 8
    if not np.any(outside):
        return float('inf')
    pixel_difference = np.mean(np.abs(candidate_np - source_np), axis=2)
    return float(np.mean(pixel_difference[outside]))

def change_to_preservation_score(change_score, scale=25.0):
    """Bước change_to_preservation_score: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    if not np.isfinite(change_score):
        return 0.0
    return float(np.exp(-max(0.0, change_score) / scale))

def visible_edit_mask(edit_mask, person_silhouette):
    """The inpaint area includes background for diffusion, but must not publish that background."""
    return np.where(person_silhouette, edit_mask, 0).astype(np.uint8)


def composite_preserving_source(candidate_image, source_image, edit_mask, protected_mask=None):
    """Bước composite_preserving_source: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    candidate_np = np.asarray(candidate_image.convert('RGB'), dtype=np.float32)
    source_np = np.asarray(source_image.convert('RGB'), dtype=np.float32)
    if edit_mask is None:
        alpha = np.ones((*source_np.shape[:2], 1), dtype=np.float32)
    else:
        alpha = edit_mask.astype(np.float32)[:, :, None] / 255.0
    if protected_mask is not None:
        protection = protected_mask.astype(np.float32)[:, :, None] / 255.0
        alpha *= 1.0 - protection
    protected = candidate_np * alpha + source_np * (1.0 - alpha)
    return Image.fromarray(np.clip(protected, 0, 255).astype(np.uint8))

def upper_body_protected_mask(image_size, pose_context, target_label, *, return_forearm=False):
    """Bảo vệ phần cánh tay mà vest/áo ngắn tay không được phép tự vẽ thêm vải."""
    if pose_context is None:
        return None
    label = (target_label or '').lower()
    if 'long_sleeved' in label:
        return None
    (width, height) = image_size
    keypoints = pose_context['keypoints']
    confidences = pose_context['confidences']
    if len(keypoints) < 11:
        return None
    shoulder_span = abs(float(keypoints[6][0] - keypoints[5][0]))
    thickness = max(12, min(42, int(round(max(60.0, shoulder_span) * 0.18))))
    start_fraction = 0.15 if any((name in label for name in ('vest', 'sling'))) else 0.70
    protected = np.zeros((height, width), dtype=np.uint8)
    forearm = np.zeros_like(protected) if return_forearm else None
    for (shoulder_index, elbow_index, wrist_index) in ((5, 7, 9), (6, 8, 10)):
        if min(confidences[[shoulder_index, elbow_index, wrist_index]]) < 0.25:
            continue
        shoulder = keypoints[shoulder_index]
        elbow = keypoints[elbow_index]
        wrist = keypoints[wrist_index]
        start = shoulder + start_fraction * (elbow - shoulder)
        start_xy = tuple(np.rint(start).astype(int))
        elbow_xy = tuple(np.rint(elbow).astype(int))
        wrist_xy = tuple(np.rint(wrist).astype(int))
        cv2.line(protected, start_xy, elbow_xy, 255, thickness, cv2.LINE_AA)
        cv2.line(protected, elbow_xy, wrist_xy, 255, thickness, cv2.LINE_AA)
        cv2.circle(protected, elbow_xy, thickness // 2, 255, -1, cv2.LINE_AA)
        cv2.circle(protected, wrist_xy, thickness // 2, 255, -1, cv2.LINE_AA)
        if forearm is not None:
            # Chừa vùng khuỷu tay cho cổ tay áo lửng/ngắn tay dáng rộng, bắt đầu từ 25% cẳng tay
            width_px = 2 * thickness
            forearm_start = elbow + 0.25 * (wrist - elbow)
            forearm_start_xy = tuple(np.rint(forearm_start).astype(int))
            cv2.line(forearm, forearm_start_xy, wrist_xy, 255, width_px)
            cv2.circle(forearm, wrist_xy, width_px // 2, 255, -1)
    if not np.any(protected):
        return None
    protection = cv2.GaussianBlur(protected, (0, 0), sigmaX=4, sigmaY=4)
    return (protection, forearm) if return_forearm else protection

def protected_region_artifact_score(candidate_image, source_image, protected_mask):
    """Trả (tỷ lệ artefact nặng, điểm bảo toàn) trên vùng da tay cần giữ nguyên."""
    if protected_mask is None:
        return (0.0, 1.0)
    candidate_np = np.asarray(candidate_image.convert('RGB'), dtype=np.float32)
    source_np = np.asarray(source_image.convert('RGB'), dtype=np.float32)
    weights = protected_mask.astype(np.float32) / 255.0
    if float(np.sum(weights)) < 1.0:
        return (0.0, 1.0)
    channel_difference = np.mean(np.abs(candidate_np - source_np), axis=2)
    source_luma = 0.299 * source_np[:, :, 0] + 0.587 * source_np[:, :, 1] + 0.114 * source_np[:, :, 2]
    candidate_luma = 0.299 * candidate_np[:, :, 0] + 0.587 * candidate_np[:, :, 1] + 0.114 * candidate_np[:, :, 2]
    severe_artifact = (source_luma - candidate_luma > 50.0) | (channel_difference > 70.0)
    artifact_ratio = float(np.sum(weights * severe_artifact) / np.sum(weights))
    preservation = float(np.exp(-artifact_ratio / 0.08))
    return (artifact_ratio, preservation)

def garment_fit_score(candidate_mask, pose_context, expected_category, target_label):
    """Heuristic 0..1 phát hiện garment lệch tâm, tràn thân hoặc phình bất thường."""
    if candidate_mask is None or pose_context is None:
        return 0.5
    binary = candidate_mask > 127
    (ys, xs) = np.where(binary)
    if len(xs) < 50:
        return 0.0
    (x1, x2) = (float(xs.min()), float(xs.max() + 1))
    (y1, y2) = (float(ys.min()), float(ys.max() + 1))
    (person_x1, person_y1, person_x2, person_y2) = pose_context['box']
    person_width = max(1.0, float(person_x2 - person_x1))
    center_offset = abs((x1 + x2) / 2.0 - (person_x1 + person_x2) / 2.0) / person_width
    outside_ratio = float(np.mean((xs < person_x1 - 5) | (xs > person_x2 + 5) | (ys < person_y1 - 5) | (ys > person_y2 + 5)))
    penalty = 0.35 * min(1.0, center_offset / 0.22) + 0.25 * min(1.0, outside_ratio / 0.05)
    if expected_category == 'upper_body':
        keypoints = pose_context['keypoints']
        confidences = pose_context['confidences']
        if min(confidences[[5, 6, 11, 12]]) >= 0.25:
            shoulder_span = max(1.0, abs(float(keypoints[6][0] - keypoints[5][0])))
            shoulder_y = float(np.mean(keypoints[[5, 6], 1]))
            hip_y = float(np.mean(keypoints[[11, 12], 1]))
            torso_height = max(1.0, hip_y - shoulder_y)
            width_ratio = (x2 - x1) / shoulder_span
            height_ratio = (y2 - y1) / torso_height
            long_sleeved = 'long_sleeved' in (target_label or '').lower()
            max_width_ratio = 3.0 if long_sleeved else 1.9
            max_height_ratio = 2.2 if long_sleeved else 1.45
            penalty += 0.2 * min(1.0, max(0.0, width_ratio - max_width_ratio) / 0.8)
            penalty += 0.2 * min(1.0, max(0.0, height_ratio - max_height_ratio) / 0.8)
    elif expected_category == 'lower_body':
        keypoints = pose_context.get('keypoints')
        confidences = pose_context.get('confidences')
        if keypoints is not None and len(keypoints) >= 17 and min(confidences[[11, 12]]) >= 0.2:
            hip_y = float(np.mean(keypoints[[11, 12], 1]))
            knee_y = float(np.mean(keypoints[[13, 14], 1]))
            ankle_y = float(np.mean(keypoints[[15, 16], 1]))
            if ankle_y > knee_y:
                leg_span = max(10.0, ankle_y - hip_y)
                cand_hem_y = float(ys.max())
                norm_hem = (cand_hem_y - hip_y) / leg_span
                is_shorts = (target_label == 'shorts') or ('short' in (target_label or '').lower())
                if is_shorts:
                    if norm_hem > 0.82:
                        penalty += 0.40  # Quần short nhưng bị dài quá gối sát mắt cá chân
                else:
                    if norm_hem < 0.68:
                        penalty += 0.50  # Quần dài nhưng bị cụt thành quần short!
    return float(np.clip(1.0 - penalty, 0.0, 1.0))

def calculate_limb_integrity_score(candidate_mask, pose_context, garment_label):
    """
    Chấm điểm độ toàn vẹn chi thể (0.0 - 1.0):
    Phát hiện và phạt nặng candidate nếu áo try-on phình ra che mất cổ tay hoặc cẳng tay
    đang đút vào túi quần của người mẫu.
    """
    if candidate_mask is None or pose_context is None:
        return 1.0
    keypoints = pose_context.get('keypoints')
    confidences = pose_context.get('confidences')
    if keypoints is None or confidences is None or len(keypoints) < 11:
        return 1.0
    mask = (candidate_mask > 127).astype(np.uint8)
    (h, w) = mask.shape[:2]
    shoulder_span = abs(float(keypoints[6][0] - keypoints[5][0]))
    wrist_radius = max(14, int(round(max(60.0, shoulder_span) * 0.08)))
    forearm_thickness = max(12, int(round(max(60.0, shoulder_span) * 0.09)))
    label = (garment_label or '').lower()
    allows_forearm_coverage = any((word in label for word in ['long_sleeved', 'coat', 'jacket', 'sweater']))
    if allows_forearm_coverage:
        return 1.0
    is_short_sleeve = True
    total_penalty = 0.0
    for (elbow_idx, wrist_idx) in ((7, 9), (8, 10)):
        if confidences[wrist_idx] >= 0.25:
            (wx, wy) = (int(round(keypoints[wrist_idx][0])), int(round(keypoints[wrist_idx][1])))
            wrist_roi = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(wrist_roi, (wx, wy), wrist_radius, 255, -1)
            overlap_pixels = np.count_nonzero((wrist_roi > 0) & (mask > 0))
            roi_pixels = np.count_nonzero(wrist_roi > 0)
            if roi_pixels > 0:
                overlap_ratio = overlap_pixels / roi_pixels
                if overlap_ratio > 0.05:
                    total_penalty += min(1.0, overlap_ratio * 2.0) * 0.5
        if is_short_sleeve and confidences[elbow_idx] >= 0.25 and (confidences[wrist_idx] >= 0.25):
            (ex, ey) = keypoints[elbow_idx]
            (wx, wy) = keypoints[wrist_idx]
            p_start = (int(round(ex + 0.35 * (wx - ex))), int(round(ey + 0.35 * (wy - ey))))
            p_end = (int(round(wx)), int(round(wy)))
            forearm_roi = np.zeros((h, w), dtype=np.uint8)
            cv2.line(forearm_roi, p_start, p_end, 255, forearm_thickness, cv2.LINE_AA)
            forearm_overlap = np.count_nonzero((forearm_roi > 0) & (mask > 0))
            forearm_total = np.count_nonzero(forearm_roi > 0)
            if forearm_total > 0:
                forearm_ratio = forearm_overlap / forearm_total
                if forearm_ratio > 0.05:
                    total_penalty += min(1.0, forearm_ratio * 2.0) * 0.5
    return float(np.clip(1.0 - total_penalty, 0.0, 1.0))

def evaluate_dress_proportions_and_color(candidate_image, candidate_mask, pose_context, reference_image, reference_mask):
    """
    Tiêu chí chuyên biệt cho Dress:
    1. Phạt nếu váy phình to bất thường so với khoảng cách hai hông (hip_span).
    2. Phạt nếu váy bị loang mảng màu lạ không có trong đầm mẫu (chống hallucination).
    """
    if candidate_mask is None or pose_context is None:
        return (0.5, 0.5)
    keypoints = pose_context.get('keypoints')
    confidences = pose_context.get('confidences')
    if keypoints is None or len(keypoints) < 13:
        return (0.5, 0.5)
    mask = (candidate_mask > 127).astype(np.uint8)
    (ys, xs) = np.where(mask > 0)
    if len(xs) < 100:
        return (0.0, 0.0)
    hip_width = abs(float(keypoints[12][0] - keypoints[11][0]))
    if hip_width < 20.0:
        hip_width = max(30.0, abs(float(keypoints[6][0] - keypoints[5][0])) * 0.75)
    dress_width = float(xs.max() - xs.min())
    width_ratio = dress_width / hip_width
    fit_penalty = 0.0
    if width_ratio > 1.45:
        fit_penalty = min(1.0, (width_ratio - 1.45) / 0.6)
    dress_fit_score = float(np.clip(1.0 - fit_penalty, 0.0, 1.0))
    color_score = garment_color_similarity(reference_image, reference_mask, candidate_image, candidate_mask)
    return (dress_fit_score, color_score)

def garment_color_similarity(reference_image, reference_mask, candidate_image, candidate_mask):
    """So màu trong vùng garment bằng Lab histogram + moment khoảng cách mượt; tránh cliff ở biên bin trắng/kem."""
    if reference_mask is None or candidate_mask is None:
        return 0.0

    def extract_lab_stats(image, mask):
        image_np = np.asarray(image.convert('RGB'), dtype=np.uint8)
        mask_u8 = (mask > 127).astype(np.uint8) * 255
        clean_mask = cv2.erode(mask_u8, np.ones((5, 5), np.uint8), iterations=1)
        if np.count_nonzero(clean_mask) < 50:
            clean_mask = mask_u8
        if np.count_nonzero(clean_mask) < 50:
            return None, None
        lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        histogram = cv2.calcHist([lab_image], [0, 1, 2], clean_mask, [8, 12, 12], [0, 256, 0, 256, 0, 256])
        histogram = cv2.normalize(histogram, None, alpha=1.0, norm_type=cv2.NORM_L1)
        pixels = lab_image[clean_mask > 0].astype(np.float32)
        return histogram, (pixels.mean(axis=0), float(pixels[:, 0].std()))

    ref_hist, ref_stats = extract_lab_stats(reference_image, reference_mask)
    cand_hist, cand_stats = extract_lab_stats(candidate_image, candidate_mask)
    if ref_hist is None or cand_hist is None:
        return 0.0
    hist_sim = float(np.clip(1.0 - cv2.compareHist(ref_hist, cand_hist, cv2.HISTCMP_BHATTACHARYYA), 0.0, 1.0))
    ref_mean, ref_std_l = ref_stats
    cand_mean, cand_std_l = cand_stats
    dist = float(np.sqrt(
        ((cand_mean[0] - ref_mean[0]) / 35.0) ** 2
        + ((cand_mean[1] - ref_mean[1]) / 20.0) ** 2
        + ((cand_mean[2] - ref_mean[2]) / 20.0) ** 2
        + 0.5 * ((cand_std_l - ref_std_l) / 30.0) ** 2
    ))
    moment_sim = float(np.exp(-dist))
    return float(np.clip(0.55 * hist_sim + 0.45 * moment_sim, 0.0, 1.0))


def harmonize_light_garment_tone(composite_image, candidate_mask, reference_image, reference_mask):
    """Chỉ bù phơi sáng/cân bằng trắng dịu trên áo nền trắng-kem để triệt ám xám (127,127,127), giữ nguyên cổ/bo màu đậm."""
    if reference_mask is None or candidate_mask is None:
        return composite_image
    ref_u8 = (np.asarray(reference_mask) > 127).astype(np.uint8) * 255
    cand_u8 = (np.asarray(candidate_mask) > 127).astype(np.uint8) * 255
    if np.count_nonzero(ref_u8) < 100 or np.count_nonzero(cand_u8) < 100:
        return composite_image

    ref_lab = cv2.cvtColor(np.asarray(reference_image.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_eroded = cv2.erode(ref_u8, np.ones((5, 5), np.uint8), iterations=1)
    ref_px = ref_lab[(ref_eroded if np.count_nonzero(ref_eroded) >= 64 else ref_u8) > 0]
    ref_chroma = np.hypot(ref_px[:, 1] - 128.0, ref_px[:, 2] - 128.0)
    light_ref = ref_px[(ref_px[:, 0] >= 185.0) & (ref_chroma <= 24.0)]
    if len(light_ref) / max(1, len(ref_px)) < 0.65:
        return composite_image

    comp_rgb = np.asarray(composite_image.convert("RGB"), dtype=np.uint8)
    comp_lab = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    cand_eroded = cv2.erode(cand_u8, np.ones((7, 7), np.uint8), iterations=1)
    cand_px = comp_lab[(cand_eroded if np.count_nonzero(cand_eroded) >= 64 else cand_u8) > 0]
    cand_chroma = np.hypot(cand_px[:, 1] - 128.0, cand_px[:, 2] - 128.0)
    body_px = cand_px[(cand_px[:, 0] >= 115.0) & (cand_chroma <= 26.0)]
    if len(body_px) < 64:
        return composite_image

    dl = float(np.median(light_ref[:, 0]) - np.median(body_px[:, 0]))
    da = float(np.median(light_ref[:, 1]) - np.median(body_px[:, 1]))
    db = float(np.median(light_ref[:, 2]) - np.median(body_px[:, 2]))
    if abs(dl) < 8.0 and abs(da) < 4.0 and abs(db) < 4.0:
        return composite_image

    # Weight = 0 trên cổ/bo/logo đậm màu (L < 120 hoặc sắc độ cao) và mượt dần ở viền mask
    pix_l = comp_lab[:, :, 0]
    pix_c = np.hypot(comp_lab[:, :, 1] - 128.0, comp_lab[:, :, 2] - 128.0)
    light_gate = np.clip((pix_l - 120.0) / 45.0, 0.0, 1.0)
    neutral_gate = np.clip(1.0 - (pix_c - 12.0) / 18.0, 0.0, 1.0)
    mask_soft = cv2.GaussianBlur(cand_u8.astype(np.float32) / 255.0, (9, 9), 0)
    weight = mask_soft * light_gate * neutral_gate

    shifted_lab = comp_lab.copy()
    shifted_lab[:, :, 0] = np.clip(pix_l + weight * float(np.clip(dl * 0.70, -18.0, 42.0)), 0.0, 255.0)
    shifted_lab[:, :, 1] = np.clip(comp_lab[:, :, 1] + weight * float(np.clip(da * 0.70, -14.0, 14.0)), 0.0, 255.0)
    shifted_lab[:, :, 2] = np.clip(comp_lab[:, :, 2] + weight * float(np.clip(db * 0.70, -14.0, 14.0)), 0.0, 255.0)
    harmonized = cv2.cvtColor(np.rint(shifted_lab).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(harmonized)


def harmonize_sleeveless_arm_skin(generated_image, source_image, candidate_mask, person_labels, source_top):
    """Đồng bộ màu da vai/bắp tay vùng cắt tay áo cũ theo màu da thật trên ảnh gốc, xóa vết hằn cộc tay."""
    cand_bool = np.asarray(candidate_mask) > 127
    source_bool = np.asarray(source_top, dtype=bool)
    arms = np.isin(person_labels, (14, 15))
    if not np.any(arms) or not np.any(source_bool):
        return generated_image, None

    arm_transition = (cv2.dilate(source_bool.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1) > 0) & arms
    arm_zone = cv2.dilate(arms.astype(np.uint8), np.ones((45, 45), np.uint8), iterations=1) > 0
    exposed_arm = (source_bool | arm_transition) & ~cand_bool & arm_zone
    if np.count_nonzero(exposed_arm) < 64:
        return generated_image, None

    src_lab = cv2.cvtColor(np.asarray(source_image.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    gen_lab = cv2.cvtColor(np.asarray(generated_image.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    old_sleeve_dilated = cv2.dilate(source_bool.astype(np.uint8), np.ones((19, 19), np.uint8), iterations=1) > 0
    bare_arm = arms & ~old_sleeve_dilated & ~cand_bool
    ref_px = src_lab[bare_arm]
    ref_px = ref_px[ref_px[:, 0] >= 85.0]
    if len(ref_px) < 120:
        # Ảnh khách mặc áo dài tay che hết cánh tay: lấy vùng cổ/mặt dưới làm mẫu màu da thật
        head_ys, head_xs = np.where(person_labels == 11)
        if len(head_ys) > 0:
            y_neck = int(head_ys.min() + (head_ys.max() - head_ys.min()) * 0.65)
            neck_mask = (person_labels == 11) & (np.arange(person_labels.shape[0])[:, None] >= y_neck)
            ref_px = src_lab[neck_mask]
            ref_px = ref_px[ref_px[:, 0] >= 85.0]
    if len(ref_px) < 48:
        return generated_image, exposed_arm

    core_exposed = cv2.erode(exposed_arm.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    gen_px = gen_lab[core_exposed if np.count_nonzero(core_exposed) >= 48 else exposed_arm]
    gen_px = gen_px[gen_px[:, 0] >= 85.0]
    if len(gen_px) < 32:
        return generated_image, exposed_arm

    dl = float(np.clip((np.median(ref_px[:, 0]) - np.median(gen_px[:, 0])) * 0.92, -38.0, 38.0))
    da = float(np.clip((np.median(ref_px[:, 1]) - np.median(gen_px[:, 1])) * 0.95, -22.0, 22.0))
    db = float(np.clip((np.median(ref_px[:, 2]) - np.median(gen_px[:, 2])) * 0.95, -22.0, 22.0))

    weight = cv2.GaussianBlur(exposed_arm.astype(np.float32), (21, 21), 0)
    weight = np.where(cand_bool, 0.0, weight)
    shifted = gen_lab.copy()
    shifted[:, :, 0] = np.clip(gen_lab[:, :, 0] + weight * dl, 0.0, 255.0)
    shifted[:, :, 1] = np.clip(gen_lab[:, :, 1] + weight * da, 0.0, 255.0)
    shifted[:, :, 2] = np.clip(gen_lab[:, :, 2] + weight * db, 0.0, 255.0)
    return Image.fromarray(cv2.cvtColor(np.rint(shifted).astype(np.uint8), cv2.COLOR_LAB2RGB)), exposed_arm
DRESS_STYLE_FAMILY = {'vest_dress': 'sleeveless_dress', 'sling_dress': 'sleeveless_dress', 'short_sleeved_dress': 'short_sleeved_dress', 'long_sleeved_dress': 'long_sleeved_dress'}

def candidate_match_quality(candidate_label, target_label, expected_category):
    """Trả mức khớp subtype; confidence chỉ được dùng khi hai candidate cùng mức."""
    candidate_label = (candidate_label or '').lower()
    target_label = (target_label or '').lower()
    if not candidate_label:
        return (0, 'không nhận diện đúng category')
    if candidate_label == target_label:
        return (3, 'khớp subtype chính xác')
    if expected_category == 'dress':
        candidate_family = DRESS_STYLE_FAMILY.get(candidate_label)
        target_family = DRESS_STYLE_FAMILY.get(target_label)
        if candidate_family and candidate_family == target_family:
            return (2, f'cùng nhóm kiểu tay ({target_family})')
        return (1, 'đúng dress nhưng sai kiểu tay/subtype')
    if expected_category == 'lower_body':
        if candidate_label in {'shorts', 'trousers', 'skirt'}:
            return (2, f"cùng nhóm lower-body ({candidate_label} thay vì {target_label})")
        return (1, f"đúng lower-body nhưng detector nhận '{candidate_label}' thay vì '{target_label}'")
    return (1, 'đúng upper-body nhưng khác subtype')


def analyze_model_garment_fit(pose_context, garment_mask):
    """Đo tỷ lệ chiều dài tay áo trên người mẫu shop hoặc candidate (theo notebook Colab ver2)."""
    if pose_context is None or garment_mask is None:
        return None
    kpts = pose_context.get('keypoints')
    confs = pose_context.get('confidences')
    if kpts is None or confs is None or len(kpts) < 11:
        return None
    mask = garment_mask > 127 if getattr(garment_mask, 'dtype', None) != bool else garment_mask
    ys, xs = np.where(mask)
    if len(xs) < 50:
        return None
    sleeve_ratios = []
    for s_idx, e_idx in ((5, 7), (6, 8)):
        if confs[s_idx] >= 0.25 and confs[e_idx] >= 0.25:
            sy, sx = kpts[s_idx][1], kpts[s_idx][0]
            ey, ex = kpts[e_idx][1], kpts[e_idx][0]
            arm_span = max(10.0, ey - sy)
            arm_side_ys = ys[xs < sx + 15] if s_idx == 5 else ys[xs > sx - 15]
            if len(arm_side_ys):
                sleeve_hem_y = float(arm_side_ys.max())
                sleeve_length = max(0.0, sleeve_hem_y - sy)
                sleeve_ratios.append(min(2.2, sleeve_length / arm_span))
    avg_sleeve = float(np.mean(sleeve_ratios)) if sleeve_ratios else 0.5
    return {"sleeve_ratio": round(avg_sleeve, 3)}


def lower_body_length_fidelity(candidate_mask, pose_context, target_subtype, catalog_length=None):
    """Đảm bảo quần dài không bị nhận nhầm/chọn nhầm thành quần short và ngược lại."""
    if candidate_mask is None or pose_context is None:
        return 1.0
    keypoints = pose_context.get("keypoints")
    if keypoints is None or len(keypoints) < 17:
        return 1.0
    mask = candidate_mask > 127 if getattr(candidate_mask, 'dtype', None) != bool else candidate_mask
    ys, xs = np.where(mask)
    if len(ys) < 50:
        return 0.5
    cand_hem_y = float(ys.max())
    hip_y = float(np.mean(keypoints[[11, 12], 1]))
    knee_y = float(np.mean(keypoints[[13, 14], 1]))
    ankle_y = float(np.mean(keypoints[[15, 16], 1]))
    if ankle_y <= knee_y:
        return 1.0
    leg_span = max(10.0, ankle_y - hip_y)
    norm_hem = (cand_hem_y - hip_y) / leg_span
    is_shorts = (target_subtype == "shorts") or (catalog_length in ("SHORT", "NGAN_TREN_GOI")) or ("short" in (target_subtype or "").lower())
    if is_shorts:
        if norm_hem > 0.82:
            return 0.2
        return 1.0
    else:
        # Quần dài (trousers, jeans, kaki, jogger, DAI_CHAM_GOT): gấu quần phải tới gần mắt cá chân
        if norm_hem < 0.68:
            return 0.0
        return float(np.clip((norm_hem - 0.65) / 0.25, 0.2, 1.0))

