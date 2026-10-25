"""Snapshot logic sản phẩm từ notebook; không load model khi import.
Model/runtime được service worker gắn trong một process, không chạy trong API.
Nguồn: product_reference_analysis_colab.ipynb, cell 6 và 8.
"""
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
DEVICE = "cpu"
fashion_model = None
pose_model = None


MIN_IMAGE_SIDE = 160
POSE_CONFIDENCE = 0.3
LABEL_TO_DISPLAY = {'short_sleeved_shirt': 'short sleeve top', 'long_sleeved_shirt': 'long sleeve top', 'short_sleeved_outwear': 'short sleeve outerwear', 'long_sleeved_outwear': 'long sleeve outerwear', 'vest': 'vest', 'sling': 'sling', 'shorts': 'shorts', 'trousers': 'trousers', 'skirt': 'skirt', 'short_sleeved_dress': 'short sleeve dress', 'long_sleeved_dress': 'long sleeve dress', 'vest_dress': 'vest dress', 'sling_dress': 'sling dress'}
UPPER_LABELS = {'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling'}
LOWER_LABELS = {'shorts', 'trousers', 'skirt'}
DRESS_LABELS = {'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'}
CATEGORY_OVERRIDE = None
DRESS_PROMOTION = {'short_sleeved_shirt': 'short_sleeved_dress', 'short_sleeved_outwear': 'short_sleeved_dress', 'long_sleeved_shirt': 'long_sleeved_dress', 'long_sleeved_outwear': 'long_sleeved_dress', 'vest': 'vest_dress', 'sling': 'sling_dress'}
DRESS_DEMOTION = {'short_sleeved_dress': 'short_sleeved_shirt', 'long_sleeved_dress': 'long_sleeved_shirt', 'vest_dress': 'vest', 'sling_dress': 'sling'}

def label_family(label):
    """Bước label_family: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    if label in UPPER_LABELS:
        return 'upper_body'
    if label in LOWER_LABELS:
        return 'lower_body'
    if label in DRESS_LABELS:
        return 'dress'
    return 'unknown'

def normalized_filename_text(filename):
    """Bước normalized_filename_text: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    text = Path(filename).stem.casefold().replace('đ', 'd')
    text = unicodedata.normalize('NFKD', text)
    text = ''.join((char for char in text if not unicodedata.combining(char)))
    return re.sub('[^a-z0-9]+', ' ', text).strip()

def filename_category_hint(filename):
    """Chỉ trả hint cho các cụm đủ mạnh; không suy diễn mọi từ 'váy' là dress."""
    text = f' {normalized_filename_text(filename)} '
    if any((token in text for token in (' dam ', ' dress ', ' ao dai ', ' jumpsuit '))):
        return 'dress'
    if any((token in text for token in (' chan vay ', ' skirt '))):
        return 'lower_body'
    if any((token in text for token in (' ao thun ', ' ao polo ', ' ao so mi ', ' ao ni ', ' ao len ', ' cardigan ', ' shirt ', ' blouse ', ' polo ', ' top '))):
        return 'upper_body'
    return None

def promote_label_to_family(label, family):
    """Bước promote_label_to_family: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    if label_family(label) == family:
        return label
    if family == 'dress' and label in DRESS_PROMOTION:
        return DRESS_PROMOTION[label]
    if family == 'upper_body' and label in DRESS_DEMOTION:
        return DRESS_DEMOTION[label]
    return None

def safe_stem(filename):
    """Bước safe_stem: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    stem = re.sub('[^0-9A-Za-z_-]+', '_', Path(filename).stem).strip('_')
    return stem[:70] or 'image'

def fashion_candidates(image, confidence=0.25):
    """Bước fashion_candidates: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    result = fashion_model.predict(np.asarray(image), conf=confidence, imgsz=1024, retina_masks=True, device=DEVICE, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    image_area = max(1, image.width * image.height)
    candidates = []
    for (index, (box, score, class_id)) in enumerate(zip(boxes, scores, classes)):
        label = str(result.names[int(class_id)])
        if label not in LABEL_TO_DISPLAY:
            continue
        area = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / image_area
        candidates.append({'index': index, 'label': label, 'confidence': float(score), 'area_ratio': float(area), 'box': box.astype(float), 'result': result})
    return candidates

def choose_primary_garment(candidates):
    """Bước choose_primary_garment: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    if not candidates:
        return None
    return max(candidates, key=lambda item: item['confidence'] * math.sqrt(max(item['area_ratio'], 1e-06)))

def best_pose(image):
    """Bước best_pose: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    result = pose_model.predict(np.asarray(image), classes=[0], conf=0.25, device=DEVICE, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None
    scores = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = int(np.argmax(scores * np.sqrt(np.maximum(areas, 1))))
    keypoints = result.keypoints.xy[best].cpu().numpy()
    confidences = result.keypoints.conf
    if confidences is None:
        return None
    return {'confidence': float(scores[best]), 'box': boxes[best].astype(float), 'xy': keypoints.astype(float), 'conf': confidences[best].cpu().numpy().astype(float)}

def has_landmark_pair(pose, indices):
    """Bước has_landmark_pair: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    return pose is not None and any((pose['conf'][idx] >= POSE_CONFIDENCE for idx in indices))

def consensus_label(product_items):
    """Bước consensus_label: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    votes = Counter()
    for item in product_items:
        detection = item['detection']
        votes[detection['label']] += detection['confidence'] * max(detection['area_ratio'], 0.05)
    return max(votes, key=votes.get)

def _candidate_strength(candidate):
    """Bước _candidate_strength: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    return float(candidate['confidence']) * math.sqrt(max(float(candidate['area_ratio']), 0.01))

def model_reference_family_votes(model_items):
    """Lấy bằng chứng category từ ảnh người mẫu mà chưa ép theo ảnh sản phẩm."""
    votes = Counter()
    evidence = []
    for item in model_items:
        candidates = item['candidates']
        pose = item['pose']
        upper = choose_primary_garment([value for value in candidates if label_family(value['label']) == 'upper_body'])
        dress = choose_primary_garment([value for value in candidates if label_family(value['label']) == 'dress'])
        lower = choose_primary_garment([value for value in candidates if label_family(value['label']) == 'lower_body'])
        item_evidence = []
        separate_bottom = lower if lower and lower['label'] in {'trousers', 'shorts'} else None
        if upper is not None and separate_bottom is not None and (upper['confidence'] >= 0.45) and (separate_bottom['confidence'] >= 0.45):
            score = 1.5 * min(upper['confidence'], separate_bottom['confidence'])
            votes['upper_body'] += score
            item_evidence.append(f"top+{separate_bottom['label']}={score:.2f}")
        if upper is not None:
            score = _candidate_strength(upper)
            hip_y = visible_pose_y(pose, (11, 12))
            person_height = max(1.0, pose['box'][3] - pose['box'][1])
            if hip_y is not None and upper['box'][3] <= hip_y + 0.16 * person_height:
                score *= 1.35
                item_evidence.append(f'upper_above_hip={score:.2f}')
            else:
                item_evidence.append(f'upper={score:.2f}')
            votes['upper_body'] += score
        if dress is not None:
            score = 1.2 * _candidate_strength(dress)
            votes['dress'] += score
            item_evidence.append(f'dress={score:.2f}')
        if lower is not None:
            score = _candidate_strength(lower)
            votes['lower_body'] += score
            item_evidence.append(f'lower={score:.2f}')
        evidence.append({'file': item['original_name'], 'signals': item_evidence})
    return (votes, evidence)

def _best_label_for_family(product_items, model_items, family, fallback_label):
    """Bước _best_label_for_family: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    label_votes = Counter()
    for item in product_items:
        candidates = item.get('candidates', [item['detection']])
        for detection in candidates:
            if label_family(detection['label']) == family:
                label_votes[detection['label']] += _candidate_strength(detection)
    for item in model_items:
        for candidate in item['candidates']:
            if label_family(candidate['label']) == family:
                label_votes[candidate['label']] += 0.65 * _candidate_strength(candidate)
    if label_votes:
        return max(label_votes, key=label_votes.get)
    return promote_label_to_family(fallback_label, family) or fallback_label

def resolve_session_category(product_items, model_items):
    """Bước resolve_session_category: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    model_label = consensus_label(product_items)
    if CATEGORY_OVERRIDE is not None:
        if CATEGORY_OVERRIDE not in LABEL_TO_DISPLAY:
            raise ValueError(f"CATEGORY_OVERRIDE='{CATEGORY_OVERRIDE}' không hợp lệ. Chọn một trong: {sorted(LABEL_TO_DISPLAY)}")
        resolved_label = CATEGORY_OVERRIDE
        source = 'manual_override'
        family_scores = Counter({label_family(resolved_label): 1.0})
        model_evidence = []
    else:
        family_scores = Counter()
        for item in product_items:
            detection = item['detection']
            family_scores[label_family(detection['label'])] += _candidate_strength(detection)
            hint = item.get('category_hint')
            if hint:
                family_scores[hint] += 1.25
        (reference_votes, model_evidence) = model_reference_family_votes(model_items)
        for (family, score) in reference_votes.items():
            family_scores[family] += score
        resolved_family = max(('upper_body', 'lower_body', 'dress'), key=lambda family: family_scores.get(family, 0.0))
        resolved_label = _best_label_for_family(product_items, model_items, resolved_family, model_label)
        source = 'joint_product+filename+model_reference'
    resolved_family = label_family(resolved_label)
    for item in product_items:
        original_detection = item['detection']
        original_model_label = original_detection['label']
        family_candidates = [candidate for candidate in item.get('candidates', []) if label_family(candidate['label']) == resolved_family and candidate['confidence'] >= 0.18 and (candidate['area_ratio'] >= 0.03)]
        detection = choose_primary_garment(family_candidates)
        if detection is not None:
            detection = dict(detection)
            item['detection'] = detection
            reselection = 'family_candidate'
        else:
            detection = original_detection
            reselection = 'promoted_primary_fallback'
        detection['model_label'] = original_model_label
        promoted = promote_label_to_family(detection['label'], resolved_family)
        if promoted is not None:
            detection['label'] = promoted
        elif label_family(detection['label']) != resolved_family:
            detection['label'] = resolved_label
        detection['category_resolution_source'] = source
        detection['product_detection_reselection'] = reselection
    details = {'family_scores': {family: round(float(family_scores.get(family, 0.0)), 4) for family in ('upper_body', 'lower_body', 'dress')}, 'model_reference_evidence': model_evidence}
    return (resolved_label, model_label, source, details)

def make_promoted_detection(candidate, target_label, source):
    """Bước make_promoted_detection: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    promoted = dict(candidate)
    promoted['model_label'] = candidate['label']
    promoted['label'] = target_label
    promoted['category_resolution_source'] = source
    return promoted

def make_composite_dress_detection(members, target_label):
    """Bước make_composite_dress_detection: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    boxes = np.stack([member['box'] for member in members])
    composite = {'index': members[0]['index'], 'label': target_label, 'model_label': '+'.join((member['label'] for member in members)), 'confidence': float(np.mean([member['confidence'] for member in members])), 'area_ratio': float(min(1.0, sum((member['area_ratio'] for member in members)))), 'box': np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()], dtype=float), 'result': members[0]['result'], '_composite_members': members, 'category_resolution_source': 'model_top+skirt_composite'}
    return composite

def visible_pose_y(pose, indices):
    """Bước visible_pose_y: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    values = [pose['xy'][idx, 1] for idx in indices if pose['conf'][idx] >= POSE_CONFIDENCE]
    return float(np.median(values)) if values else None

def choose_model_garment(candidates, pose, session_label, session_family):
    """Bước choose_model_garment: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    family_options = [item for item in candidates if label_family(item['label']) == session_family]
    exact_options = [item for item in family_options if item['label'] == session_label]
    direct = choose_primary_garment(exact_options or family_options)
    if session_family != 'dress':
        return direct
    if direct is not None and direct['confidence'] >= 0.25:
        return direct
    upper = choose_primary_garment([item for item in candidates if label_family(item['label']) == 'upper_body'])
    skirt = choose_primary_garment([item for item in candidates if item['label'] == 'skirt'])
    if upper is not None and skirt is not None:
        (upper_box, skirt_box) = (upper['box'], skirt['box'])
        horizontal_overlap = max(0.0, min(upper_box[2], skirt_box[2]) - max(upper_box[0], skirt_box[0]))
        narrow_width = max(1.0, min(upper_box[2] - upper_box[0], skirt_box[2] - skirt_box[0]))
        person_height = max(1.0, pose['box'][3] - pose['box'][1])
        vertical_gap = max(0.0, skirt_box[1] - upper_box[3])
        if horizontal_overlap / narrow_width >= 0.25 and vertical_gap <= 0.18 * person_height:
            return make_composite_dress_detection([upper, skirt], session_label)
    if upper is not None:
        hip_y = visible_pose_y(pose, (11, 12))
        knee_y = visible_pose_y(pose, (13, 14))
        if hip_y is not None and knee_y is not None:
            minimum_dress_bottom = hip_y + 0.12 * max(knee_y - hip_y, 1.0)
            if upper['box'][3] >= minimum_dress_bottom:
                return make_promoted_detection(upper, session_label, 'model_long_upper_promoted_to_dress')
    if direct is not None:
        return make_promoted_detection(direct, direct['label'], 'deepfashion2_low_confidence_dress')
    return None

def finalize_model_items(model_items, session_label, session_family):
    """Bước finalize_model_items: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    rejected = []
    for item in model_items:
        garment = choose_model_garment(item['candidates'], item['pose'], session_label, session_family)
        if session_family in {'lower_body', 'dress'} and (not has_landmark_pair(item['pose'], (13, 14))):
            rejected.append(f"{item['original_name']}: category chung là {session_family}, cần ảnh thấy rõ đầu gối để đo chiều dài")
            continue
        if garment is None:
            detected_summary = ', '.join((f"{LABEL_TO_DISPLAY[value['label']]}={value['confidence']:.2f}" for value in sorted(item['candidates'], key=lambda candidate: candidate['confidence'], reverse=True)[:4])) or 'không có garment'
            rejected.append(f"{item['original_name']}: không dựng được garment thuộc vùng {session_family}; model thấy [{detected_summary}]")
            continue
        item['detection'] = garment
        ankle_note = '' if has_landmark_pair(item['pose'], (15, 16)) else '; chưa thấy cổ chân'
        print(f"✅ {item['original_name']} → garment cuối={LABEL_TO_DISPLAY[garment['label']]} (source={garment.get('category_resolution_source', 'deepfashion2')}){ankle_note}")
    if rejected:
        details = '\n- '.join(rejected)
        raise ValueError(f'Category đã được giải bằng cả ảnh mẫu và ảnh sản phẩm nhưng vẫn có ảnh mẫu không phù hợp:\n- {details}')
    return model_items
COLOR_PALETTE = {'đen': '#1C1C1C', 'trắng': '#F4F1E8', 'xám': '#808080', 'đỏ': '#C62828', 'đỏ đô': '#7F1D1D', 'cam': '#E97820', 'vàng': '#E5C229', 'be': '#D8C3A5', 'nâu': '#70452E', 'xanh lá': '#3D7A43', 'xanh olive': '#7A7D42', 'xanh mint': '#8EC9B3', 'xanh dương': '#295AA8', 'xanh navy': '#1D2E55', 'xanh cyan': '#2B9EB3', 'tím': '#75499B', 'hồng': '#D96C9D', 'kem': '#EEE0BD'}
FIT_LABELS = {'cropped': 'dáng crop / trên eo', 'at_waist': 'chạm eo', 'at_hip': 'chạm hoặc phủ hông', 'upper_thigh': 'dài tới đùi trên', 'above_knee': 'trên đầu gối', 'at_knee': 'chạm đầu gối', 'below_knee': 'qua đầu gối', 'mid_calf': 'dài giữa bắp chân', 'lower_calf': 'dài dưới bắp chân', 'ankle_length': 'dài tới cổ chân', 'unknown': 'chưa đủ dữ liệu'}

def _foreground_runs(row):
    """Trả về các đoạn foreground liên tục dạng (x_start, x_end)."""
    padded = np.pad(row.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))

def _component_count(mask):
    """Bước _component_count: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    (count, _, stats, _) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    minimum_area = max(16, int(mask.sum() * 0.0005))
    return int(sum((stats[index, cv2.CC_STAT_AREA] >= minimum_area for index in range(1, count))))

def _dress_lower_branch_cleanup(mask, label):
    """Loại nhánh nền lớn nằm hai bên thân váy nhưng giữ tà trung tâm và váy xòe liền khối."""
    if label not in DRESS_LABELS:
        return (mask.copy(), {'cleanup_applied': False, 'removed_fraction': 0.0})
    (ys, xs) = np.where(mask)
    if len(xs) < 500:
        return (mask.copy(), {'cleanup_applied': False, 'removed_fraction': 0.0})
    (x1, x2, y1, y2) = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    garment_height = max(1, y2 - y1 + 1)
    garment_width = max(1, x2 - x1 + 1)
    split_ratio = 0.6 if label == 'long_sleeved_dress' else 0.38
    split_y = min(y2, int(round(y1 + split_ratio * garment_height)))
    band_start = max(y1, split_y - max(6, int(0.07 * garment_height)))
    bbox_center = (x1 + x2) / 2.0
    (anchor_centers, anchor_widths) = ([], [])
    for row_y in range(band_start, split_y):
        runs = _foreground_runs(mask[row_y])
        if not runs:
            continue
        run = min(runs, key=lambda pair: abs((pair[0] + pair[1]) / 2.0 - bbox_center))
        anchor_centers.append((run[0] + run[1]) / 2.0)
        anchor_widths.append(run[1] - run[0] + 1)
    if not anchor_centers:
        return (mask.copy(), {'cleanup_applied': False, 'removed_fraction': 0.0})
    anchor_center = float(np.median(anchor_centers))
    anchor_width = max(8.0, float(np.percentile(anchor_widths, 40)))
    lower = np.zeros_like(mask, dtype=np.uint8)
    lower[split_y:y2 + 1] = mask[split_y:y2 + 1].astype(np.uint8)
    kernel_size = max(3, min(7, int(round(min(garment_width, garment_height) * 0.004))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    opened = cv2.morphologyEx(lower, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    (component_total, labels, stats, centroids) = cv2.connectedComponentsWithStats(opened, 8)
    if component_total <= 2:
        return (mask.copy(), {'cleanup_applied': False, 'removed_fraction': 0.0})
    central_left = anchor_center - max(anchor_width * 0.9, garment_width * 0.1)
    central_right = anchor_center + max(anchor_width * 0.9, garment_width * 0.1)
    minimum_component_area = max(64, int(mask.sum() * 0.002))
    keep_indices = []
    for component_index in range(1, component_total):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < minimum_component_area:
            continue
        comp_x = int(stats[component_index, cv2.CC_STAT_LEFT])
        comp_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        comp_right = comp_x + comp_width - 1
        comp_center = float(centroids[component_index][0])
        overlap = max(0.0, min(comp_right, central_right) - max(comp_x, central_left) + 1)
        overlap_ratio = overlap / max(comp_width, 1)
        center_distance = abs(comp_center - anchor_center)
        if comp_x <= anchor_center <= comp_right or overlap_ratio >= 0.3 or center_distance <= max(anchor_width * 0.8, garment_width * 0.13):
            keep_indices.append(component_index)
    if not keep_indices:
        return (mask.copy(), {'cleanup_applied': False, 'removed_fraction': 0.0})
    kept_lower = np.isin(labels, keep_indices)
    refined = mask.copy()
    refined[split_y:y2 + 1] = kept_lower[split_y:y2 + 1]
    removed = int(mask.sum() - refined.sum())
    removed_fraction = removed / max(int(mask.sum()), 1)
    if removed_fraction < 0.005:
        return (mask.copy(), {'cleanup_applied': False, 'removed_fraction': 0.0})
    return (refined, {'cleanup_applied': True, 'removed_fraction': round(float(removed_fraction), 4), 'split_y': split_y, 'anchor_x': round(anchor_center, 1), 'anchor_width': round(anchor_width, 1)})

def _remove_distant_dress_components(mask, label):
    """Bỏ component lớn xa thân chính; vẫn giữ tay rời phía trên và hai tà gần trục giữa."""
    if label not in DRESS_LABELS:
        return (mask.copy(), {'component_cleanup_applied': False})
    binary = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    (count, labels, stats, centroids) = cv2.connectedComponentsWithStats(opened, 8)
    if count <= 2:
        return (mask.copy(), {'component_cleanup_applied': False})
    component_indices = list(range(1, count))
    main_index = max(component_indices, key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]))
    main_x = int(stats[main_index, cv2.CC_STAT_LEFT])
    main_y = int(stats[main_index, cv2.CC_STAT_TOP])
    main_width = int(stats[main_index, cv2.CC_STAT_WIDTH])
    main_height = int(stats[main_index, cv2.CC_STAT_HEIGHT])
    main_right = main_x + main_width - 1
    main_center_x = float(centroids[main_index][0])
    (ys, xs) = np.where(mask)
    (overall_x1, overall_x2) = (int(xs.min()), int(xs.max()))
    (overall_y1, overall_y2) = (int(ys.min()), int(ys.max()))
    overall_width = max(1, overall_x2 - overall_x1 + 1)
    overall_height = max(1, overall_y2 - overall_y1 + 1)
    minimum_area = max(64, int(mask.sum() * 0.001))
    keep_indices = [main_index]
    removed_indices = []
    for component_index in component_indices:
        if component_index == main_index:
            continue
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        comp_x = int(stats[component_index, cv2.CC_STAT_LEFT])
        comp_y = int(stats[component_index, cv2.CC_STAT_TOP])
        comp_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        comp_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        comp_right = comp_x + comp_width - 1
        comp_bottom = comp_y + comp_height - 1
        comp_center_x = float(centroids[component_index][0])
        horizontal_gap = max(0, main_x - comp_right, comp_x - main_right)
        central_distance_limit = max(overall_width * 0.16, min(main_width * 0.3, overall_width * 0.22))
        near_center = abs(comp_center_x - main_center_x) <= central_distance_limit
        plausible_sleeve = comp_y <= overall_y1 + 0.28 * overall_height and comp_bottom <= overall_y1 + 0.65 * overall_height and (horizontal_gap <= 0.1 * overall_width)
        if area >= minimum_area and (near_center or plausible_sleeve):
            keep_indices.append(component_index)
        else:
            removed_indices.append(component_index)
    if not removed_indices:
        return (mask.copy(), {'component_cleanup_applied': False})
    kept_seed = np.isin(labels, keep_indices).astype(np.uint8)
    restored = (cv2.dilate(kept_seed, kernel, iterations=1) > 0) & mask
    return (restored, {'component_cleanup_applied': True, 'removed_component_count': len(removed_indices), 'kept_component_count': len(keep_indices), 'main_component_bbox': [main_x, main_y, main_width, main_height]})

def _trouser_gap_profile(raw_mask):
    """Đo khoảng trống thật giữa hai ống từ raw segmentation làm topology prior."""
    (ys, xs) = np.where(raw_mask)
    if len(xs) < 200:
        return None
    (x1, x2, y1, y2) = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    width = max(1, x2 - x1 + 1)
    height = max(1, y2 - y1 + 1)
    center = (x1 + x2) / 2.0
    minimum_run = max(3, int(width * 0.08))
    rows = []
    for row_y in range(int(y1 + 0.18 * height), y2 + 1):
        runs = [run for run in _foreground_runs(raw_mask[row_y]) if run[1] - run[0] + 1 >= minimum_run]
        left_runs = [run for run in runs if run[1] < center]
        right_runs = [run for run in runs if run[0] > center]
        if not left_runs or not right_runs:
            continue
        left = max(left_runs, key=lambda run: run[1])
        right = min(right_runs, key=lambda run: run[0])
        (gap_left, gap_right) = (left[1] + 1, right[0] - 1)
        gap_width = gap_right - gap_left + 1
        if 1 <= gap_width <= 0.38 * width:
            rows.append((row_y, gap_left, gap_right))
    minimum_rows = max(6, int(height * 0.035))
    if len(rows) < minimum_rows:
        return None
    return {'rows': rows, 'bbox': [x1, y1, x2, y2], 'center': center, 'width': width, 'height': height}

def _solidify_light_trousers(refined_mask, raw_mask, lab_image, label):
    """Vá phần rỗng bên trong quần nhưng tuyệt đối không ngoại suy biên ngoài.

    ``refined_mask`` bám theo pixel ảnh; ``raw_mask`` cung cấp topology ban đầu
    của hai ống. Chỉ vá các khe giả rất mảnh nằm *bên trong* biên trái-phải đã
    quan sát ở cùng hàng; tuyệt đối không ngoại suy hoặc làm nở biên ngoài.
    """
    if label not in {'trousers', 'shorts'}:
        return (refined_mask.copy(), {'trouser_topology_applied': False})
    refined = refined_mask.astype(bool)
    raw = raw_mask.astype(bool)
    profile = _trouser_gap_profile(raw)
    if profile is None:
        return (refined_mask.copy(), {'trouser_topology_applied': False, 'trouser_topology_skip_reason': 'no_stable_raw_leg_gap', 'trouser_topology_mode': 'pixel_anchored'})
    candidate = (refined | raw).astype(np.uint8)
    (count, component_map, stats, _) = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if count > 2:
        refined_pixels = max(1, int(refined.sum()))
        component_scores = []
        for component_id in range(1, count):
            component = component_map == component_id
            overlap = int((component & refined).sum())
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            component_scores.append((overlap, area, component_id))
        component_scores.sort(reverse=True)
        keep = np.zeros_like(candidate, dtype=bool)
        (best_overlap, best_area, best_id) = component_scores[0]
        for (overlap, area, component_id) in component_scores:
            is_main = component_id == best_id
            is_supported = overlap >= max(16, int(0.002 * refined_pixels))
            is_material = area >= max(24, int(0.003 * best_area))
            if is_main or (is_supported and is_material):
                keep |= component_map == component_id
        candidate = keep.astype(np.uint8)
    rows = profile['rows']
    row_y = np.array([item[0] for item in rows], dtype=float)
    gap_left = np.array([item[1] for item in rows], dtype=float)
    gap_right = np.array([item[2] for item in rows], dtype=float)
    (x1, y1, x2, raw_bottom) = profile['bbox']
    garment_width = profile['width']

    def rolling_median(values, window):
        radius = max(1, window // 2)
        return np.array([np.median(values[max(0, index - radius):min(len(values), index + radius + 1)]) for index in range(len(values))], dtype=float)
    smoothing_window = max(9, min(31, int(profile['height'] * 0.03) | 1))
    gap_left = rolling_median(gap_left, smoothing_window)
    gap_right = rolling_median(gap_right, smoothing_window)
    candidate_bool = candidate.astype(bool)
    row_span_guard = np.zeros_like(candidate_bool)
    for y in range(candidate_bool.shape[0]):
        xs = np.where(candidate_bool[y])[0]
        if len(xs):
            row_span_guard[y, int(xs.min()):int(xs.max()) + 1] = True
    repair_width = max(5, min(51, int(round(garment_width * 0.08)) | 1))
    repair_height = max(7, min(41, int(round(profile['height'] * 0.025)) | 1))
    horizontal_close = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (repair_width, 3))).astype(bool)
    vertical_close = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, repair_height))).astype(bool)
    repair_roi = np.zeros_like(candidate_bool)
    repair_start = max(y1, int(row_y.min()) - repair_height)
    repair_roi[repair_start:, x1:x2 + 1] = True
    repair_candidates = (horizontal_close | vertical_close) & ~candidate_bool & row_span_guard & repair_roi
    (repair_count, repair_labels, repair_stats, _) = cv2.connectedComponentsWithStats(repair_candidates.astype(np.uint8), connectivity=8)
    approved_repair = np.zeros_like(candidate_bool)
    candidate_area = max(1, int(candidate_bool.sum()))
    maximum_repair_area = max(48, int(candidate_area * 0.025))
    tiny_repair_area = max(24, int(candidate_area * 0.0025))
    repair_budget = max(64, int(candidate_area * 0.025))
    eligible_repairs = []
    for component_id in range(1, repair_count):
        area = int(repair_stats[component_id, cv2.CC_STAT_AREA])
        component_width = int(repair_stats[component_id, cv2.CC_STAT_WIDTH])
        component_height = int(repair_stats[component_id, cv2.CC_STAT_HEIGHT])
        short_side = min(component_width, component_height)
        long_side = max(component_width, component_height, 1)
        slenderness = short_side / long_side
        is_slender = slenderness <= 0.48
        if area <= maximum_repair_area and (is_slender or area <= tiny_repair_area):
            eligible_repairs.append((slenderness, -area, component_id, area))
    repair_used = 0
    for (_, _, component_id, area) in sorted(eligible_repairs):
        if repair_used + area > repair_budget:
            continue
        approved_repair |= repair_labels == component_id
        repair_used += area
    solid = candidate_bool | approved_repair
    foreground_y = np.where(solid)[0]
    product_bottom = int(foreground_y.max()) if len(foreground_y) else raw_bottom
    gap_start = int(row_y.min())
    gap_mask = np.zeros_like(solid)
    observed_bottom = min(int(row_y.max()), product_bottom)
    for y in range(gap_start, observed_bottom + 1):
        left = int(round(np.interp(y, row_y, gap_left)))
        right = int(round(np.interp(y, row_y, gap_right)))
        left = max(x1, min(x2, left))
        right = max(left, min(x2, right))
        gap_mask[y, left:right + 1] = True
    tail_count = max(12, min(len(rows), len(rows) // 2))
    tail_y = row_y[-tail_count:]
    if len(np.unique(tail_y)) >= 2:
        left_fit = np.polyfit(tail_y, gap_left[-tail_count:], 1)
        right_fit = np.polyfit(tail_y, gap_right[-tail_count:], 1)
        left_fit[0] = np.clip(left_fit[0], -0.45, 0.45)
        right_fit[0] = np.clip(right_fit[0], -0.45, 0.45)
    else:
        left_fit = np.array([0.0, float(np.median(gap_left))])
        right_fit = np.array([0.0, float(np.median(gap_right))])
    extrapolated_rows = 0
    contrast_rows = 0
    for y in range(observed_bottom + 1, product_bottom + 1):
        left = int(round(np.polyval(left_fit, y)))
        right = int(round(np.polyval(right_fit, y)))
        left = max(x1, min(x2, left))
        right = max(left, min(x2, right))
        extrapolated_rows += 1
        band = max(4, int(garment_width * 0.05))
        left_band = lab_image[y, max(x1, left - 3 * band):max(x1, left - band)]
        right_band = lab_image[y, min(x2 + 1, right + band):min(x2 + 1, right + 3 * band)]
        reference_pixels = np.concatenate([left_band, right_band], axis=0)
        corridor = lab_image[y, left:right + 1]
        if len(reference_pixels) == 0 or len(corridor) == 0:
            continue
        reference = np.median(reference_pixels, axis=0)
        garment_spread = float(np.percentile(np.linalg.norm(reference_pixels - reference, axis=1), 80))
        corridor_distance = float(np.median(np.linalg.norm(corridor - reference, axis=1)))
        if corridor_distance >= max(7.0, garment_spread + 4.0):
            gap_mask[y, left:right + 1] = True
            contrast_rows += 1
    approved_repair &= ~gap_mask
    result = (candidate_bool | approved_repair) & ~gap_mask
    external_added = result & ~candidate_bool & ~row_span_guard
    if external_added.any():
        result[external_added] = False
    return (result, {'trouser_topology_applied': True, 'trouser_topology_mode': 'pixel_anchored', 'trouser_outer_envelope_extrapolated': False, 'trouser_raw_gap_rows': len(rows), 'trouser_gap_extrapolated_rows': extrapolated_rows, 'trouser_gap_contrast_rows': contrast_rows, 'trouser_gap_pixels': int((gap_mask & solid).sum()), 'trouser_internal_repair_pixels': int(approved_repair.sum()), 'trouser_internal_repair_budget': repair_budget, 'trouser_internal_repair_fraction': round(float(approved_repair.sum()) / candidate_area, 4), 'trouser_repair_kernel_wh': [repair_width, repair_height], 'trouser_external_added_pixels': int(external_added.sum())})

def _fit_quadratic_border_background(lab, border, max_samples=25000):
    """Ước lượng nền studio trơn có gradient/vignette từ các pixel ở viền ảnh.

    Mặt phẳng bậc hai theo (x, y) mô tả được cả gradient tuyến tính lẫn vignette
    nhẹ. Hai lượt least-squares loại bớt vật thể vô tình chạm viền nhưng không
    cố giải thích nền có texture; trường hợp đó sẽ bị quality gate bên dưới từ chối.
    """
    (height, width) = lab.shape[:2]
    (yy, xx) = np.indices((height, width), dtype=np.float32)
    x = 2.0 * xx / max(width - 1, 1) - 1.0
    y = 2.0 * yy / max(height - 1, 1) - 1.0
    border_x = x[border]
    border_y = y[border]
    border_lab = lab[border].astype(np.float32)
    sample_count = len(border_x)
    if sample_count > max_samples:
        sample_indices = np.linspace(0, sample_count - 1, max_samples, dtype=np.int64)
        border_x = border_x[sample_indices]
        border_y = border_y[sample_indices]
        border_lab = border_lab[sample_indices]
    design = np.column_stack([np.ones_like(border_x), border_x, border_y, border_x * border_y, border_x * border_x, border_y * border_y]).astype(np.float32)
    (coefficients, *_) = np.linalg.lstsq(design, border_lab, rcond=None)
    fit_residual = np.linalg.norm(design @ coefficients - border_lab, axis=1)
    inlier_limit = float(np.percentile(fit_residual, 85))
    inliers = fit_residual <= max(inlier_limit, 2.0)
    if int(inliers.sum()) >= max(24, 4 * design.shape[1]):
        (coefficients, *_) = np.linalg.lstsq(design[inliers], border_lab[inliers], rcond=None)
    features = (np.ones_like(x), x, y, x * y, x * x, y * y)
    background = np.empty_like(lab, dtype=np.float32)
    for channel in range(3):
        predicted = np.zeros_like(x, dtype=np.float32)
        for (feature, coefficient) in zip(features, coefficients[:, channel]):
            predicted += feature * float(coefficient)
        background[:, :, channel] = np.clip(predicted, 0.0, 255.0)
    return background

def _studio_background_cleanup(mask, image, label):
    """Tinh chỉnh hai chiều bằng nền studio: bỏ nền dư và phục hồi garment bị thiếu."""
    metadata = {'background_cleanup_applied': False, 'background_cleanup_skip_reason': 'disabled'}
    if image is None or label not in LABEL_TO_DISPLAY:
        metadata['background_cleanup_skip_reason'] = 'not_applicable'
        return (mask.copy(), metadata)
    rgb = np.asarray(image.convert('RGB') if hasattr(image, 'convert') else image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        metadata['background_cleanup_skip_reason'] = 'invalid_image'
        return (mask.copy(), metadata)
    rgb = rgb[:, :, :3].astype(np.uint8)
    (height, width) = mask.shape
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border_size = max(4, int(round(min(height, width) * 0.035)))
    border = np.zeros((height, width), dtype=bool)
    border[:border_size, :] = True
    border[-border_size:, :] = True
    border[:, :border_size] = True
    border[:, -border_size:] = True
    border_pixels = lab[border]
    constant_background = np.median(border_pixels, axis=0)
    constant_border_distances = np.linalg.norm(border_pixels - constant_background, axis=1)
    global_uniformity = float(np.percentile(constant_border_distances, 90))
    quadratic_background = _fit_quadratic_border_background(lab, border)
    quadratic_distance_map = np.linalg.norm(lab - quadratic_background, axis=2)
    quadratic_border_distances = quadratic_distance_map[border]
    surface_residual_p90 = float(np.percentile(quadratic_border_distances, 90))
    metadata['background_uniformity'] = round(global_uniformity, 2)
    metadata['background_global_uniformity'] = round(global_uniformity, 2)
    metadata['background_surface_residual_p90'] = round(surface_residual_p90, 2)
    use_quadratic_background = global_uniformity > 14.0 and surface_residual_p90 <= 12.0 and (surface_residual_p90 + 2.0 < global_uniformity)
    if global_uniformity > 20.0 and (not use_quadratic_background):
        metadata['background_model'] = 'rejected_complex_border'
        metadata['background_cleanup_skip_reason'] = 'non_uniform_border'
        return (mask.copy(), metadata)
    if use_quadratic_background:
        background_map = quadratic_background
        distance_map = quadratic_distance_map
        border_distances = quadratic_border_distances
        metadata['background_model'] = 'quadratic_border'
    else:
        background_map = np.broadcast_to(constant_background.reshape(1, 1, 3), lab.shape)
        distance_map = np.linalg.norm(lab - background_map, axis=2)
        border_distances = constant_border_distances
        metadata['background_model'] = 'constant_border'
    distance_threshold = float(np.clip(np.percentile(border_distances, 98) + 6.0, 10.0, 30.0))
    raw_area = int(mask.sum())
    chroma_distance_map = np.linalg.norm(lab[:, :, 1:3] - background_map[:, :, 1:3], axis=2)
    border_chroma_distances = chroma_distance_map[border]
    chroma_threshold = float(np.clip(np.percentile(border_chroma_distances, 98) + 3.0, 4.0, 18.0))
    raw_chroma_distances = chroma_distance_map[mask.astype(bool)]
    garment_chroma_separation = float(np.median(raw_chroma_distances) if len(raw_chroma_distances) else 0.0)
    shadow_aware = garment_chroma_separation >= max(10.0, 1.35 * chroma_threshold)
    if shadow_aware:
        adaptive_chroma_threshold = max(chroma_threshold, min(18.0, 0.28 * garment_chroma_separation))
        chroma_foreground = chroma_distance_map > adaptive_chroma_threshold
        raw_supported = mask.astype(bool) & (distance_map > 0.55 * distance_threshold)
        image_foreground = chroma_foreground | raw_supported
    else:
        image_foreground = distance_map > distance_threshold
    metadata['background_distance_threshold'] = round(distance_threshold, 2)
    metadata['background_chroma_threshold'] = round(chroma_threshold, 2)
    metadata['garment_chroma_separation'] = round(garment_chroma_separation, 2)
    metadata['shadow_aware_foreground'] = bool(shadow_aware)
    kernel_size = max(3, min(11, int(round(min(height, width) * 0.007))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    foreground_u8 = cv2.morphologyEx(image_foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    foreground_u8 = cv2.morphologyEx(foreground_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    (contours, _) = cv2.findContours(foreground_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected_seed = np.zeros_like(foreground_u8)
    minimum_contour_area = max(32.0, raw_area * 0.0008)
    minimum_overlap = max(24, int(raw_area * 0.0015))
    selected_contours = 0
    for contour in contours:
        if cv2.contourArea(contour) < minimum_contour_area:
            continue
        contour_mask = np.zeros_like(foreground_u8)
        cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
        contour_pixels = int(contour_mask.sum())
        overlap = int((contour_mask.astype(bool) & mask).sum())
        overlap_ratio = overlap / max(contour_pixels, 1)
        if overlap >= minimum_overlap and overlap_ratio >= 0.08:
            selected_seed |= foreground_u8 & contour_mask
            selected_contours += 1
    if selected_contours == 0:
        metadata['background_cleanup_skip_reason'] = 'no_raw_anchored_silhouette'
        return (mask.copy(), metadata)
    padded = np.pad(selected_seed, 1, mode='constant', constant_values=0).astype(np.uint8)
    flooded = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 1)
    enclosed_holes = (flooded == 0).astype(np.uint8)
    (hole_count, hole_labels, hole_stats, _) = cv2.connectedComponentsWithStats(enclosed_holes, 8)
    maximum_fill_area = max(64, int(selected_seed.sum() * 0.005))
    small_hole_indices = [index for index in range(1, hole_count) if int(hole_stats[index, cv2.CC_STAT_AREA]) <= maximum_fill_area]
    small_holes = np.isin(hole_labels, small_hole_indices)
    silhouette = (padded.astype(bool) | small_holes)[1:-1, 1:-1].astype(np.uint8)
    if label in {'trousers', 'shorts'}:
        refined = silhouette.astype(bool)
    else:
        refined = cv2.dilate(silhouette, np.ones((3, 3), np.uint8), iterations=1) > 0
    (refined, trouser_topology) = _solidify_light_trousers(refined, mask, lab, label)

    def refinement_metrics(value):
        overlap = int((value & mask).sum())
        area = int(value.sum())
        added = int((value & ~mask).sum())
        removed = int((mask & ~value).sum())
        return {'raw_overlap': overlap, 'raw_retained_ratio': overlap / max(raw_area, 1), 'refined_area': area, 'area_ratio': area / max(raw_area, 1), 'added_pixels': added, 'removed_pixels': removed, 'added_fraction': added / max(raw_area, 1), 'removed_fraction': removed / max(raw_area, 1)}
    metrics = refinement_metrics(refined)
    metadata['upper_expansion_guard_applied'] = False
    if label in UPPER_LABELS and metrics['added_fraction'] > 0.45:
        (raw_y_guard, raw_x_guard) = np.where(mask)
        if len(raw_x_guard) and len(raw_y_guard):
            raw_width = int(raw_x_guard.max() - raw_x_guard.min() + 1)
            raw_height = int(raw_y_guard.max() - raw_y_guard.min() + 1)
            guard_radius = max(5, min(41, int(round(0.045 * max(raw_width, raw_height)))))
            guard_size = 2 * guard_radius + 1
            raw_support = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard_size, guard_size)), iterations=1).astype(bool)
            refined = mask.astype(bool) | refined & raw_support
            metrics = refinement_metrics(refined)
            metadata['upper_expansion_guard_applied'] = True
            metadata['upper_expansion_guard_radius_px'] = guard_radius
    raw_overlap = metrics['raw_overlap']
    raw_retained_ratio = metrics['raw_retained_ratio']
    refined_area = metrics['refined_area']
    area_ratio = metrics['area_ratio']
    added_pixels = metrics['added_pixels']
    removed_pixels = metrics['removed_pixels']
    added_fraction = metrics['added_fraction']
    removed_fraction = metrics['removed_fraction']
    (raw_y, raw_x) = np.where(mask)
    (refined_y, refined_x) = np.where(refined)
    raw_top = int(raw_y.min()) if len(raw_y) else 0
    raw_bottom = int(raw_y.max()) if len(raw_y) else 0
    refined_top = int(refined_y.min()) if len(refined_y) else 0
    refined_bottom = int(refined_y.max()) if len(refined_y) else 0
    vertical_extension = max(0, refined_bottom - raw_bottom)
    raw_bbox = [int(raw_x.min()), raw_top, int(raw_x.max()), raw_bottom] if len(raw_x) else None
    refined_bbox = [int(refined_x.min()), refined_top, int(refined_x.max()), refined_bottom] if len(refined_x) else None
    metadata.update({**trouser_topology, 'background_selected_contours': selected_contours, 'background_raw_retained_ratio': round(float(raw_retained_ratio), 4), 'background_area_ratio': round(float(area_ratio), 4), 'background_added_fraction': round(float(added_fraction), 4), 'background_removed_fraction': round(float(removed_fraction), 4), 'background_vertical_extension_px': vertical_extension, 'background_vertical_extension_ratio': round(vertical_extension / max(height, 1), 4), 'raw_bbox_xyxy': raw_bbox, 'refined_bbox_xyxy': refined_bbox, 'raw_height_ratio': round((raw_bottom - raw_top + 1) / max(height, 1), 4), 'refined_height_ratio': round((refined_bottom - refined_top + 1) / max(height, 1), 4)})
    if raw_retained_ratio < 0.5:
        metadata['background_cleanup_skip_reason'] = 'unsafe_raw_loss'
        return (mask.copy(), metadata)
    if area_ratio > 2.5:
        metadata['background_cleanup_skip_reason'] = 'unsafe_expansion'
        return (mask.copy(), metadata)
    if label in UPPER_LABELS and added_fraction > 0.45:
        metadata['background_cleanup_skip_reason'] = 'unsafe_upper_expansion'
        return (mask.copy(), metadata)
    if added_fraction < 0.005 and removed_fraction < 0.005:
        metadata['background_cleanup_skip_reason'] = 'no_material_change'
        return (mask.copy(), metadata)
    metadata['background_cleanup_applied'] = True
    metadata['background_cleanup_skip_reason'] = None
    return (refined, metadata)

def refine_garment_mask(raw_mask, label, image=None, refine_background=False):
    """Kết hợp màu nền studio và hình học; ưu tiên bỏ qua nếu độ an toàn thấp."""
    raw = raw_mask.astype(bool)
    (background_cleaned, background_cleanup) = _studio_background_cleanup(raw, image if refine_background else None, label)
    (component_cleaned, component_cleanup) = _remove_distant_dress_components(background_cleaned, label)
    (refined, branch_cleanup) = _dress_lower_branch_cleanup(component_cleaned, label)
    raw_area = int(raw.sum())
    refined_area = int(refined.sum())
    removed_fraction = float((raw & ~refined).sum() / max(raw_area, 1))
    added_fraction = float((refined & ~raw).sum() / max(raw_area, 1))
    status = 'pass'
    if removed_fraction >= 0.2 or added_fraction >= 0.75:
        status = 'review_required'
    elif removed_fraction >= 0.01 and added_fraction >= 0.01:
        status = 'auto_refined'
    elif added_fraction >= 0.01:
        status = 'auto_extended'
    elif removed_fraction >= 0.01:
        status = 'auto_cleaned'
    quality = {'status': status, 'label': label, 'raw_components': _component_count(raw), 'refined_components': _component_count(refined), 'raw_area': raw_area, 'refined_area': refined_area, **background_cleanup, **component_cleanup, **branch_cleanup, 'removed_fraction': round(float(removed_fraction), 4), 'added_fraction': round(float(added_fraction), 4), 'area_change_fraction': round(float((refined_area - raw_area) / max(raw_area, 1)), 4)}
    return (refined, quality)

def _single_detection_raw_mask(detection, width, height):
    """Bước _single_detection_raw_mask: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    result = detection['result']
    index = detection['index']
    if result.masks is None or len(result.masks) <= index:
        raise ValueError(f"Detection '{detection['label']}' không có segmentation mask.")
    raw = result.masks.data[index].cpu().numpy()
    if raw.shape != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    return raw

def detection_mask(detection, image_size, image=None, refine_background=False):
    """Bước detection_mask: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    composite_members = detection.get('_composite_members', [])
    cache_key = (tuple(image_size), bool(refine_background), detection['label'], len(composite_members))
    if detection.get('_mask_cache_key') == cache_key and '_mask_cache' in detection:
        return detection['_mask_cache'].copy()
    (width, height) = image_size
    if composite_members:
        raw = np.maximum.reduce([_single_detection_raw_mask(member, width, height) for member in composite_members])
    else:
        raw = _single_detection_raw_mask(detection, width, height)
    (refined, quality) = refine_garment_mask(raw >= 0.5, detection['label'], image=image, refine_background=refine_background)
    quality['category_resolution_source'] = detection.get('category_resolution_source', 'deepfashion2')
    quality['model_label'] = detection.get('model_label', detection['label'])
    quality['product_detection_reselection'] = detection.get('product_detection_reselection', 'original_primary')
    detection['mask_quality'] = quality
    detection['_mask_cache_key'] = cache_key
    detection['_mask_cache'] = refined.copy()
    return refined

def _tight_mask_bbox(mask):
    """Bước _tight_mask_bbox: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    (ys, xs) = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

def _normalized_variant_mask(mask, size=384, bbox=None):
    """Chuẩn hóa silhouette về bbox vuông để so topology, không so màu/canvas."""
    bbox = bbox or _tight_mask_bbox(mask)
    if bbox is None:
        return (None, None)
    (x1, y1, x2, y2) = bbox
    crop = mask[y1:y2, x1:x2].astype(np.uint8)
    normalized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST) > 0
    return (normalized, bbox)

def _variant_trouser_gap_mask(mask):
    """Lấy riêng khe thật giữa hai ống trong hệ tọa độ của từng biến thể."""
    profile = _trouser_gap_profile(mask.astype(bool))
    gap = np.zeros_like(mask, dtype=bool)
    if profile is None:
        return (gap, None)
    rows = profile['rows']
    row_y = np.asarray([item[0] for item in rows], dtype=float)
    left = np.asarray([item[1] for item in rows], dtype=float)
    right = np.asarray([item[2] for item in rows], dtype=float)
    window = max(7, min(25, int(profile['height'] * 0.025) | 1))
    radius = window // 2

    def smooth(values):
        return np.asarray([np.median(values[max(0, i - radius):min(len(values), i + radius + 1)]) for i in range(len(values))], dtype=float)
    left = smooth(left)
    right = smooth(right)
    (x1, y1, x2, y2) = profile['bbox']
    start_y = int(row_y.min())
    observed_bottom = int(row_y.max())
    for y in range(start_y, y2 + 1):
        sample_y = min(y, observed_bottom)
        gap_left = int(round(np.interp(sample_y, row_y, left)))
        gap_right = int(round(np.interp(sample_y, row_y, right)))
        gap_left = max(x1, min(x2, gap_left))
        gap_right = max(gap_left, min(x2, gap_right))
        gap[y, gap_left:gap_right + 1] = True
    return (gap, {'start_y': start_y, 'bottom_y': y2, 'rows': int(y2 - start_y + 1), 'area': int(gap.sum())})

def _consensus_variant_bboxes(bboxes, shapes):
    """Neo cùng sản phẩm/cùng góc vào bbox canvas chung, độc lập bbox mask lỗi."""
    relative = []
    for (bbox, shape) in zip(bboxes, shapes):
        (height, width) = shape
        (x1, y1, x2, y2) = bbox
        relative.append([x1 / max(width, 1), y1 / max(height, 1), x2 / max(width, 1), y2 / max(height, 1)])
    relative = np.asarray(relative, dtype=np.float32)
    horizontal_centers = (relative[:, 0] + relative[:, 2]) / 2.0
    horizontal_center_spread = float(np.ptp(horizontal_centers))
    top_edge_spread = float(np.ptp(relative[:, 1]))
    same_view_alignment = bool(horizontal_center_spread <= 0.06 and top_edge_spread <= 0.08)
    median_relative = np.median(relative, axis=0)
    targets = []
    for (bbox, shape) in zip(bboxes, shapes):
        (height, width) = shape
        if same_view_alignment:
            x1 = int(round(float(median_relative[0]) * width))
            y1 = int(round(float(median_relative[1]) * height))
            x2 = int(round(float(median_relative[2]) * width))
            y2 = int(round(float(median_relative[3]) * height))
            (x1, x2) = (max(0, x1), min(width, max(x1 + 1, x2)))
            (y1, y2) = (max(0, y1), min(height, max(y1 + 1, y2)))
            targets.append([x1, y1, x2, y2])
        else:
            targets.append(list(bbox))
    return (targets, {'mode': 'consensus_canvas' if same_view_alignment else 'individual_bbox', 'horizontal_center_spread': round(horizontal_center_spread, 4), 'top_edge_spread': round(top_edge_spread, 4), 'alignment_anchors': 'horizontal_center_and_top_edge', 'median_relative_bbox': [round(float(value), 5) for value in median_relative]})

def synchronize_product_variant_masks(masks, qualities, label=None, minimum_iou=0.78):
    """Đồng bộ biên ngoài giữa biến thể nhưng giữ topology riêng từng ảnh.

    Với quần, template chỉ biểu quyết outer envelope. Khe giữa hai ống được đo
    lại từ từng mask rồi map vào bbox canvas đã đồng bộ, nên không bị lấp hoặc
    biến thành khe của một màu khác. Bbox canvas chung ngăn một mask raw bị co
    làm riêng biến thể đó nhỏ/ngắn hơn dù các ảnh cùng góc chụp.
    """
    if len(masks) < 2:
        return ([mask.copy() for mask in masks], {'applied': False, 'skip_reason': 'single_variant'})
    bboxes = [_tight_mask_bbox(mask) for mask in masks]
    if any((bbox is None for bbox in bboxes)):
        return ([value.copy() for value in masks], {'applied': False, 'skip_reason': 'empty_variant_mask'})
    explicit_trouser_label = label in {'trousers', 'shorts'}
    topology_votes = []
    topology_vote_details = []
    dress_labels = {'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress', 'dress', 'jumpsuit'}
    for (mask, bbox) in zip(masks, bboxes):
        (x1, y1, x2, y2) = bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        profile = _trouser_gap_profile(mask.astype(bool))
        aspect_ratio = box_height / box_width
        stable_gap = False
        gap_span_ratio = 0.0
        gap_start_ratio = 1.0
        gap_reaches_bottom = False
        if profile is not None:
            observed_rows = np.asarray([item[0] for item in profile['rows']], dtype=np.int32)
            gap_span_ratio = (int(observed_rows.max()) - int(observed_rows.min()) + 1) / box_height
            gap_start_ratio = (int(observed_rows.min()) - y1) / box_height
            gap_reaches_bottom = int(observed_rows.max()) >= y2 - max(3, int(0.035 * box_height))
            stable_gap = bool(aspect_ratio >= 1.3 and gap_span_ratio >= 0.28 and (gap_start_ratio <= 0.62) and gap_reaches_bottom)
        topology_votes.append(stable_gap)
        topology_vote_details.append({'aspect_ratio': round(float(aspect_ratio), 4), 'gap_span_ratio': round(float(gap_span_ratio), 4), 'gap_start_ratio': round(float(gap_start_ratio), 4), 'gap_reaches_bottom': bool(gap_reaches_bottom), 'vote': bool(stable_gap)})
    required_votes = max(2, int(np.ceil(len(masks) * 0.6)))
    topology_vote_count = int(sum(topology_votes))
    mask_consensus_trouser = bool(label not in dress_labels and topology_vote_count >= required_votes)
    is_trouser = explicit_trouser_label or mask_consensus_trouser
    topology_source = 'session_label' if explicit_trouser_label else 'mask_consensus' if mask_consensus_trouser else 'not_trousers'
    (gap_masks, gap_details, synchronization_sources) = ([], [], [])
    for mask in masks:
        if is_trouser:
            (gap, detail) = _variant_trouser_gap_mask(mask)
        else:
            (gap, detail) = (np.zeros_like(mask, dtype=bool), None)
        gap_masks.append(gap)
        gap_details.append(detail)
        synchronization_sources.append(mask.astype(bool) | gap)
    normalized = []
    for (source_mask, bbox) in zip(synchronization_sources, bboxes):
        (item, _) = _normalized_variant_mask(source_mask, bbox=bbox)
        normalized.append(item)
    pair_ious = []
    for left_index in range(len(normalized)):
        for right_index in range(left_index + 1, len(normalized)):
            (left, right) = (normalized[left_index], normalized[right_index])
            union = int((left | right).sum())
            pair_ious.append(float((left & right).sum() / max(union, 1)))
    median_iou = float(np.median(pair_ious)) if pair_ious else 1.0
    minimum_pair_iou = float(min(pair_ious)) if pair_ious else 1.0
    if median_iou < minimum_iou or minimum_pair_iou < minimum_iou - 0.1:
        return ([mask.copy() for mask in masks], {'applied': False, 'skip_reason': 'different_view_or_unstable_shape', 'median_iou_before': round(median_iou, 4), 'minimum_iou_before': round(minimum_pair_iou, 4)})
    raw_weights = []
    for quality in qualities:
        change = float(quality.get('added_fraction', 0.0)) + float(quality.get('removed_fraction', 0.0))
        raw_weights.append(1.0 / max(0.08, 0.08 + change))
    weights = np.asarray(raw_weights, dtype=np.float32)
    median_weight = float(np.median(weights))
    weights = np.clip(weights, median_weight * 0.5, median_weight * 2.0)
    weights /= max(float(weights.sum()), 1e-06)
    probability = np.zeros_like(normalized[0], dtype=np.float32)
    for (weight, normalized_mask) in zip(weights, normalized):
        probability += float(weight) * normalized_mask.astype(np.float32)
    outer_template = probability >= 0.5
    (count, labels, stats, _) = cv2.connectedComponentsWithStats(outer_template.astype(np.uint8), connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        outer_template = labels == largest
    if not outer_template.any():
        return ([mask.copy() for mask in masks], {'applied': False, 'skip_reason': 'empty_consensus_template', 'median_iou_before': round(median_iou, 4)})
    (target_bboxes, bbox_metadata) = _consensus_variant_bboxes(bboxes, [mask.shape for mask in masks])
    synchronized = []
    change_metrics = []
    gap_preserved = []
    for (original, original_bbox, target_bbox, gap) in zip(masks, bboxes, target_bboxes, gap_masks):
        (target_x1, target_y1, target_x2, target_y2) = target_bbox
        target_width = max(1, target_x2 - target_x1)
        target_height = max(1, target_y2 - target_y1)
        mapped_outer = cv2.resize(outer_template.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST).astype(bool)
        mapped_gap = np.zeros_like(mapped_outer, dtype=bool)
        if is_trouser and gap.any():
            (x1, y1, x2, y2) = original_bbox
            gap_crop = gap[y1:y2, x1:x2].astype(np.uint8)
            mapped_gap = cv2.resize(gap_crop, (target_width, target_height), interpolation=cv2.INTER_NEAREST).astype(bool)
        mapped = mapped_outer & ~mapped_gap
        output = np.zeros_like(original, dtype=bool)
        output[target_y1:target_y2, target_x1:target_x2] = mapped
        synchronized.append(output)
        gap_preserved.append(int(mapped_gap.sum()))
        base_area = max(1, int(original.sum()))
        changed = int((output ^ original).sum())
        change_metrics.append({'changed_fraction': round(changed / base_area, 4), 'added_fraction': round(float((output & ~original).sum()) / base_area, 4), 'removed_fraction': round(float((original & ~output).sum()) / base_area, 4), 'source_bbox_xyxy': list(original_bbox), 'target_bbox_xyxy': list(target_bbox)})
    normalized_after = [_normalized_variant_mask(mask)[0] for mask in synchronized]
    after_ious = []
    for left_index in range(len(normalized_after)):
        for right_index in range(left_index + 1, len(normalized_after)):
            (left, right) = (normalized_after[left_index], normalized_after[right_index])
            after_ious.append(float((left & right).sum() / max(int((left | right).sum()), 1)))
    metadata = {'applied': True, 'skip_reason': None, 'median_iou_before': round(median_iou, 4), 'minimum_iou_before': round(minimum_pair_iou, 4), 'median_iou_after': round(float(np.median(after_ious)), 4), 'template_support_threshold': 0.5, 'variant_weights': [round(float(value), 4) for value in weights], 'bbox_alignment': bbox_metadata, 'trouser_gap_individualized': bool(is_trouser), 'trouser_topology_source': topology_source, 'trouser_topology_vote_count': topology_vote_count, 'trouser_topology_required_votes': required_votes, 'trouser_topology_vote_details': topology_vote_details, 'trouser_gap_details': gap_details, 'trouser_gap_pixels_after': gap_preserved, 'changes': change_metrics}
    return (synchronized, metadata)

def base_fabric_mask(mask, label):
    """Bước base_fabric_mask: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    mask_u8 = mask.astype(np.uint8)
    (ys, xs) = np.where(mask_u8 > 0)
    if len(xs) < 200:
        return mask
    (x1, x2, y1, y2) = (xs.min(), xs.max(), ys.min(), ys.max())
    kernel_size = max(3, int(round(min(x2 - x1 + 1, y2 - y1 + 1) * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    eroded = cv2.erode(mask_u8, np.ones((kernel_size, kernel_size), np.uint8), iterations=1).astype(bool)
    if label in UPPER_LABELS | DRESS_LABELS:
        width = max(1, x2 - x1 + 1)
        height = max(1, y2 - y1 + 1)
        central = np.zeros_like(eroded)
        (cx1, cx2) = (int(x1 + 0.25 * width), int(x1 + 0.75 * width))
        (cy1, cy2) = (int(y1 + 0.18 * height), int(y1 + 0.72 * height))
        central[cy1:cy2 + 1, cx1:cx2 + 1] = True
        safe = eroded & ~central
        if safe.sum() >= max(500, int(eroded.sum() * 0.18)):
            return safe
    return eroded if eroded.sum() >= 200 else mask

def nearest_color_name(rgb):
    """Bước nearest_color_name: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    rgb_array = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB).reshape(3).astype(float)
    (best_name, best_distance) = (None, float('inf'))
    for (name, hex_code) in COLOR_PALETTE.items():
        candidate = np.array([int(hex_code[1:3], 16), int(hex_code[3:5], 16), int(hex_code[5:7], 16)], dtype=np.uint8).reshape(1, 1, 3)
        candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_RGB2LAB).reshape(3).astype(float)
        distance = float(np.linalg.norm(lab - candidate_lab))
        if distance < best_distance:
            (best_name, best_distance) = (name, distance)
    return best_name

def merge_color_clusters(centers, counts):
    """Bước merge_color_clusters: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    clusters = []
    total = max(1, int(np.sum(counts)))
    for (center, count) in sorted(zip(centers, counts), key=lambda pair: pair[1], reverse=True):
        share = float(count / total)
        if share < 0.07:
            continue
        name = nearest_color_name(center)
        match = next((item for item in clusters if item['name'] == name and np.linalg.norm(item['rgb'] - center) < 60), None)
        if match is None:
            clusters.append({'rgb': center.astype(float), 'weight': share, 'name': name})
        else:
            combined = match['weight'] + share
            match['rgb'] = (match['rgb'] * match['weight'] + center * share) / combined
            match['weight'] = combined
    clusters.sort(key=lambda item: item['weight'], reverse=True)
    if clusters and clusters[0]['weight'] >= 0.82:
        clusters = clusters[:1]
    return clusters[:5]

def extract_base_colors(image, mask, label):
    """Bước extract_base_colors: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    rgb = np.asarray(image.convert('RGB'))
    sample_mask = base_fabric_mask(mask, label)
    pixels = rgb[sample_mask]
    if len(pixels) < 50:
        raise ValueError('Mask garment quá nhỏ để trích xuất màu.')
    rng = np.random.default_rng(42)
    if len(pixels) > 12000:
        pixels = pixels[rng.choice(len(pixels), 12000, replace=False)]
    quantized = pixels // 8 * 8 + 4
    unique_count = len(np.unique(quantized, axis=0))
    cluster_count = min(5, max(1, unique_count))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit(quantized)
    counts = np.bincount(kmeans.labels_, minlength=cluster_count)
    merged = merge_color_clusters(kmeans.cluster_centers_, counts)
    results = []
    retained = sum((item['weight'] for item in merged)) or 1.0
    for item in merged:
        clipped = np.clip(np.rint(item['rgb']), 0, 255).astype(int)
        results.append({'name': item['name'], 'hex': f'#{clipped[0]:02X}{clipped[1]:02X}{clipped[2]:02X}', 'share': round(item['weight'] / retained, 3)})
    return results

def visible_pair_y(pose, indices):
    """Bước visible_pair_y: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    values = [pose['xy'][idx, 1] for idx in indices if pose['conf'][idx] >= POSE_CONFIDENCE]
    return float(np.median(values)) if values else None

def body_position(y, pose):
    """Bước body_position: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    shoulder = visible_pair_y(pose, (5, 6))
    hip = visible_pair_y(pose, (11, 12))
    knee = visible_pair_y(pose, (13, 14))
    ankle = visible_pair_y(pose, (15, 16))
    landmarks = [(0.0, shoulder), (1.0, hip), (2.0, knee), (3.0, ankle)]
    landmarks = [(position, value) for (position, value) in landmarks if value is not None]
    if len(landmarks) < 2:
        return None
    landmarks.sort(key=lambda item: item[1])
    for ((p1, y1), (p2, y2)) in zip(landmarks, landmarks[1:]):
        if y1 <= y <= y2 and y2 > y1:
            return p1 + (y - y1) * (p2 - p1) / (y2 - y1)
    if y < landmarks[0][1]:
        ((p1, y1), (p2, y2)) = landmarks[:2]
    else:
        ((p1, y1), (p2, y2)) = landmarks[-2:]
    return p1 + (y - y1) * (p2 - p1) / max(y2 - y1, 1e-06)

def classify_fit(position, family, label):
    """Bước classify_fit: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    if position is None:
        return 'unknown'
    if family == 'upper_body':
        if position < 0.55:
            return 'cropped'
        if position < 0.88:
            return 'at_waist'
        if position < 1.22:
            return 'at_hip'
        return 'upper_thigh'
    if label == 'shorts':
        return 'upper_thigh' if position < 1.55 else 'above_knee' if position < 1.88 else 'at_knee'
    if label == 'trousers':
        if position < 2.18:
            return 'at_knee'
        if position < 2.62:
            return 'mid_calf'
        if position < 2.9:
            return 'lower_calf'
        return 'ankle_length'
    if position < 1.82:
        return 'above_knee'
    if position < 2.15:
        return 'at_knee'
    if position < 2.58:
        return 'below_knee'
    if position < 2.84:
        return 'mid_calf'
    if position < 3.08:
        return 'lower_calf'
    return 'ankle_length'

def analyze_model_reference(item):
    """Bước analyze_model_reference: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    image = Image.open(item['path']).convert('RGB')
    mask = detection_mask(item['detection'], image.size)
    ys = np.where(mask)[0]
    if len(ys) == 0:
        return {'file': item['original_name'], 'position': None, 'fit_code': 'unknown'}
    hem_y = float(np.percentile(ys, 98))
    position = body_position(hem_y, item['pose'])
    return {'file': item['original_name'], 'position': None if position is None else round(float(position), 3), 'fit_code': classify_fit(position, session_family, session_label), 'detected_label': item['detection']['label']}

def aggregate_reference_fit(model_results):
    """Bước aggregate_reference_fit: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu."""
    positions = [item['position'] for item in model_results if item['position'] is not None]
    if not positions:
        return {'code': 'unknown', 'label_vi': FIT_LABELS['unknown'], 'body_position': None, 'confidence': 'low', 'support': 0, 'warning': 'Không đo được vị trí gấu từ các ảnh tham chiếu.'}
    median_position = float(np.median(positions))
    fit_code = classify_fit(median_position, session_family, session_label)
    spread = float(np.max(positions) - np.min(positions)) if len(positions) > 1 else 0.0
    confidence = 'high' if len(positions) >= 2 and spread <= 0.3 else 'medium'
    warning = None
    if spread > 0.45:
        confidence = 'low'
        warning = 'Các ảnh người mẫu cho vị trí gấu không nhất quán; cần kiểm tra thủ công.'
    return {'code': fit_code, 'label_vi': FIT_LABELS[fit_code], 'body_position': round(median_position, 3), 'confidence': confidence, 'support': len(positions), 'spread': round(spread, 3), 'warning': warning}
