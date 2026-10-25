"""
==============================================================================
DeepFashion2 YOLOv8-seg Test & Preprocessing Script - YoFaSoShop_VTON
Tác giả: Senior Python & Deep Learning Specialist
Mục đích Tuần 4:
  1. Kiểm thử bóc tách phông nền ảnh sản phẩm thời trang shop (áo, quần, váy, đầm).
  2. Tạo ảnh PNG trong suốt (RGBA) chuẩn thương mại điện tử cho YoFaSo Shop.
  3. Trích xuất bảng màu chủ đạo (Color Palette K-Means) và Fashion Landmarks
     để phục vụ thuật toán chấm điểm thẩm mỹ (Color Harmony & Ratio) ở Tuần 5.
==============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 13 danh mục chuẩn của tập dữ liệu DeepFashion2
DEEPFASHION2_CLASSES = {
    0: "short_sleeved_shirt",
    1: "long_sleeved_shirt",
    2: "short_sleeved_outwear",
    3: "long_sleeved_outwear",
    4: "vest",
    5: "sling",
    6: "shorts",
    7: "trousers",
    8: "skirt",
    9: "short_sleeved_dress",
    10: "long_sleeved_dress",
    11: "vest_dress",
    12: "sling_dress"
}

# Ánh xạ sang danh mục phân loại đồ YoFaSo Shop
CATEGORY_MAPPING = {
    "short_sleeved_shirt": "top",
    "long_sleeved_shirt": "top",
    "short_sleeved_outwear": "outwear",
    "long_sleeved_outwear": "outwear",
    "vest": "top",
    "sling": "top",
    "shorts": "bottom",
    "trousers": "bottom",
    "skirt": "bottom",
    "short_sleeved_dress": "dress",
    "long_sleeved_dress": "dress",
    "vest_dress": "dress",
    "sling_dress": "dress"
}

def extract_dominant_colors(image_np, mask_np, num_colors=3):
    """
    Trích xuất bảng màu chủ đạo (Color Palette) từ vùng ảnh trang phục đã tách nền
    bằng thuật toán K-Means Clustering trên không gian màu RGB.
    """
    try:
        from sklearn.cluster import KMeans
        import numpy as np
    except ImportError:
        return [{"hex": "#000000", "ratio": 1.0, "rgb": [0, 0, 0]}]

    # Lấy các pixel thuộc vùng trang phục (mask > 0)
    foreground_pixels = image_np[mask_np > 0]
    if len(foreground_pixels) < 100:
        return []

    # Giảm số lượng mẫu pixel để tăng tốc độ K-Means
    if len(foreground_pixels) > 5000:
        indices = np.random.choice(len(foreground_pixels), 5000, replace=False)
        sample_pixels = foreground_pixels[indices]
    else:
        sample_pixels = foreground_pixels

    kmeans = KMeans(n_clusters=min(num_colors, len(sample_pixels)), random_state=42, n_init=5)
    kmeans.fit(sample_pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    total_count = len(labels)

    palette = []
    # Sắp xếp theo tỷ lệ xuất hiện giảm dần
    sorted_indices = np.argsort(-counts)
    for idx in sorted_indices:
        r, g, b = colors[idx][:3]
        hex_code = f"#{r:02x}{g:02x}{b:02x}".upper()
        ratio = round(float(counts[idx] / total_count), 4)
        palette.append({
            "hex": hex_code,
            "rgb": [int(r), int(g), int(b)],
            "ratio": ratio
        })

    return palette

def process_single_image(image_path, model, output_dir, conf_thresh=0.4):
    """
    Xử lý bóc tách phông nền cho 1 ảnh sản phẩm
    """
    import numpy as np
    from PIL import Image

    print(f"\n[*] Đang xử lý: {image_path.name}")
    
    # 1. Chạy inference YOLOv8-seg
    results = model.predict(source=str(image_path), conf=conf_thresh, verbose=False)
    result = results[0]

    if result.masks is None or len(result.masks) == 0:
        print(f"  [-] Không phát hiện thấy trang phục trong ảnh: {image_path.name}")
        return None

    # 2. Đọc ảnh gốc bằng PIL
    orig_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_img)
    h, w = orig_np.shape[:2]

    # 3. Lấy mask có diện tích lớn nhất hoặc độ tin cậy cao nhất
    boxes = result.boxes
    best_idx = 0
    best_conf = float(boxes.conf[0])
    best_cls_id = int(boxes.cls[0])

    for i in range(len(boxes)):
        if float(boxes.conf[i]) > best_conf:
            best_conf = float(boxes.conf[i])
            best_idx = i
            best_cls_id = int(boxes.cls[i])

    class_name = DEEPFASHION2_CLASSES.get(best_cls_id, f"class_{best_cls_id}")
    app_category = CATEGORY_MAPPING.get(class_name, "general")

    # Lấy mask nhị phân (kích thước theo ảnh gốc)
    mask_tensor = result.masks.data[best_idx].cpu().numpy()
    import cv2
    mask_resized = cv2.resize((mask_tensor * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (mask_resized > 127).astype(np.uint8)

    # 4. Làm mịn viền mask (Anti-aliasing / Gaussian blur viền) để viền áo không bị răng cưa
    smoothed_mask = cv2.GaussianBlur(binary_mask * 255, (5, 5), 0)

    # 5. Tạo ảnh RGBA trong suốt
    rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_img[:, :, :3] = orig_np
    rgba_img[:, :, 3] = smoothed_mask

    # 6. Trích xuất bảng màu chủ đạo
    color_palette = extract_dominant_colors(orig_np, binary_mask, num_colors=3)

    # 7. Lưu ảnh bóc tách nền
    out_stem = image_path.stem
    rgba_output_path = os.path.join(output_dir, f"{out_stem}_nobg.png")
    mask_output_path = os.path.join(output_dir, f"{out_stem}_mask.png")
    meta_output_path = os.path.join(output_dir, f"{out_stem}_meta.json")

    Image.fromarray(rgba_img).save(rgba_output_path, "PNG")
    Image.fromarray(binary_mask * 255).save(mask_output_path, "PNG")

    # 8. Lưu metadata thông số kỹ thuật (Dùng cho Tuần 5 chấm điểm)
    metadata = {
        "source_file": image_path.name,
        "deepfashion2_class": class_name,
        "yofaso_category": app_category,
        "confidence": round(best_conf, 4),
        "resolution": {"width": w, "height": h},
        "dominant_colors": color_palette,
        "outputs": {
            "transparent_rgba": os.path.basename(rgba_output_path),
            "binary_mask": os.path.basename(mask_output_path)
        }
    }

    with open(meta_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  [+] Nhận diện: {class_name} ({app_category}) - Độ tin cậy: {best_conf:.1%}")
    print(f"  [+] Bảng màu : {[c['hex'] + ' (' + str(int(c['ratio']*100)) + '%)' for c in color_palette]}")
    print(f"  [+] Đã lưu   : {rgba_output_path}")

    return metadata

def run_segmentation_batch(weights_path, input_dir, output_dir, conf_thresh=0.4):
    print("=" * 80)
    print(" CHẠY THỬ NGHIỆM BÓC TÁCH NỀN SẢN PHẨM VỚI DEEPFASHION2 (YOLOV8-SEG) ")
    print("=" * 80)
    print(f"[*] Trọng số mô hình : {weights_path}")
    print(f"[*] Thư mục ảnh mẫu  : {input_dir}")
    print(f"[*] Thư mục kết quả  : {output_dir}")
    print("-" * 80)

    if not os.path.exists(weights_path):
        print(f"[-] LỖI: Không tìm thấy file trọng số tại: {weights_path}", file=sys.stderr)
        return

    os.makedirs(output_dir, exist_ok=True)

    # Kiểm tra thư viện ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[-] CẢNH BÁO: Chưa cài đặt thư viện `ultralytics`.", file=sys.stderr)
        print("    -> Vui lòng chạy lệnh: pip install ultralytics opencv-python scikit-learn pillow")
        return

    # Nạp mô hình YOLOv8s-seg
    print("[*] Đang nạp mô hình YOLOv8s-seg...")
    model = YOLO(weights_path)
    print("[+] Nạp mô hình thành công!")

    # Quét tất cả các file ảnh trong thư mục input
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    input_path = Path(input_dir)
    image_files = [p for p in input_path.rglob("*") if p.suffix.lower() in image_extensions]

    if not image_files:
        print(f"[-] Không tìm thấy ảnh nào trong: {input_dir}")
        print("    -> Hãy copy 20-30 ảnh thời trang vào thư mục `data/test_samples/garments/`.")
        return

    print(f"[*] Tìm thấy {len(image_files)} ảnh mẫu cần xử lý.\n")
    processed_count = 0
    all_meta = []

    for img_file in image_files:
        meta = process_single_image(img_file, model, output_dir, conf_thresh)
        if meta:
            all_meta.append(meta)
            processed_count += 1

    summary_file = os.path.join(output_dir, "batch_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f" HOÀN TẤT THỬ NGHIỆM BÓC TÁCH NỀN ({processed_count}/{len(image_files)} ẢNH THÀNH CÔNG) ")
    print(f" File báo cáo tổng hợp: {summary_file}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFashion2 YOLOv8-seg Test Script")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file .pt")
    parser.add_argument("--input", type=str, default=None, help="Thư mục ảnh đầu vào")
    parser.add_argument("--output", type=str, default=None, help="Thư mục xuất kết quả")
    parser.add_argument("--conf", type=float, default=0.4, help="Ngưỡng confidence score (default 0.4)")
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    ml_service_dir = current_dir.parent
    project_root = ml_service_dir.parent

    weights = args.weights or str(ml_service_dir / "weights" / "deepfashion2_yolov8s-seg.pt")
    input_dir = args.input or str(project_root / "data" / "test_samples" / "garments")
    output_dir = args.output or str(project_root / "data" / "test_samples" / "results" / "segmented")

    run_segmentation_batch(weights, input_dir, output_dir, args.conf)
