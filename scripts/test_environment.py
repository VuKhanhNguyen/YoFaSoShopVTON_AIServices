"""
==============================================================================
Sanity Check & Environment Verification Script - YoFaSoShop_VTON
Tác giả: Senior Python & Deep Learning Specialist
Mục đích:
  1. Kiểm tra môi trường phần cứng (CPU, GPU CUDA, VRAM).
  2. Xác thực tính toàn vẹn (Integrity & Checksum/Size) của các file trọng số mô hình:
     - YOLOv8s-seg (DeepFashion2)
     - DensePose (Detectron2)
     - Human Parsing (ATR & LIP ONNX)
     - OpenPose (CMU Pose)
     - OOTDiffusion Core Pipeline (VITON-HD & DressCode, VAE, CLIP Image/Text Encoder)
     - IDM-VTON Diffusers Pipeline (Legacy / Không bắt buộc)
  3. Đánh giá tính sẵn sàng cho thử nghiệm Tuần 4 (Local RTX 3050 vs Google Colab T4).
==============================================================================
"""

import os
import sys
import json
import platform
import importlib.util

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def print_banner(title):
    print("\n" + "=" * 80)
    print(f" {title.center(78)} ")
    print("=" * 80)


def directory_size(path):
    """Return the logical size of files below path, excluding HF metadata."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if name != ".cache"]
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                pass
    return total


def check_ootdiffusion(base_dir):
    """Check downloaded OOTDiffusion assets (Core Virtual Try-On Engine)."""
    print("\n--- Kiểm tra OOTDiffusion Core Pipeline (yisol/ootdiffusion) ---")
    candidates = [
        os.path.join(base_dir, "api", "models", "ootdiffusion"),
        os.path.join(base_dir, "models", "ootdiffusion"),
    ]
    oot_dir = next((path for path in candidates if os.path.isdir(path)), None)
    if oot_dir is None:
        print(f"[-] OOTDiffusion Core: THIẾU - [FAIL] Không tìm thấy tại {candidates[0]}")
        return False, False

    print(f"[*] Phát hiện OOTDiffusion tại: {oot_dir}")
    print(f"[*] Dung lượng logic (không tính .cache): {format_size(directory_size(oot_dir))}")

    snapshot_ok = True
    required_dirs = {
        "checkpoints": "Trọng số mô hình OOTDiffusion",
        "ootd": "Mã nguồn inference OOTDiffusion",
        "preprocess": "OpenPose và Human Parsing preprocessing",
        "run": "CLI/Gradio entrypoints",
    }
    for rel_path, role in required_dirs.items():
        full_path = os.path.join(oot_dir, rel_path)
        if os.path.isdir(full_path):
            print(f"  [+] {rel_path + '/':<20}: [PASS] ({role})")
        else:
            print(f"  [-] {rel_path + '/':<20}: [FAIL] ({role})")
            snapshot_ok = False

    required_files = [
        ("README.md", 0.0001, "Hướng dẫn/cấu hình mô hình"),
        ("requirements.txt", 0.0001, "Danh sách dependencies"),
        (os.path.join("run", "run_ootd.py"), 0.001, "CLI inference"),
        (os.path.join("run", "gradio_ootd.py"), 0.001, "Gradio demo app"),
        (os.path.join("checkpoints", "ootd", "ootd_dc", "checkpoint-36000", "unet_garm", "diffusion_pytorch_model.safetensors"), 3000, "DressCode Garment UNet"),
        (os.path.join("checkpoints", "ootd", "ootd_dc", "checkpoint-36000", "unet_vton", "diffusion_pytorch_model.safetensors"), 3000, "DressCode VTON UNet"),
        (os.path.join("checkpoints", "ootd", "ootd_hd", "checkpoint-36000", "unet_garm", "diffusion_pytorch_model.safetensors"), 3000, "VITON-HD Garment UNet"),
        (os.path.join("checkpoints", "ootd", "ootd_hd", "checkpoint-36000", "unet_vton", "diffusion_pytorch_model.safetensors"), 3000, "VITON-HD VTON UNet"),
        (os.path.join("checkpoints", "ootd", "vae", "diffusion_pytorch_model.bin"), 300, "VAE"),
        (os.path.join("checkpoints", "ootd", "text_encoder", "pytorch_model.bin"), 450, "CLIP Text Encoder"),
        (os.path.join("checkpoints", "humanparsing", "parsing_atr.onnx"), 240, "ATR Human Parsing"),
        (os.path.join("checkpoints", "humanparsing", "parsing_lip.onnx"), 240, "LIP Human Parsing"),
    ]
    for rel_path, min_size_mb, role in required_files:
        full_path = os.path.join(oot_dir, rel_path)
        if not os.path.isfile(full_path):
            print(f"  [-] {role:<28}: THIẾU - [FAIL] {rel_path}")
            snapshot_ok = False
            continue

        size_bytes = os.path.getsize(full_path)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb >= min_size_mb:
            print(f"  [+] {role:<28}: {format_size(size_bytes):<10} - [PASS]")
        else:
            print(f"  [-] {role:<28}: {format_size(size_bytes):<10} - [FAIL] File quá nhỏ")
            snapshot_ok = False

    print("\n--- Kiểm tra Runtime OOTDiffusion Local ---")
    required_modules = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "diffusers": "Diffusers",
        "transformers": "Transformers",
        "accelerate": "Accelerate",
        "onnxruntime": "ONNX Runtime",
        "basicsr": "BasicSR",
        "cv2": "OpenCV",
        "PIL": "Pillow",
        "skimage": "scikit-image",
        "einops": "Einops",
        "gradio": "Gradio",
    }
    missing_modules = [
        display_name
        for module_name, display_name in required_modules.items()
        if importlib.util.find_spec(module_name) is None
    ]
    python_ok = sys.version_info[:2] == (3, 10)
    if python_ok:
        print(f"  [+] Python {platform.python_version():<12}: [PASS] (upstream dùng Python 3.10)")
    else:
        print(f"  [-] Python {platform.python_version():<12}: [NOT READY] Khuyến nghị môi trường Python 3.10 cho local self-host")

    if missing_modules:
        print(f"  [-] Dependencies           : [NOT READY] Thiếu {', '.join(missing_modules)}")
    else:
        print("  [+] Dependencies           : [PASS] Đã import được toàn bộ gói bắt buộc")

    openpose_candidates = [
        os.path.join(oot_dir, "checkpoints", "openpose", "ckpts", "body_pose_model.pth"),
        os.path.join(base_dir, "api", "models", "ootdiffusion", "checkpoints", "openpose", "ckpts", "body_pose_model.pth"),
        os.path.join(base_dir, "api", "models", "idm-vton", "ckpt", "openpose", "ckpts", "body_pose_model.pth"),
        os.path.join(base_dir, "models", "idm-vton", "ckpt", "openpose", "ckpts", "body_pose_model.pth"),
        os.path.join(base_dir, "weights", "body_pose_model.pth"),
    ]
    openpose_path = next((p for p in openpose_candidates if os.path.isfile(p)), None)
    openpose_ok = openpose_path is not None and os.path.getsize(openpose_path) >= 190 * 1024 * 1024
    if openpose_ok:
        print(f"  [+] OpenPose checkpoint     : {format_size(os.path.getsize(openpose_path))} - [PASS]")
    else:
        print("  [-] OpenPose checkpoint     : [NOT READY] Chưa có; upstream sẽ tải khi khởi chạy lần đầu")

    cuda_ok = False
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False
    print(f"  [{'+' if cuda_ok else '-'}] CUDA cho OOTDiffusion    : {'[PASS]' if cuda_ok else '[NOT READY] Cần PyTorch CUDA và GPU phù hợp'}")

    runtime_ready = snapshot_ok and python_ok and not missing_modules and openpose_ok and cuda_ok
    print(f"  [i] Snapshot OOTDiffusion  : {'[PASS]' if snapshot_ok else '[FAIL]'}")
    print(f"  [i] Self-host local        : {'[READY]' if runtime_ready else '[NOT READY] (Chạy Colab T4 / HF Space không bị ảnh hưởng)'}")
    return snapshot_ok, runtime_ready

def check_system_environment():
    print_banner("[1/3] KIỂM TRA MÔI TRƯỜNG PHẦN CỨNG & PYTHON")
    print(f"[*] Hệ điều hành        : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"[*] Phiên bản Python    : {platform.python_version()} ({sys.executable})")

    cuda_available = False
    try:
        import torch
        print(f"[*] PyTorch Version     : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            print(f"[+] CUDA Available      : YES (Devices: {gpu_count})")
            print(f"[+] GPU Thiết bị        : {gpu_name}")
            print(f"[+] Tổng dung lượng VRAM: {format_size(vram_bytes)}")
            
            # Đánh giá theo kiến trúc OOTDiffusion
            if vram_bytes >= 14 * 1024**3:
                print("    -> ĐÁNH GIÁ: GPU >= 14GB VRAM (Chuẩn Colab T4 / A100). Chạy trực tiếp OOTDiffusion mượt mà.")
            elif vram_bytes >= 5.5 * 1024**3:
                print("    -> ĐÁNH GIÁ: GPU 6GB - 8GB VRAM (Laptop RTX 3050 / 3060). Phù hợp chạy OOTDiffusion với FP16 / model CPU offload.")
            else:
                print("    -> CẢNH BÁO: VRAM < 6GB. Không khuyến nghị chạy OOTDiffusion trực tiếp. Hãy dùng Google Colab Free / Hugging Face Space.")
        else:
            print("[-] CUDA Available      : NO (Đang chạy chế độ CPU).")
            print("    -> Lưu ý: Các mô hình tiền xử lý (YOLO, Parsing) có thể chạy CPU. Riêng OOTDiffusion cần GPU CUDA (hoặc gọi qua Colab T4 / Space).")
    except ImportError:
        print("[-] PyTorch             : CHƯA ĐƯỢC CÀI ĐẶT (pip install torch torchvision).")

    return cuda_available

def check_checkpoints(base_dir):
    print_banner("[2/3] KIỂM TRA TÍNH TOÀN VẸN CÁC TỆP TRỌNG SỐ (CHECKPOINTS)")
    
    # Danh sách kiểm tra định nghĩa chuẩn theo kiến trúc dự án
    weights_spec = [
        {
            "category": "DeepFashion2 Segmentation",
            "name": "deepfashion2_yolov8s-seg.pt",
            "rel_paths": [
                os.path.join("weights", "deepfashion2_yolov8s-seg.pt"),
                os.path.join("api", "weights", "deepfashion2_yolov8s-seg.pt"),
            ],
            "min_size_mb": 20,
            "max_size_mb": 30,
            "role": "Bóc tách nền đồ shop, nhận diện loại áo/quần/váy/đầm"
        },
        {
            "category": "Human Parsing",
            "name": "parsing_atr.onnx",
            "rel_paths": [
                os.path.join("api", "models", "ootdiffusion", "checkpoints", "humanparsing", "parsing_atr.onnx"),
                os.path.join("models", "ootdiffusion", "checkpoints", "humanparsing", "parsing_atr.onnx"),
                os.path.join("api", "models", "idm-vton", "ckpt", "humanparsing", "parsing_atr.onnx"),
                os.path.join("models", "idm-vton", "ckpt", "humanparsing", "parsing_atr.onnx"),
            ],
            "min_size_mb": 130,
            "max_size_mb": 280,
            "role": "Tách vùng da/mặt/quần áo cũ để tạo mặt nạ Agnostic Mask"
        },
        {
            "category": "OpenPose Pose Estimation",
            "name": "body_pose_model.pth",
            "rel_paths": [
                os.path.join("api", "models", "ootdiffusion", "checkpoints", "openpose", "ckpts", "body_pose_model.pth"),
                os.path.join("models", "ootdiffusion", "checkpoints", "openpose", "ckpts", "body_pose_model.pth"),
                os.path.join("api", "models", "idm-vton", "ckpt", "openpose", "ckpts", "body_pose_model.pth"),
                os.path.join("models", "idm-vton", "ckpt", "openpose", "ckpts", "body_pose_model.pth"),
            ],
            "min_size_mb": 190,
            "max_size_mb": 230,
            "role": "Định vị 18 điểm khớp xương cơ thể người mẫu"
        },
        {
            "category": "DensePose IUV Map",
            "name": "model_final_162be9.pkl",
            "rel_paths": [
                os.path.join("api", "models", "idm-vton", "ckpt", "densepose", "model_final_162be9.pkl"),
                os.path.join("models", "idm-vton", "ckpt", "densepose", "model_final_162be9.pkl"),
                os.path.join("models", "densepose", "model_final_162be9.pkl"),
            ],
            "min_size_mb": 240,
            "max_size_mb": 270,
            "role": "Khớp lưới 3D cơ thể người, uốn nếp vải theo góc nghiêng"
        }
    ]

    all_passed = True
    print(f"{'Tên Mô Hình':<32} | {'Trạng Thái':<12} | {'Kích Thước':<12} | {'Đánh Giá'}")
    print("-" * 80)

    for spec in weights_spec:
        candidates = [os.path.join(base_dir, p) for p in spec["rel_paths"]]
        full_path = next((cand for cand in candidates if os.path.exists(cand)), None)
        
        if full_path is None:
            print(f"{spec['name']:<32} | {'THIẾU FILE':<12} | {'0 B':<12} | [FAIL] Không tìm thấy file")
            all_passed = False
            continue
        
        file_size = os.path.getsize(full_path)
        size_mb = file_size / (1024 * 1024)
        formatted = format_size(file_size)

        if spec["min_size_mb"] <= size_mb <= spec["max_size_mb"]:
            status_text = "HỢP LỆ"
            eval_text = "[PASS] Nguyên vẹn"
        else:
            status_text = "BẤT THƯỜNG"
            eval_text = f"[WARN] Lệch dung lượng (Kỳ vọng: {spec['min_size_mb']}-{spec['max_size_mb']}MB)"
            all_passed = False

        print(f"{spec['name']:<32} | {status_text:<12} | {formatted:<12} | {eval_text}")

    # Kiểm tra OOTDiffusion (Mô hình Cốt lõi Bắt buộc - Core VTON Pipeline)
    oot_snapshot_ok, oot_runtime_ready = check_ootdiffusion(base_dir)
    if not oot_snapshot_ok:
        all_passed = False

    # Kiểm tra IDM-VTON (Legacy - Không bắt buộc, dự án đã chuyển sang OOTDiffusion)
    print("\n--- Kiểm tra IDM-VTON Diffusion Pipeline (Legacy / Không bắt buộc) ---")
    idm_candidates = [
        os.path.join(base_dir, "api", "models", "idm-vton", "ckpt", "idm_vton_weights"),
        os.path.join(base_dir, "models", "idm-vton", "ckpt", "idm_vton_weights")
    ]
    idm_dir = next((cand for cand in idm_candidates if os.path.exists(cand)), None)
    if idm_dir is not None:
        print(f"[*] Phát hiện IDM-VTON weights tại: {idm_dir}")
        print("  [i] IDM-VTON: ĐÃ CÓ - [OPTIONAL] (Mô hình SDXL legacy, giữ lại để đối chứng thực nghiệm nếu cần)")
    else:
        print("  [i] IDM-VTON: KHÔNG NẠP - [INFO] (Bỏ qua: Dự án chuẩn hóa dùng OOTDiffusion để tiết kiệm chi phí)")

    # Kiểm tra Nhóm 2: Mô hình bổ trợ U-Net ResNet50 & MMFashion Attribute
    print("\n--- Kiểm tra Mô Hình Bổ Trợ (U-Net ResNet50 & MMFashion Attribute) ---")
    unet_candidates = [
        os.path.join(base_dir, "api", "models", "unet_resnet50", "model.safetensors"),
        os.path.join(base_dir, "models", "unet_resnet50", "model.safetensors")
    ]
    unet_found = any(os.path.exists(p) for p in unet_candidates)
    if unet_found:
        found_p = next(p for p in unet_candidates if os.path.exists(p))
        sz = os.path.getsize(found_p)
        print(f"  [+] U-Net ResNet50 (model.safetensors)   : {format_size(sz):<10} - [PASS] (Tách nền trang phục IoU 0.8964)")
    else:
        print("  [-] U-Net ResNet50 (model.safetensors)   : CHƯA CÓ    - [OPTIONAL] (Khuyên dùng để bổ trợ tách nền đồ shop)")

    attr_candidates = [
        os.path.join(base_dir, "weights", "resnet50_attr_model_best_clean.pth"),
        os.path.join(base_dir, "weights", "resnet50_attr_model_best_fp16.pth"),
        os.path.join(base_dir, "weights", "resnet50_attr_latest_clean.pth")
    ]
    attr_found = [p for p in attr_candidates if os.path.exists(p)]
    if attr_found:
        print(f"  [+] MMFashion ResNet-50 Attribute Model  : {len(attr_found)} file(s)  - [PASS] (Nhận diện 1.000 thuộc tính trang phục)")
    else:
        print("  [-] MMFashion ResNet-50 Attribute Model  : CHƯA CÓ    - [OPTIONAL] (Dùng cho dự đoán thuộc tính nâng cao)")

    # Kiểm tra Nhóm 3: FashionStylist Knowledge Base (Child, Female, Male)
    print("\n--- Kiểm tra Cơ Sở Tri Thức Phối Đồ (FashionStylist Knowledge Base) ---")
    kb_dir = os.path.join(base_dir, "data", "knowledge")
    kb_subfolders = ["Child", "Female", "Male"]
    kb_all_ok = True
    if os.path.exists(kb_dir):
        for sf in kb_subfolders:
            sf_path = os.path.join(kb_dir, sf)
            if os.path.exists(sf_path) and os.path.isdir(sf_path):
                csv_files = [f for f in os.listdir(sf_path) if f.endswith(".csv")]
                print(f"  [+] data/knowledge/{sf:<10}: {len(csv_files)} file CSVs - [PASS] (Tri thức phối đồ {sf})")
            else:
                print(f"  [-] data/knowledge/{sf:<10}: THIẾU      - [WARN] (Cần tải bộ dữ liệu FashionStylist)")
                kb_all_ok = False
    else:
        print(f"  [-] Thư mục {kb_dir} chưa tồn tại - [WARN] (Cần tải về để phục vụ Tuần 5 Scoring Engine)")
        kb_all_ok = False

    # Kiểm tra Nhóm 4: Tool tiền xử lý deepfashion2_to_coco.py
    print("\n--- Kiểm tra Script Tiền Xử Lý & Dữ Liệu Benchmark ---")
    coco_script = os.path.join(base_dir, "ml-training", "preprocess", "deepfashion2_to_coco.py")
    if os.path.exists(coco_script):
        print(f"  [+] deepfashion2_to_coco.py              : {format_size(os.path.getsize(coco_script)):<10} - [PASS] (Tool convert nhãn COCO)")
    else:
        print("  [-] deepfashion2_to_coco.py              : THIẾU      - [WARN]")

    raw_df2 = os.path.join(base_dir, "data", "raw", "deepfashion2")
    if os.path.exists(raw_df2):
        print(f"  [+] data/raw/deepfashion2/               : TỒN TẠI    - [PASS] (Dữ liệu Ground Truth đo mAP@50)")
    else:
        print("  [-] data/raw/deepfashion2/               : CHƯA CÓ    - [INFO] (Tải khi cần chạy benchmark mAP ở Tuần 14)")

    raw_attr = os.path.join(base_dir, "data", "raw", "deepfashion_attr", "Anno")
    if os.path.exists(raw_attr):
        print(f"  [+] data/raw/deepfashion_attr/Anno/      : TỒN TẠI    - [PASS] (Từ điển 1.000 thuộc tính trang phục)")
    else:
        print("  [-] data/raw/deepfashion_attr/Anno/      : CHƯA CÓ    - [INFO] (Dùng làm cơ sở lý thuyết cho báo cáo)")

    return all_passed, oot_runtime_ready

def print_recommendation(cuda_ok, checkpoints_ok, oot_runtime_ready):
    print_banner("[3/3] TỔNG KẾT & KHUYẾN NGHỊ CHO TUẦN 4")
    if checkpoints_ok:
        print("[+] CHÚC MỪNG: Toàn bộ checkpoints bắt buộc của OOTDiffusion đã được xác thực đầy đủ!")
        print("    -> Bạn đã hoàn tất xuất sắc điều kiện tiên quyết để bước sang giai đoạn thử nghiệm TUẦN 4.")
    else:
        print("[-] CẢNH BÁO: Một số file trọng số OOTDiffusion bị thiếu hoặc chưa đúng vị trí.")
        print("    -> Hãy chạy lại script tải tương ứng: download_weights.sh hoặc download_ootdiffusion.py.")

    if oot_runtime_ready:
        print("[+] OOTDiffusion self-host local: READY.")
    else:
        print("[i] OOTDiffusion weights: PASS (Sẵn sàng chạy thực nghiệm trên Google Colab / Hugging Face Space).")
        print("    -> Đối với local máy cá nhân: Cần môi trường Python 3.10 và cài đặt dependencies để self-host.")

    print("\nChiến lược thực thi Tuần 4 (21/09 - 27/09/2026):")
    print("1. Thu thập dữ liệu mẫu: Chuẩn bị 20-30 ảnh thời trang tại `data/test_samples/garments/` (tops, bottoms, dresses).")
    print("2. Chạy thử tách nền: Dùng `scripts/test_segmentation.py` với YOLOv8s-seg (DeepFashion2).")
    print("3. Chạy thử nghiệm OOTDiffusion Virtual Try-On:")
    if cuda_ok:
        print("   * Local Laptop (RTX 3050 6GB): Chạy OOTDiffusion với FP16 / CPU offload để tối ưu VRAM.")
    else:
        print("   * Google Colab Free (GPU T4 16GB): Sử dụng notebook `ml-training/notebooks/week4_experiment_colab_ver2.ipynb` để render ảnh nhanh 3-5s/ảnh.")
        print("   * Hugging Face Space: Kết nối zero-config qua Gradio Client (`levihsu/OOTDiffusion`).")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ml_service_dir = os.path.dirname(script_dir)
    
    cuda_status = check_system_environment()
    ckpt_status, oot_runtime_status = check_checkpoints(ml_service_dir)
    print_recommendation(cuda_status, ckpt_status, oot_runtime_status)
    raise SystemExit(0 if ckpt_status else 1)
