import os
import sys

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch

def extract_state_dict(checkpoint_path: str, output_path: str):
    if not os.path.exists(checkpoint_path):
        print(f"[MISSING] File khong ton tai: {checkpoint_path}")
        return

    print(f"\n[INFO] Dang nap checkpoint: {checkpoint_path}")
    raw_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"   Dung luong ban dau: {raw_size_mb:.2f} MB")

    try:
        # Load weights on CPU
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Trich xuat state_dict
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                print("   -> Tim thay key 'state_dict' (chuan MMCV / MMFashion)")
                clean_state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                print("   -> Tim thay key 'model'")
                clean_state_dict = checkpoint["model"]
            elif "model_state_dict" in checkpoint:
                print("   -> Tim thay key 'model_state_dict'")
                clean_state_dict = checkpoint["model_state_dict"]
            else:
                first_val = next(iter(checkpoint.values())) if checkpoint else None
                if isinstance(first_val, torch.Tensor):
                    print("   -> Toan bo dict la state_dict tensor")
                    clean_state_dict = checkpoint
                else:
                    print("   -> Dict chua cac thong tin metadata, loc lay tensor...")
                    clean_state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        else:
            print("   -> Checkpoint la doi tuong model truc tiep, lay state_dict()...")
            clean_state_dict = checkpoint.state_dict()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Luu lai ban gon nhe (FP32 Clean)
        torch.save(clean_state_dict, output_path)
        clean_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[SUCCESS] Da toi uu xong (FP32 Clean): {output_path}")
        print(f"   Dung luong moi: {clean_size_mb:.2f} MB (Giam {(1 - clean_size_mb/raw_size_mb)*100:.1f}%)")

        # Xuất thêm bản FP16 Half-Precision siêu nhẹ (911 MB)
        fp16_output_path = output_path.replace("_clean.pth", "_fp16.pth")
        fp16_state_dict = {
            k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in clean_state_dict.items()
        }
        torch.save(fp16_state_dict, fp16_output_path)
        fp16_size_mb = os.path.getsize(fp16_output_path) / (1024 * 1024)
        print(f"[SUCCESS] Da xuat ban FP16: {fp16_output_path} ({fp16_size_mb:.2f} MB)")

    except Exception as e:
        print(f"[ERROR] Loi khi xu ly {checkpoint_path}: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Kiểm tra cả api/weights và YoFaSoShop_AIService/weights
    weights_dir = os.path.join(script_dir, "weights")
    if not os.path.exists(weights_dir):
        parent_weights = os.path.join(os.path.dirname(script_dir), "weights")
        if os.path.exists(parent_weights):
            weights_dir = parent_weights
    
    # Tìm kiếm các file .pth trong weights/pth/ hoặc weights/
    candidates = [
        os.path.join(weights_dir, "pth", "model_best.pth"),
        os.path.join(weights_dir, "pth", "latest.pth"),
        os.path.join(weights_dir, "model_best.pth"),
        os.path.join(weights_dir, "latest.pth"),
    ]

    processed_any = False
    for p in candidates:
        if os.path.exists(p):
            base_name = os.path.splitext(os.path.basename(p))[0]
            out_p = os.path.join(weights_dir, f"resnet50_attr_{base_name}_clean.pth")
            extract_state_dict(p, out_p)
            processed_any = True

    if not processed_any:
        print(f"[SCAN] Dang tim file .pth trong {weights_dir}...")
        for root, dirs, files in os.walk(weights_dir):
            for file in files:
                if file.endswith(".pth") and not file.endswith("_clean.pth") and not file.endswith("_fp16.pth"):
                    full_p = os.path.join(root, file)
                    base_name = os.path.splitext(file)[0]
                    out_p = os.path.join(weights_dir, f"resnet50_attr_{base_name}_clean.pth")
                    extract_state_dict(full_p, out_p)
                    processed_any = True

    if not processed_any:
        print("[WARNING] Khong tim thay file .pth nao de toi uu!")
