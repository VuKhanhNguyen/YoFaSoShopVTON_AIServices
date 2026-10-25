"""
Script tải trọn bộ trọng số IDM-VTON Diffusion Model từ Hugging Face
Tự động bỏ qua các phần đã có (DensePose, HumanParsing, OpenPose, Assets)
"""
import os
import sys
from huggingface_hub import snapshot_download

def download_idm_vton():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    target_dir = os.path.join(base_dir, "api", "models", "idm-vton", "ckpt", "idm_vton_weights")
    
    print("=" * 70)
    print(" BẮT ĐẦU TẢI TRỌNG SỐ IDM-VTON DIFFUSION MODEL ")
    print(" Thư mục lưu:", target_dir)
    print("=" * 70)
    print("Đang kết nối Hugging Face để tải các module cốt lõi:")
    print("  - unet/ (~5.5 GB)")
    print("  - unet_encoder/ (~1.5 GB)")
    print("  - image_encoder/ (~1.2 GB)")
    print("  - vae/ (~335 MB)")
    print("  - text_encoder/ & text_encoder_2/ (~1.5 GB)")
    print("  - tokenizer/ & scheduler/ & model_index.json")
    print("-" * 70)
    print("Đang bỏ qua: densepose, humanparsing, openpose, assets (vì bạn đã có sẵn)...")
    print("Quá trình tải có thể mất từ 5 - 15 phút tùy thuộc tốc độ mạng của bạn.")
    print("=" * 70)

    try:
        snapshot_download(
            repo_id="yisol/IDM-VTON",
            local_dir=target_dir,
            ignore_patterns=[
                "densepose/*",
                "humanparsing/*",
                "openpose/*",
                "assets/*",
                "*.png",
                "*.jpg"
            ],
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("\n" + "=" * 70)
        print(" TẢI TRỌNG SỐ IDM-VTON HOÀN TẤT THÀNH CÔNG! ")
        print("=" * 70)
    except Exception as e:
        print(f"\n[LỖI]: {e}", file=sys.stderr)
        print("Gợi ý: Hãy kiểm tra kết nối mạng và thử chạy lại script này.")

if __name__ == "__main__":
    download_idm_vton()
