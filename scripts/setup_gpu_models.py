"""Script thiết lập trọn gói tất cả mô hình trên máy chủ GPU thuê (Vast.ai, RunPod, v.v.).

Chạy một lệnh duy nhất để tải toàn bộ:
1. Checkpoint OOTDiffusion (VITON-HD & DressCode)
2. Trọng số SAM 2 (sam2_b.pt + sam2_t.pt)
3. Mô hình VLM Qwen2-VL-2B-Instruct
4. Trọng số DeepFashion2 & YOLO-Pose (nếu chưa có từ Git LFS)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_subscript(cmd: list[str]) -> None:
    print(f"\n[EXEC] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[LỖI] Lệnh thất bại với mã lỗi {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Master Setup: Tải trọn gói tất cả mô hình cần thiết cho YoFaSoShop AI Service trên GPU."
    )
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra, không tải dữ liệu.")
    parser.add_argument("--skip-ootd", action="store_true", help="Bỏ qua OOTDiffusion.")
    parser.add_argument("--skip-vlm-sam", action="store_true", help="Bỏ qua VLM và SAM 2.")
    args = parser.parse_args()

    print("=" * 76)
    print("YoFaSoShop AI Service · Master Setup Models & Weights")
    print("=" * 76)

    python_bin = sys.executable

    # 1. Tải OOTDiffusion
    if not args.skip_ootd:
        print("\n>>> [BƯỚC 1/3] Kiểm tra & tải OOTDiffusion Checkpoints...")
        ootd_script = ROOT / "scripts" / "download_ootdiffusion.py"
        cmd = [python_bin, str(ootd_script)]
        if args.dry_run:
            cmd.append("--dry-run")
        run_subscript(cmd)

    # 2. Tải SAM 2 & VLM
    if not args.skip_vlm_sam:
        print("\n>>> [BƯỚC 2/3] Kiểm tra & tải SAM 2 và VLM (Qwen2-VL-2B)...")
        vlm_script = ROOT / "scripts" / "download_vlm_and_sam.py"
        cmd = [python_bin, str(vlm_script), "--sam-variant", "default", "--vlm-model", "qwen"]
        if args.dry_run:
            cmd.append("--dry-run")
        run_subscript(cmd)

    # 3. Kiểm tra DeepFashion2 & YOLO-Pose
    print("\n>>> [BƯỚC 3/3] Kiểm tra DeepFashion2 & YOLO-Pose...")
    weights_dir = ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    df_weight = weights_dir / "deepfashion2_yolov8s-seg.pt"
    pose_weight = weights_dir / "yolo26s-pose.pt"

    if df_weight.is_file() and df_weight.stat().st_size > 1024**2:
        print(f"  [+] DeepFashion2 weight: ĐÃ CÓ ({df_weight.stat().st_size / 1024**2:.1f} MB)")
    else:
        print("  [-] Thiếu deepfashion2_yolov8s-seg.pt, đang tải từ Hugging Face Bingsu/adetailer...")
        if not args.dry_run:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="Bingsu/adetailer",
                filename="deepfashion2_yolov8s-seg.pt",
                local_dir=str(weights_dir),
            )
            print("  [+] Đã tải xong deepfashion2_yolov8s-seg.pt")

    if pose_weight.is_file() and pose_weight.stat().st_size > 1024**2:
        print(f"  [+] YOLO-Pose weight: ĐÃ CÓ ({pose_weight.stat().st_size / 1024**2:.1f} MB)")
    else:
        print("  [-] Chưa thấy yolo26s-pose.pt. Hãy đảm bảo bạn đã chạy 'git lfs pull' trên repo.")
        # Tải fallback yolo26s-pose nếu cần
        if not args.dry_run:
            import urllib.request
            fallback_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt"
            print(f"  [TẢI DỰ PHÒNG] Tải standard pose model từ {fallback_url}...")
            urllib.request.urlretrieve(fallback_url, str(pose_weight))
            print("  [+] Đã lưu fallback yolo26s-pose.pt")

    print("\n" + "=" * 76)
    print("HOÀN TẤT THIẾT LẬP TOÀN BỘ MÔ HÌNH TRÊN GPU SERVER!")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())

