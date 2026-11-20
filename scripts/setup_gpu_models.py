"""Script thiết lập trọn gói mô hình local trên máy chủ GPU thuê (Vast.ai 24GB VRAM / 40GB Disk).

Đã loại bỏ hoàn toàn OOTDiffusion (VTON chuyển sang Cloud Provider FASHN v1.6 qua fal.ai).
Chỉ tải các mô hình local phục vụ validation & Admin CRUD:
1. Trọng số SAM 2.1 Large (sam2.1_l.pt ~428 MB) & VLM Qwen3-VL-4B-Instruct (BF16 ~8.5 GB)
2. Trọng số DeepFashion2 (YOLOv8s-seg ~23 MB) & YOLO26-Pose (~23 MB)
"""
from __future__ import annotations

import argparse
import os
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


def check_vton_key_status() -> None:
    """Kiểm tra nhanh API key tương ứng với VTON_PROVIDER trong .env (không in giá trị secret)."""
    env_file = ROOT / ".env"
    env_map: dict[str, str] = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, v = stripped.split("=", 1)
                env_map[k.strip()] = v.strip().strip('"').strip("'")

    provider = (os.getenv("VTON_PROVIDER") or env_map.get("VTON_PROVIDER") or "fashn_v16").strip()
    if provider == "genlook":
        key_name, key_url = "GENLOOK_API_KEY", "https://platform.genlook.app"
    else:
        key_name, key_url = "FAL_KEY", "https://fal.ai/dashboard/keys"

    val = (os.getenv(key_name) or env_map.get(key_name) or "").strip()
    if val and not val.startswith("CHANGE_ME"):
        print(f"  [+] {key_name} (VTON_PROVIDER={provider}): ĐÃ CẤU HÌNH")
    else:
        print(f"  [!] CẢNH BÁO: Chưa điền {key_name} thật trong .env cho VTON_PROVIDER={provider}!")
        print(f"      -> Lấy API key tại: {key_url}")
        print(f"      -> Điền vào .env:  {key_name}=<key_cua_ban>")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Master Setup: Tải mô hình local (SAM 2.1 Large, Qwen3-VL-4B-Instruct, DeepFashion2, YOLO-Pose) cho GPU server. Không tải OOTDiffusion."
    )
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra, không tải dữ liệu.")
    parser.add_argument("--skip-vlm-sam", action="store_true", help="Bỏ qua SAM 2.1 và Qwen3-VL-4B-Instruct.")
    parser.add_argument("--skip-vlm", action="store_true", help="Chỉ tải SAM 2.1 Large, bỏ qua Qwen3-VL-4B-Instruct.")
    args = parser.parse_args()

    print("=" * 76)
    print("YoFaSoShop AI Service · Setup Models (SAM 2.1 Large + Qwen3-VL-4B · No OOTD)")
    print("=" * 76)

    python_bin = sys.executable

    # 1. Tải SAM 2.1 Large & Qwen3-VL-4B-Instruct (phục vụ bóc tách nền & phân tích sản phẩm Admin CRUD)
    if not args.skip_vlm_sam:
        vlm_mode = "skip" if args.skip_vlm else "qwen"
        print(f"\n>>> [BƯỚC 1/2] Kiểm tra & tải SAM 2.1 Large và VLM ({vlm_mode})...")
        vlm_script = ROOT / "scripts" / "download_vlm_and_sam.py"
        cmd = [python_bin, str(vlm_script), "--sam-variant", "default", "--vlm-model", vlm_mode]
        if args.dry_run:
            cmd.append("--dry-run")
        run_subscript(cmd)

    # 2. Kiểm tra & tải DeepFashion2 + YOLO26-Pose
    print("\n>>> [BƯỚC 2/2] Kiểm tra DeepFashion2 & YOLO-Pose...")
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
        print("  [-] Chưa thấy yolo26s-pose.pt, đang tải từ Ultralytics release...")
        if not args.dry_run:
            import urllib.request

            fallback_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt"
            print(f"  [TẢI DỰ PHÒNG] Tải pose model từ {fallback_url}...")
            urllib.request.urlretrieve(fallback_url, str(pose_weight))
            print("  [+] Đã lưu yolo26s-pose.pt")

    print("\n>>> Kiểm tra cấu hình Cloud VTON...")
    check_vton_key_status()

    print("\n" + "=" * 76)
    print("HOÀN TẤT THIẾT LẬP MÔ HÌNH GPU SERVER!")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
