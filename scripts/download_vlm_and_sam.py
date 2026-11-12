"""Download official checkpoints for SAM 2 and VLM (Qwen2-VL / Florence-2).

Following GPU.md and VLM_and_SAM.md architecture specifications.
- SAM 2 weights are placed in `weights/sam2/` or `weights/`
- VLM model repositories are placed in `api/models/vlm/`
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_FREE_GIB = 12

# Official Checkpoint Sources
SAM_URLS = {
    # Ultralytics optimized single-file weights (Standard & recommended for inference)
    "sam2_b": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt",
    "sam2_t": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_t.pt",
    "sam2.1_b": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt",
    "sam2.1_t": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt",
    # Meta official research checkpoints
    "meta_sam2.1_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "meta_sam2.1_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
}

VLM_REPOS = {
    "qwen": {
        "repo_id": "Qwen/Qwen2-VL-2B-Instruct",
        "description": "Alibaba Qwen2-VL-2B-Instruct (Vision-Language model ~4.5 GB)",
        "folder": "Qwen2-VL-2B-Instruct",
    },
    "florence": {
        "repo_id": "microsoft/Florence-2-base",
        "description": "Microsoft Florence-2-base (Vision-Language model ~0.9 GB)",
        "folder": "Florence-2-base",
    },
}


def configure_console() -> None:
    """Ensure utf-8 console output for Windows/Linux."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải tự động checkpoint chính thức cho SAM 2 và VLM (Qwen2-VL / Florence-2)."
    )
    parser.add_argument(
        "--sam-variant",
        choices=["default", "tiny-only", "base-only", "all-ultralytics", "meta-official", "skip"],
        default="default",
        help="Lựa chọn biến thể SAM 2 cần tải (default: sam2_b + sam2_t).",
    )
    parser.add_argument(
        "--vlm-model",
        choices=["qwen", "florence", "both", "skip"],
        default="qwen",
        help="Lựa chọn VLM cần tải (default: Qwen2-VL-2B-Instruct).",
    )
    parser.add_argument(
        "--sam-dir",
        type=Path,
        default=ROOT / "weights" / "sam2",
        help="Thư mục đích lưu trọng số SAM 2 (mặc định: weights/sam2/).",
    )
    parser.add_argument(
        "--vlm-dir",
        type=Path,
        default=ROOT / "api" / "models" / "vlm",
        help="Thư mục đích lưu mô hình VLM (mặc định: api/models/vlm/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in kế hoạch tải và kiểm tra dung lượng ổ đĩa, không tải file.",
    )
    parser.add_argument(
        "--force-low-disk",
        action="store_true",
        help=f"Bỏ qua cảnh báo dung lượng đĩa trống dưới {MIN_FREE_GIB} GiB.",
    )
    return parser.parse_args()


def download_file_with_progress(url: str, dest_path: Path) -> None:
    """Download a file with streaming progress display."""
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        print(f"  [ĐÃ CÓ] {dest_path.name} ({dest_path.stat().st_size / 1024**2:.1f} MB), bỏ qua.")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    print(f"  [ĐANG TẢI] {url} -> {dest_path.name}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (YoFaSoShop-VTON Downloader)"},
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
            total_size = int(response.info().get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunk

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    mb_down = downloaded / 1024**2
                    mb_total = total_size / 1024**2
                    sys.stdout.write(f"\r    Tiến trình: {percent:5.1f}% ({mb_down:6.1f} / {mb_total:6.1f} MB)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r    Đã tải: {downloaded / 1024**2:6.1f} MB")
                    sys.stdout.flush()
            sys.stdout.write("\n")

        temp_path.replace(dest_path)
        print(f"  [HOÀN THÀNH] Đã lưu vào {dest_path}")
    except Exception as exc:
        if temp_path.is_file():
            temp_path.unlink()
        raise RuntimeError(f"Lỗi khi tải {url}: {exc}") from exc


def download_hf_repo(repo_id: str, local_dir: Path, hf_token: str | None = None) -> None:
    """Download a full model snapshot from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("Cần cài huggingface-hub: pip install huggingface-hub>=0.25.0")

    print(f"  [HUGGINGFACE] Bắt đầu tải snapshot {repo_id}...")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=hf_token,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "flax_model*"],
    )
    print(f"  [HOÀN THÀNH] Snapshot {repo_id} đã lưu tại {local_dir}")


def main() -> int:
    configure_console()
    args = parse_args()

    sam_dir = args.sam_dir.expanduser().resolve()
    vlm_dir = args.vlm_dir.expanduser().resolve()

    print("=" * 76)
    print("YoFaSoShop_AIService · Trình tải tự động mô hình SAM 2 và VLM")
    print(f"SAM 2 Target: {sam_dir}")
    print(f"VLM Target  : {vlm_dir}")
    print("=" * 76)

    # 1. Kiểm tra dung lượng đĩa
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    print(f"Dung lượng đĩa trống khả dụng: {free_gib:.1f} GiB (Yêu cầu tối thiểu: {MIN_FREE_GIB} GiB)")
    if free_gib < MIN_FREE_GIB and not args.force_low_disk and not args.dry_run:
        print(
            f"[LỖI] Ổ đĩa chỉ còn {free_gib:.1f} GiB trống. Thêm --force-low-disk để bỏ qua.",
            file=sys.stderr,
        )
        return 2

    # 2. Xác định danh sách file SAM 2 cần tải
    sam_targets: list[tuple[str, Path]] = []
    if args.sam_variant == "default":
        sam_targets = [
            (SAM_URLS["sam2_b"], sam_dir / "sam2_b.pt"),
            (SAM_URLS["sam2_t"], sam_dir / "sam2_t.pt"),
        ]
    elif args.sam_variant == "tiny-only":
        sam_targets = [(SAM_URLS["sam2_t"], sam_dir / "sam2_t.pt")]
    elif args.sam_variant == "base-only":
        sam_targets = [(SAM_URLS["sam2_b"], sam_dir / "sam2_b.pt")]
    elif args.sam_variant == "all-ultralytics":
        sam_targets = [
            (SAM_URLS["sam2_b"], sam_dir / "sam2_b.pt"),
            (SAM_URLS["sam2_t"], sam_dir / "sam2_t.pt"),
            (SAM_URLS["sam2.1_b"], sam_dir / "sam2.1_b.pt"),
            (SAM_URLS["sam2.1_t"], sam_dir / "sam2.1_t.pt"),
        ]
    elif args.sam_variant == "meta-official":
        sam_targets = [
            (SAM_URLS["meta_sam2.1_base_plus"], sam_dir / "sam2.1_hiera_base_plus.pt"),
            (SAM_URLS["meta_sam2.1_tiny"], sam_dir / "sam2.1_hiera_tiny.pt"),
        ]

    # 3. Xác định danh sách VLM cần tải
    vlm_targets: list[tuple[str, Path]] = []
    if args.vlm_model in ("qwen", "both"):
        q = VLM_REPOS["qwen"]
        vlm_targets.append((q["repo_id"], vlm_dir / q["folder"]))
    if args.vlm_model in ("florence", "both"):
        f = VLM_REPOS["florence"]
        vlm_targets.append((f["repo_id"], vlm_dir / f["folder"]))

    # In kế hoạch
    print("\n[KẾ HOẠCH TẢI]:")
    if sam_targets:
        print("  1. SAM 2 Checkpoints:")
        for url, p in sam_targets:
            print(f"     - {p.name} từ {url}")
    else:
        print("  1. SAM 2: Bỏ qua (skip).")

    if vlm_targets:
        print("  2. VLM Repositories:")
        for repo_id, p in vlm_targets:
            print(f"     - {repo_id} -> {p}")
    else:
        print("  2. VLM: Bỏ qua (skip).")

    if args.dry_run:
        print("\n[DRY-RUN] Kiểm tra thành công. Không có file nào được tải xuống.")
        return 0

    # Lấy token Hugging Face nếu có trong .env
    token = None
    env_file = ROOT / ".env"
    if env_file.is_file():
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
        token = os.getenv("HF_TOKEN") or None

    # Thực thi tải SAM 2
    if sam_targets:
        print("\n===> BẮT ĐẦU TẢI SAM 2 CHECKPOINTS...")
        for url, dest in sam_targets:
            download_file_with_progress(url, dest)

    # Thực thi tải VLM
    if vlm_targets:
        print("\n===> BẮT ĐẦU TẢI VLM REPOSITORIES TỪ HUGGING FACE...")
        for repo_id, dest in vlm_targets:
            download_hf_repo(repo_id, dest, hf_token=token)

    print("\n" + "=" * 76)
    print("HOÀN TẤT TẢI TẤT CẢ MÔ HÌNH YÊU CẦU CHO PIPELINE SAM 2 + VLM!")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())

