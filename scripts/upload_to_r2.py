"""
Script: upload_to_r2.py
Description: Tải toàn bộ ảnh sản phẩm lên Cloudflare R2 theo đúng phân loại và quy ước của YoFaSoShop:
  - Bucket yofasoshop-catalog (Public):
      * catalog/products/{slug}/{slug}--img{NN}--v{VV}--original.png
      * catalog/products/{slug}/{slug}--img{NN}--v{VV}--cutout.png
      * catalog/products/{slug}/{slug}--ref{NN}--v{VV}--original.png
  - Bucket yofasoshop-private (Private):
      * catalog-processing/products/{slug}/{slug}--img{NN}--v{VV}--mask.png
      * catalog-processing/products/{slug}/{slug}--ref{NN}--v{VV}--reference.png
"""

import os
import sys
import argparse
import re
import csv
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Đảm bảo in UTF-8 trên Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None


def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / "YoFaSoShop_Backend").exists() and (p / "YoFaSoShop_AIService").exists():
            return p
    return current.parent.parent


def load_r2_credentials(repo_root: Path):
    if load_dotenv:
        backend_env = repo_root / "YoFaSoShop_Backend" / ".env"
        ai_env = repo_root / "YoFaSoShop_AIService" / ".env"
        root_env = repo_root / ".env"
        for env_path in [backend_env, ai_env, root_env]:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)

    endpoint = os.getenv("R2_ENDPOINT")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    region = os.getenv("R2_REGION", "auto")
    bucket_public = os.getenv("R2_BUCKET_PUBLIC", "yofasoshop-catalog")
    bucket_private = os.getenv("R2_BUCKET_PRIVATE", "yofasoshop-private")
    public_base_url = os.getenv("R2_PUBLIC_BASE_URL", "").rstrip("/")

    return {
        "endpoint": endpoint,
        "access_key": access_key,
        "secret_key": secret_key,
        "region": region,
        "bucket_public": bucket_public,
        "bucket_private": bucket_private,
        "public_base_url": public_base_url,
    }


def scan_product_files(garments_dir: Path):
    """
    Quét toàn bộ thư mục sản phẩm và phân loại 5 nhóm ảnh:
    1. --original.png (biến thể) -> catalog
    2. --cutout.png (bóc nền) -> catalog
    3. --ref*--original.png (mẫu gốc) -> catalog
    4. --mask.png (mặt nạ thử đồ) -> private
    5. --ref*--reference.png (mẫu tham chiếu AI) -> private
    """
    tasks = []

    for md_path in garments_dir.rglob("product_analysis.md"):
        pdir = md_path.parent

        # Đọc slug từ product_analysis.md hoặc suy ra từ file
        slug = None
        try:
            content = md_path.read_text(encoding="utf-8")
            m = re.search(r'"MaLuuFile"\s*:\s*"([^"]+)"', content)
            if not m:
                m = re.search(r'"product_slug"\s*:\s*"([^"]+)"', content)
            if m:
                slug = m.group(1).strip()
        except Exception:
            pass

        # Quét các file trong thư mục gốc của sản phẩm
        for f in pdir.iterdir():
            if not f.is_file() or f.suffix.lower() != ".png":
                continue
            fname = f.name

            # Mẫu tham chiếu người mẫu: {slug}--ref{NN}--v{VV}--original.png
            if "--ref" in fname and "--original.png" in fname:
                cur_slug = slug or fname.split("--ref")[0]
                tasks.append({
                    "local_path": f,
                    "bucket_type": "public",
                    "object_key": f"catalog/products/{cur_slug}/{fname}",
                    "file_type": "model_ref_original",
                    "size_bytes": f.stat().st_size
                })
            # Mẫu tham chiếu người mẫu: {slug}--ref{NN}--v{VV}--reference.png
            elif "--ref" in fname and "--reference.png" in fname:
                cur_slug = slug or fname.split("--ref")[0]
                tasks.append({
                    "local_path": f,
                    "bucket_type": "private",
                    "object_key": f"catalog-processing/products/{cur_slug}/{fname}",
                    "file_type": "model_ref_reference",
                    "size_bytes": f.stat().st_size
                })
            # Ảnh gốc biến thể: {slug}--img{NN}--v{VV}--original.png
            elif "--img" in fname and "--original.png" in fname:
                cur_slug = slug or fname.split("--img")[0]
                tasks.append({
                    "local_path": f,
                    "bucket_type": "public",
                    "object_key": f"catalog/products/{cur_slug}/{fname}",
                    "file_type": "variant_original",
                    "size_bytes": f.stat().st_size
                })

        # Quét thư mục background_removed
        bg_dir = pdir / "background_removed"
        if bg_dir.exists() and bg_dir.is_dir():
            for f in bg_dir.iterdir():
                if f.is_file() and f.name.endswith("--cutout.png"):
                    cur_slug = slug or f.name.split("--img")[0]
                    tasks.append({
                        "local_path": f,
                        "bucket_type": "public",
                        "object_key": f"catalog/products/{cur_slug}/{f.name}",
                        "file_type": "variant_cutout",
                        "size_bytes": f.stat().st_size
                    })

        # Quét thư mục tryon_masks
        mask_dir = pdir / "tryon_masks"
        if mask_dir.exists() and mask_dir.is_dir():
            for f in mask_dir.iterdir():
                if f.is_file() and f.name.endswith("--mask.png"):
                    cur_slug = slug or f.name.split("--img")[0]
                    tasks.append({
                        "local_path": f,
                        "bucket_type": "private",
                        "object_key": f"catalog-processing/products/{cur_slug}/{f.name}",
                        "file_type": "variant_mask",
                        "size_bytes": f.stat().st_size
                    })

    return tasks


class R2Uploader:
    def __init__(self, config):
        if boto3 is None:
            raise RuntimeError("Chưa cài đặt boto3. Hãy chạy `pip install boto3`")
        self.config = config
        self.bucket_public = config["bucket_public"]
        self.bucket_private = config["bucket_private"]
        
        boto_config = Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            connect_timeout=15,
            read_timeout=30
        )
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=config["endpoint"],
            aws_access_key_id=config["access_key"],
            aws_secret_access_key=config["secret_key"],
            region_name=config["region"],
            config=boto_config
        )
        self.lock = Lock()
        self.uploaded_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        self.total_bytes = 0

    def object_exists_and_matches(self, bucket: str, key: str, expected_size: int) -> bool:
        try:
            resp = self.s3_client.head_object(Bucket=bucket, Key=key)
            if resp.get("ContentLength") == expected_size:
                return True
            return False
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            return False
        except Exception:
            return False

    def upload_single_file(self, task: dict, skip_existing: bool = True) -> dict:
        local_path = task["local_path"]
        bucket = self.bucket_public if task["bucket_type"] == "public" else self.bucket_private
        key = task["object_key"]
        size = task["size_bytes"]

        res = {
            "file": local_path.name,
            "bucket": bucket,
            "key": key,
            "size": size,
            "status": "pending",
            "error": None
        }

        try:
            if skip_existing and self.object_exists_and_matches(bucket, key, size):
                with self.lock:
                    self.skipped_count += 1
                res["status"] = "skipped"
                return res

            self.s3_client.upload_file(
                Filename=str(local_path),
                Bucket=bucket,
                Key=key,
                ExtraArgs={
                    "ContentType": "image/png"
                }
            )
            with self.lock:
                self.uploaded_count += 1
                self.total_bytes += size
            res["status"] = "uploaded"
            return res

        except Exception as ex:
            with self.lock:
                self.failed_count += 1
            res["status"] = "failed"
            res["error"] = str(ex)
            return res


def main():
    parser = argparse.ArgumentParser(description="Upload ảnh sản phẩm YoFaSoShop lên Cloudflare R2")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra và in thông tin, không upload")
    parser.add_argument("--concurrency", type=int, default=16, help="Số luồng upload đồng thời (mặc định: 16)")
    parser.add_argument("--force", action="store_true", help="Bỏ qua kiểm tra file đã tồn tại trên R2, ghi đè toàn bộ")
    parser.add_argument("--manifest", type=str, default="r2_upload_manifest.csv", help="Tên file manifest CSV xuất ra")
    parser.add_argument("--product-dir", type=str, default=None, help="Chỉ định thư mục sản phẩm cụ thể để upload")
    args = parser.parse_args()

    repo_root = find_repo_root()
    credentials = load_r2_credentials(repo_root)

    print("=" * 70)
    print("🚀 YOFASOSHOP - CLOUDFLARE R2 BATCH UPLOADER")
    print("=" * 70)
    print(f"📁 Repository Root : {repo_root}")
    print(f"🌐 Public Bucket   : {credentials['bucket_public']}")
    print(f"🔒 Private Bucket  : {credentials['bucket_private']}")
    print(f"🔗 S3 Endpoint     : {credentials['endpoint']}")
    if credentials["public_base_url"]:
        print(f"🌍 Public Base URL : {credentials['public_base_url']}")
    print(f"⚙️ Concurrency     : {args.concurrency} worker threads")
    print(f"⚡ Mode            : {'DRY RUN (Xem trước)' if args.dry_run else 'REAL UPLOAD'}")
    print("=" * 70)

    if not args.dry_run:
        if not credentials["endpoint"] or not credentials["access_key"] or not credentials["secret_key"]:
            print("❌ LỖI: Chưa cấu hình đầy đủ biến môi trường R2 (R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY).")
            print("Vui lòng kiểm tra file YoFaSoShop_Backend/.env hoặc YoFaSoShop_AIService/.env.")
            sys.exit(1)

    garments_dir = repo_root / "YoFaSoShop_AIService" / "data" / "garments_categories"
    if args.product_dir:
        target_dir = Path(args.product_dir).resolve()
    else:
        target_dir = garments_dir

    print(f"\n🔍 Đang quét file ảnh tại: {target_dir}...")
    tasks = scan_product_files(target_dir)

    total_tasks = len(tasks)
    total_size = sum(t["size_bytes"] for t in tasks)
    public_tasks = [t for t in tasks if t["bucket_type"] == "public"]
    private_tasks = [t for t in tasks if t["bucket_type"] == "private"]

    by_type = {}
    for t in tasks:
        by_type[t["file_type"]] = by_type.get(t["file_type"], 0) + 1

    print(f"📊 KẾT QUẢ QUÉT:")
    print(f"   • Tổng số file ảnh cần xử lý : {total_tasks:,} files ({total_size / (1024*1024):.2f} MB)")
    print(f"   • Bucket Public (catalog)    : {len(public_tasks):,} files")
    print(f"     - Biến thể original        : {by_type.get('variant_original', 0):,} files")
    print(f"     - Biến thể cutout (bóc nền): {by_type.get('variant_cutout', 0):,} files")
    print(f"     - Model ref original       : {by_type.get('model_ref_original', 0):,} files")
    print(f"   • Bucket Private (private)   : {len(private_tasks):,} files")
    print(f"     - Biến thể mask            : {by_type.get('variant_mask', 0):,} files")
    print(f"     - Model ref reference      : {by_type.get('model_ref_reference', 0):,} files")
    print("=" * 70)

    if total_tasks == 0:
        print("⚠️ Không tìm thấy file ảnh nào cần upload. Kiểm tra lại đường dẫn.")
        return

    # Nếu chỉ dry-run, in 10 dòng mẫu và lưu manifest danh sách
    if args.dry_run:
        print("\n🔎 Xem trước 10 file đầu tiên:")
        for i, t in enumerate(tasks[:10], 1):
            print(f"  [{i:02d}] {t['bucket_type'].upper():7s} -> {t['object_key']}")
        print(f"  ... và {total_tasks - 10} file khác.")

        manifest_path = repo_root / args.manifest
        with open(manifest_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["local_path", "bucket_type", "object_key", "file_type", "size_bytes"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for t in tasks:
                writer.writerow({
                    "local_path": str(t["local_path"]),
                    "bucket_type": t["bucket_type"],
                    "object_key": t["object_key"],
                    "file_type": t["file_type"],
                    "size_bytes": t["size_bytes"]
                })
        print(f"\n📄 Đã ghi danh sách kiểm tra xem trước vào: {manifest_path}")
        print("💡 Để tiến hành upload thật lên R2, chạy lệnh:")
        print("   python scripts/upload_to_r2.py")
        return

    # Bắt đầu upload thật
    uploader = R2Uploader(credentials)
    skip_existing = not args.force

    start_time = time.time()
    results = []

    print(f"\n🚀 Bắt đầu upload {total_tasks:,} file với {args.concurrency} workers...")
    print("   (Nhấn Ctrl+C bất cứ lúc nào để dừng, chạy lại script sẽ tự động tiếp tục các file chưa xong)\n")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_task = {
            executor.submit(uploader.upload_single_file, task, skip_existing): task
            for task in tasks
        }

        completed = 0
        last_report_time = time.time()

        for future in as_completed(future_to_task):
            res = future.result()
            results.append(res)
            completed += 1

            now = time.time()
            if completed % 50 == 0 or completed == total_tasks or (now - last_report_time) > 3.0:
                last_report_time = now
                elapsed = now - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                pct = (completed / total_tasks) * 100
                print(
                    f"   ⏳ Tiến độ: [{completed:4d}/{total_tasks:4d}] ({pct:5.1f}%) | "
                    f"Tải lên: {uploader.uploaded_count:4d} | "
                    f"Đã có sẵn: {uploader.skipped_count:4d} | "
                    f"Lỗi: {uploader.failed_count:2d} | "
                    f"Tốc độ: {speed:4.1f} f/s",
                    end="\r"
                )

    total_time = time.time() - start_time
    print("\n\n" + "=" * 70)
    print("🎉 HOÀN TẤT QUÁ TRÌNH UPLOAD!")
    print("=" * 70)
    print(f"⏱️ Tổng thời gian       : {total_time:.2f} giây")
    print(f"✅ Đã tải lên mới       : {uploader.uploaded_count:,} files ({(uploader.total_bytes / (1024*1024)):.2f} MB)")
    print(f"⏩ Bỏ qua (đã có sẵn)   : {uploader.skipped_count:,} files")
    print(f"❌ Số file thất bại     : {uploader.failed_count:,} files")

    # Xuất báo cáo manifest
    manifest_path = repo_root / args.manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["file", "bucket", "key", "size", "status", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"📄 Báo cáo chi tiết kết quả đã được ghi vào: {manifest_path}")

    if uploader.failed_count > 0:
        print("\n⚠️ Một số file tải lên bị lỗi. Bạn có thể chạy lại lệnh để tải lại các file đó.")
    else:
        print("\n🌟 Toàn bộ file đã được đồng bộ chuẩn xác 100% lên Cloudflare R2!")


if __name__ == "__main__":
    main()
