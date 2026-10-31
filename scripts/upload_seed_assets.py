"""
Script: upload_seed_assets.py
Description: Tải các tài nguyên dữ liệu mẫu (Seed Data) lên Cloudflare R2:
  1. Ảnh kết quả VTON (5 phiên thử đồ mẫu):
     - Bucket: yofasoshop-private
     - Key: tryon/{MaPhienThuDo}/{uuid}--person.png (Ảnh gốc khách lấy từ test_samples/persons)
     - Key: tryon/{MaPhienThuDo}/{uuid}--result.png (Ảnh kết quả VTON lấy từ garments_categories/results)
  2. Ảnh đại diện người dùng (15 tài khoản gồm 1 Admin, 5 Khách hàng, 3 Thu ngân, 3 Thủ kho, 3 Shipper):
     - Bucket: yofasoshop-catalog
     - Key: avatars/{MaNguoiDung}/avatar-{uuid}.png (Lấy ngẫu nhiên từ test_samples/persons)
  3. Tạo thư mục placeholder 'reviews/' trên bucket yofasoshop-catalog:
     - Key: reviews/.keep
  4. Xuất file seed_data_manifest.json & seed_sql_reference.sql để hỗ trợ viết script INSERT vào MySQL.
"""

import os
import sys
import argparse
import random
import uuid
import json
import csv
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

    return {
        "endpoint": os.getenv("R2_ENDPOINT"),
        "access_key": os.getenv("R2_ACCESS_KEY_ID"),
        "secret_key": os.getenv("R2_SECRET_ACCESS_KEY"),
        "region": os.getenv("R2_REGION", "auto"),
        "bucket_public": os.getenv("R2_BUCKET_PUBLIC", "yofasoshop-catalog"),
        "bucket_private": os.getenv("R2_BUCKET_PRIVATE", "yofasoshop-private"),
        "public_base_url": os.getenv("R2_PUBLIC_BASE_URL", "").rstrip("/"),
    }


# Danh sách 15 người dùng mẫu chuẩn theo ERD_optimize.md
SAMPLE_USERS = [
    {"id": "ND00000001", "role": "ROLE_SHOP_ADMIN", "role_title": "Quản trị viên", "name": "Nguyễn Văn Admin", "gender": "NAM"},
    # 5 Khách hàng
    {"id": "ND00000002", "role": "ROLE_CUSTOMER", "role_title": "Khách hàng", "name": "Trần Thị Mai Anh", "gender": "NU"},
    {"id": "ND00000003", "role": "ROLE_CUSTOMER", "role_title": "Khách hàng", "name": "Lê Hoàng Long", "gender": "NAM"},
    {"id": "ND00000004", "role": "ROLE_CUSTOMER", "role_title": "Khách hàng", "name": "Phạm Quỳnh Chi", "gender": "NU"},
    {"id": "ND00000005", "role": "ROLE_CUSTOMER", "role_title": "Khách hàng", "name": "Vũ Minh Quân", "gender": "NAM"},
    {"id": "ND00000006", "role": "ROLE_CUSTOMER", "role_title": "Khách hàng", "name": "Hoàng Thu Thảo", "gender": "NU"},
    # 3 Thu ngân
    {"id": "ND00000007", "role": "ROLE_STAFF_CASHIER", "role_title": "Thu ngân", "name": "Đặng Lan Phương", "gender": "NU"},
    {"id": "ND00000008", "role": "ROLE_STAFF_CASHIER", "role_title": "Thu ngân", "name": "Bùi Đức Thịnh", "gender": "NAM"},
    {"id": "ND00000009", "role": "ROLE_STAFF_CASHIER", "role_title": "Thu ngân", "name": "Ngô Bích Ngọc", "gender": "NU"},
    # 3 Thủ kho
    {"id": "ND00000010", "role": "ROLE_STAFF_WAREHOUSE", "role_title": "Thủ kho", "name": "Đỗ Mạnh Hùng", "gender": "NAM"},
    {"id": "ND00000011", "role": "ROLE_STAFF_WAREHOUSE", "role_title": "Thủ kho", "name": "Trương Văn Khải", "gender": "NAM"},
    {"id": "ND00000012", "role": "ROLE_STAFF_WAREHOUSE", "role_title": "Thủ kho", "name": "Lý Gia Huy", "gender": "NAM"},
    # 3 Shipper
    {"id": "ND00000013", "role": "ROLE_STAFF_SHIPPER", "role_title": "Shipper", "name": "Phan Quốc Bảo", "gender": "NAM"},
    {"id": "ND00000014", "role": "ROLE_STAFF_SHIPPER", "role_title": "Shipper", "name": "Dương Tuấn Kiệt", "gender": "NAM"},
    {"id": "ND00000015", "role": "ROLE_STAFF_SHIPPER", "role_title": "Shipper", "name": "Lương Tiến Đạt", "gender": "NAM"},
]


def prepare_seed_tasks(repo_root: Path, seed: int = 42):
    """
    Chuẩn bị danh sách upload và metadata cho:
      1. 5 phiên thử đồ ảo VTON (PhienThuDoAo)
      2. 15 ảnh đại diện người dùng (NguoiDung)
      3. Marker folder 'reviews/'
    """
    # Cố định seed để nếu chạy lại nhiều lần vẫn giữ nguyên ánh xạ ảnh
    rng = random.Random(seed)

    ai_data = repo_root / "YoFaSoShop_AIService" / "data"
    results_dir = ai_data / "garments_categories" / "results"
    persons_dir = ai_data / "test_samples" / "persons"

    person_files = sorted([f for f in persons_dir.glob("*.png") if f.is_file()])
    if not person_files:
        raise FileNotFoundError(f"Không tìm thấy ảnh người mẫu trong {persons_dir}")

    tasks = []
    metadata = {
        "tryon_sessions": [],
        "users": [],
        "folders": []
    }

    # 1. Chuẩn bị 5 phiên thử đồ ảo VTON
    # Lấy 5 ảnh kết quả: result1.png -> result5.png
    # Chọn 5 ảnh người mẫu tương ứng
    shuffled_persons = list(person_files)
    rng.shuffle(shuffled_persons)

    for i in range(1, 6):
        session_id = f"TD{i:08d}"  # TD00000001 .. TD00000005
        result_file = results_dir / f"result{i}.png"
        if not result_file.exists():
            print(f"⚠️ Cảnh báo: Không tìm thấy {result_file}")
            continue

        person_file = shuffled_persons[i - 1]

        # Sinh mã UUID ngắn gọn 8 ký tự
        uuid_person = uuid.uuid4().hex[:8]
        uuid_result = uuid.uuid4().hex[:8]

        key_person = f"tryon/{session_id}/{uuid_person}--person.png"
        key_result = f"tryon/{session_id}/{uuid_result}--result.png"

        # Task ảnh người gốc
        tasks.append({
            "local_path": person_file,
            "bucket_type": "private",
            "object_key": key_person,
            "content_type": "image/png",
            "size_bytes": person_file.stat().st_size,
            "description": f"Ảnh gốc phiên {session_id}"
        })

        # Task ảnh kết quả VTON
        tasks.append({
            "local_path": result_file,
            "bucket_type": "private",
            "object_key": key_result,
            "content_type": "image/png",
            "size_bytes": result_file.stat().st_size,
            "description": f"Ảnh kết quả VTON {session_id}"
        })

        metadata["tryon_sessions"].append({
            "MaPhienThuDo": session_id,
            "DuongDanAnhNguoiDung": key_person,
            "DuongDanAnhKetQuaVTON": key_result,
            "source_person_file": person_file.name,
            "source_result_file": result_file.name
        })

    # 2. Chuẩn bị ảnh avatar cho người dùng (15 người)
    # Lấy 15 ảnh người mẫu ngẫu nhiên từ danh sách còn lại
    avatar_pool = shuffled_persons[5:20]
    if len(avatar_pool) < len(SAMPLE_USERS):
        avatar_pool = (avatar_pool * 2)[:len(SAMPLE_USERS)]

    for idx, user in enumerate(SAMPLE_USERS):
        avatar_src = avatar_pool[idx]
        user_id = user["id"]
        uuid_avatar = uuid.uuid4().hex[:8]
        key_avatar = f"avatars/{user_id}/avatar-{uuid_avatar}.png"

        tasks.append({
            "local_path": avatar_src,
            "bucket_type": "public",
            "object_key": key_avatar,
            "content_type": "image/png",
            "size_bytes": avatar_src.stat().st_size,
            "description": f"Avatar cho {user['name']} ({user_id})"
        })

        metadata["users"].append({
            "MaNguoiDung": user_id,
            "HoVaTen": user["name"],
            "VaiTro": user["role"],
            "TenVaiTro": user["role_title"],
            "GioiTinh": user["gender"],
            "AnhDaiDien": key_avatar,
            "source_file": avatar_src.name
        })

    # 3. Tạo placeholder marker cho thư mục reviews/
    tasks.append({
        "local_path": None,  # Bộ nhớ ảo
        "raw_bytes": b"",
        "bucket_type": "public",
        "object_key": "reviews/.keep",
        "content_type": "application/octet-stream",
        "size_bytes": 0,
        "description": "Folder marker cho reviews/"
    })
    metadata["folders"].append("reviews/")

    return tasks, metadata


class SeedUploader:
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

    def object_exists_and_matches(self, bucket: str, key: str, expected_size: int) -> bool:
        try:
            resp = self.s3_client.head_object(Bucket=bucket, Key=key)
            if resp.get("ContentLength") == expected_size:
                return True
            return False
        except ClientError:
            return False
        except Exception:
            return False

    def upload_task(self, task: dict, skip_existing: bool = True) -> dict:
        bucket = self.bucket_public if task["bucket_type"] == "public" else self.bucket_private
        key = task["object_key"]
        size = task["size_bytes"]

        res = {
            "key": key,
            "bucket": bucket,
            "description": task["description"],
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

            if task.get("local_path"):
                self.s3_client.upload_file(
                    Filename=str(task["local_path"]),
                    Bucket=bucket,
                    Key=key,
                    ExtraArgs={"ContentType": task["content_type"]}
                )
            else:
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=task.get("raw_bytes", b""),
                    ContentType=task["content_type"]
                )

            with self.lock:
                self.uploaded_count += 1
            res["status"] = "uploaded"
            return res

        except Exception as ex:
            with self.lock:
                self.failed_count += 1
            res["status"] = "failed"
            res["error"] = str(ex)
            return res


def export_sql_reference(repo_root: Path, metadata: dict):
    """
    Tạo file SQL mẫu INSERT vào MySQL để người dùng tham khảo ngay lập tức.
    """
    sql_path = repo_root / "seed_r2_reference.sql"
    lines = [
        "-- =====================================================================",
        "-- DỮ LIỆU THAM CHIẾU CLOUDFLARE R2 CHO MYSQL (THEO ERD_OPTIMIZE.MD)",
        f"-- Ngày tạo: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "-- =====================================================================",
        "",
        "-- 1. CẬP NHẬT HOẶC THÊM AVATAR NGƯỜI DÙNG (Bảng NguoiDung):",
    ]

    for u in metadata["users"]:
        lines.append(
            f"-- User: {u['HoVaTen']} ({u['TenVaiTro']})\n"
            f"UPDATE NguoiDung SET AnhDaiDien = '{u['AnhDaiDien']}' WHERE MaNguoiDung = '{u['MaNguoiDung']}';"
        )

    lines.append("\n-- 2. ĐƯỜNG DẪN PHIÊN THỬ ĐỒ VTON (Bảng PhienThuDoAo):")
    for s in metadata["tryon_sessions"]:
        lines.append(
            f"-- Phiên {s['MaPhienThuDo']}:\n"
            f"-- DuongDanAnhNguoiDung  : '{s['DuongDanAnhNguoiDung']}'\n"
            f"-- DuongDanAnhKetQuaVTON : '{s['DuongDanAnhKetQuaVTON']}'"
        )

    sql_path.write_text("\n".join(lines), encoding="utf-8")
    return sql_path


def main():
    parser = argparse.ArgumentParser(description="Upload ảnh mẫu VTON, Avatar và tạo folder Reviews trên R2")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ xem trước các file sẽ upload, không thực thi")
    parser.add_argument("--concurrency", type=int, default=8, help="Số luồng upload đồng thời (mặc định: 8)")
    parser.add_argument("--force", action="store_true", help="Ghi đè file nếu đã tồn tại trên bucket")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên để cố định ánh xạ ảnh")
    args = parser.parse_args()

    repo_root = find_repo_root()
    credentials = load_r2_credentials(repo_root)

    print("=" * 70)
    print("🚀 YOFASOSHOP - CLOUDFLARE R2 SEED ASSETS UPLOADER")
    print("=" * 70)
    print(f"📁 Repository Root : {repo_root}")
    print(f"🌐 Public Bucket   : {credentials['bucket_public']}")
    print(f"🔒 Private Bucket  : {credentials['bucket_private']}")
    print(f"🔗 S3 Endpoint     : {credentials['endpoint']}")
    print(f"⚡ Mode            : {'DRY RUN (Xem trước)' if args.dry_run else 'REAL UPLOAD'}")
    print("=" * 70)

    if not args.dry_run:
        if not credentials["endpoint"] or not credentials["access_key"] or not credentials["secret_key"]:
            print("❌ LỖI: Chưa cấu hình đầy đủ biến môi trường R2.")
            sys.exit(1)

    tasks, metadata = prepare_seed_tasks(repo_root, seed=args.seed)

    print(f"\n📋 DANH SÁCH TÀI NGUYÊN SẼ UPLOAD ({len(tasks)} mục):")
    for idx, t in enumerate(tasks, 1):
        bucket_name = credentials["bucket_public"] if t["bucket_type"] == "public" else credentials["bucket_private"]
        print(f"  [{idx:02d}] [{t['bucket_type'].upper():7s}] ({bucket_name})")
        print(f"       Key: {t['object_key']}")
        print(f"       Mô tả: {t['description']}")

    # Lưu metadata json
    json_path = repo_root / "seed_r2_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Đã ghi metadata JSON vào: {json_path}")

    # Xuất file SQL tham chiếu
    sql_path = export_sql_reference(repo_root, metadata)
    print(f"📄 Đã sinh file SQL mẫu tham chiếu: {sql_path}")

    if args.dry_run:
        print("\n💡 Chạy lệnh sau để tiến hành upload thật:")
        print("   python scripts/upload_seed_assets.py")
        return

    # Bắt đầu upload thật
    uploader = SeedUploader(credentials)
    skip_existing = not args.force

    print(f"\n🚀 Bắt đầu upload {len(tasks)} mục lên Cloudflare R2...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(uploader.upload_task, t, skip_existing) for t in tasks]
        for f in as_completed(futures):
            f.result()

    elapsed = time.time() - start_time
    print("=" * 70)
    print("🎉 HOÀN TẤT UPLOAD SEED DATA!")
    print("=" * 70)
    print(f"⏱️ Thời gian thực thi    : {elapsed:.2f} giây")
    print(f"✅ Đã tải lên mới        : {uploader.uploaded_count} files/objects")
    print(f"⏩ Đã có sẵn (bỏ qua)    : {uploader.skipped_count} files/objects")
    print(f"❌ Số mục thất bại       : {uploader.failed_count} files/objects")
    print(f"📄 File SQL tham chiếu   : {sql_path.name}")
    print(f"📄 File Metadata JSON    : {json_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
