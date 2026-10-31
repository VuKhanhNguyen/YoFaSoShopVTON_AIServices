"""
Script: generate_mysql_seed_data.py
Description: Sinh toàn bộ dữ liệu mẫu (Seed Data) chuẩn xác 100% cho 19 bảng của YoFaSoShop
theo thiết kế tại ERD_optimize.md, liên kết Cloudflare R2 và quy chuẩn sáp nhập địa giới 1/7/2025.
"""

import sys
import os
import re
import json
from pathlib import Path

# Đảm bảo UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def escape_sql(val):
    if val is None:
        return "NULL"
    val_str = str(val)
    val_str = val_str.replace("\\", "\\\\").replace("'", "''")
    return f"'{val_str}'"


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    ai_data = repo_root / "YoFaSoShop_AIService" / "data"
    garments_dir = ai_data / "garments_categories"
    
    # 1. Đọc metadata R2
    meta_path = repo_root / "seed_r2_metadata.json"
    r2_meta = {}
    if meta_path.exists():
        r2_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    # 2. Đọc địa chỉ sáp nhập
    addr_path = repo_root / "YoFaSoShop_AIService" / "scratch" / "real_addresses.json"
    real_addresses = []
    if addr_path.exists():
        real_addresses = json.loads(addr_path.read_text(encoding="utf-8"))
    
    print(f"Loaded {len(real_addresses)} post-2025 addresses.")
    print(f"Scanning garments at {garments_dir}...")
    
    # ---------------------------------------------------------
    # BẢNG 1: VaiTro (5 roles)
    # ---------------------------------------------------------
    roles = [
        ("VT001", "ROLE_SHOP_ADMIN", "Quản trị viên", "Toàn quyền quản trị hệ thống, danh mục, đơn hàng và cấu hình shop"),
        ("VT002", "ROLE_CUSTOMER", "Khách hàng", "Khách hàng mua sắm online, thử đồ ảo AI và đánh giá sản phẩm"),
        ("VT003", "ROLE_STAFF_CASHIER", "Nhân viên thu ngân", "Bán lẻ tại quầy Web POS, tra cứu tích điểm và chốt kết ca"),
        ("VT004", "ROLE_STAFF_WAREHOUSE", "Nhân viên kho", "Quản lý xuất nhập kho, nhặt hàng đóng gói và kiểm kê hàng hóa"),
        ("VT005", "ROLE_STAFF_SHIPPER", "Nhân viên giao hàng", "Tiếp nhận kiện hàng xuất kho, giao hàng, thu tiền COD và bàn giao hoàn"),
    ]
    
    # ---------------------------------------------------------
    # BẢNG 2 & 3: NguoiDung & NguoiDung_VaiTro (15 users)
    # ---------------------------------------------------------
    # Mật khẩu BCrypt chuẩn cho "123456": $2a$10$7R9rR5tV3uW6lO.P1kK5ceM9xT4rW2yZ1q8bJ4s5s8p.n8rT8v9Ou
    DEFAULT_PW = "$2a$10$7R9rR5tV3uW6lO.P1kK5ceM9xT4rW2yZ1q8bJ4s5s8p.n8rT8v9Ou"
    
    user_map = {}
    if "users" in r2_meta:
        for u in r2_meta["users"]:
            user_map[u["MaNguoiDung"]] = u
            
    users_data = [
        # Admin
        ("ND00000001", "admin_yofaso", DEFAULT_PW, "Nguyễn Văn Admin", "0901234567", "admin@yofaso.vn", 0, "KIM_CUONG", "1988-05-15", "NAM", "VT001"),
        # 5 Khách hàng
        ("ND00000002", "maianh_tran", DEFAULT_PW, "Trần Thị Mai Anh", "0912345678", "maianh.tran@gmail.com", 350, "VANG", "1998-10-20", "NU", "VT002"),
        ("ND00000003", "hoanglong_le", DEFAULT_PW, "Lê Hoàng Long", "0923456789", "hoanglong.le@gmail.com", 180, "BAC", "1995-03-12", "NAM", "VT002"),
        ("ND00000004", "quynhchi_pham", DEFAULT_PW, "Phạm Quỳnh Chi", "0934567890", "quynhchi.pham@gmail.com", 520, "KIM_CUONG", "2000-08-25", "NU", "VT002"),
        ("ND00000005", "minhquan_vu", DEFAULT_PW, "Vũ Minh Quân", "0945678901", "minhquan.vu@gmail.com", 80, "DONG", "1997-12-05", "NAM", "VT002"),
        ("ND00000006", "thuthao_hoang", DEFAULT_PW, "Hoàng Thu Thảo", "0956789012", "thuthao.hoang@gmail.com", 210, "BAC", "2001-02-18", "NU", "VT002"),
        # 3 Thu ngân
        ("ND00000007", "cashier_lanphuong", DEFAULT_PW, "Đặng Lan Phương", "0967890123", "lanphuong.dang@yofaso.vn", 0, "DONG", "1999-07-14", "NU", "VT003"),
        ("ND00000008", "cashier_ducthinh", DEFAULT_PW, "Bùi Đức Thịnh", "0978901234", "ducthinh.bui@yofaso.vn", 0, "DONG", "1996-09-30", "NAM", "VT003"),
        ("ND00000009", "cashier_bichngoc", DEFAULT_PW, "Ngô Bích Ngọc", "0989012345", "bichngoc.ngo@yofaso.vn", 0, "DONG", "2002-04-08", "NU", "VT003"),
        # 3 Thủ kho
        ("ND00000010", "wh_manhhung", DEFAULT_PW, "Đỗ Mạnh Hùng", "0990123456", "manhhung.do@yofaso.vn", 0, "DONG", "1993-11-22", "NAM", "VT004"),
        ("ND00000011", "wh_vankhai", DEFAULT_PW, "Trương Văn Khải", "0909876543", "vankhai.truong@yofaso.vn", 0, "DONG", "1994-01-19", "NAM", "VT004"),
        ("ND00000012", "wh_giahuy", DEFAULT_PW, "Lý Gia Huy", "0918765432", "giahuy.ly@yofaso.vn", 0, "DONG", "1997-06-27", "NAM", "VT004"),
        # 3 Shipper
        ("ND00000013", "ship_quocbao", DEFAULT_PW, "Phan Quốc Bảo", "0927654321", "quocbao.phan@yofaso.vn", 0, "DONG", "1996-08-11", "NAM", "VT005"),
        ("ND00000014", "ship_tuankiet", DEFAULT_PW, "Dương Tuấn Kiệt", "0936543210", "tuankiet.duong@yofaso.vn", 0, "DONG", "1998-10-03", "NAM", "VT005"),
        ("ND00000015", "ship_tiendat", DEFAULT_PW, "Lương Tiến Đạt", "0945432109", "tiendat.luong@yofaso.vn", 0, "DONG", "2000-12-16", "NAM", "VT005"),
    ]
    
    # ---------------------------------------------------------
    # BẢNG 4: SoDiaChi (15 địa chỉ sau sáp nhập 1/7/2025)
    # ---------------------------------------------------------
    so_dia_chi = []
    for i, u in enumerate(users_data):
        addr_info = real_addresses[i % len(real_addresses)]
        dc_id = f"DC{i+1:08d}"
        so_dia_chi.append({
            "MaDiaChi": dc_id,
            "MaNguoiDung": u[0],
            "TenNguoiNhan": u[3],
            "SoDienThoaiNhan": u[4],
            "SoNhaTenDuong": addr_info["street"],
            "PhuongXa": addr_info["ward"],
            "TinhThanhPho": addr_info["province"],
            "LaMacDinh": 1
        })

    # ---------------------------------------------------------
    # BẢNG 5: DanhMuc (Cây phân cấp 31 danh mục chuẩn)
    # ---------------------------------------------------------
    danh_muc = [
        # Gốc
        ("DM000001", "Thời trang Nam", None, "Danh mục toàn bộ trang phục thời trang dành cho nam giới", 1),
        ("DM000002", "Thời trang Nữ", None, "Danh mục toàn bộ trang phục thời trang dành cho nữ giới", 2),
        # Cấp 2 Nam
        ("DM000003", "Áo nam", "DM000001", "Các dòng áo nam thời thượng, lịch lãm", 1),
        ("DM000004", "Quần nam", "DM000001", "Các kiểu quần nam chuẩn form, phong cách", 2),
        # Cấp 2 Nữ
        ("DM000005", "Áo nữ", "DM000002", "Các kiểu áo nữ duyên dáng, hiện đại", 1),
        ("DM000006", "Quần nữ", "DM000002", "Các mẫu quần nữ tôn dáng", 2),
        ("DM000007", "Đầm & Chân váy", "DM000002", "Đầm liền và chân váy quý phái, thanh lịch", 3),
        # Cấp 3 Lá Nam - Áo
        ("DM000008", "Áo gile & Vest nam", "DM000003", "Áo gile, vest sang trọng phong cách quý ông", 1),
        ("DM000009", "Áo len & Cardigan nam", "DM000003", "Áo len mỏng, cardigan thu đông ấm áp", 2),
        ("DM000010", "Áo nỉ & Hoodie nam", "DM000003", "Áo nỉ chui đầu, hoodie trẻ trung năng động", 3),
        ("DM000011", "Áo polo nam", "DM000003", "Áo polo thể thao, công sở co giãn cao cấp", 4),
        ("DM000012", "Áo sơ mi nam", "DM000003", "Áo sơ mi công sở, sơ mi dạo phố lịch thiệp", 5),
        ("DM000013", "Áo thun nam", "DM000003", "Áo thun cổ tròn, cổ tim cotton thoáng mát", 6),
        # Cấp 3 Lá Nam - Quần
        ("DM000014", "Quần jeans nam", "DM000004", "Quần jeans nam slimfit, ống đứng cá tính", 1),
        ("DM000015", "Quần jogger & Thể thao nam", "DM000004", "Quần jogger bo gấu, thể thao đường phố", 2),
        ("DM000016", "Quần kaki & Chinos nam", "DM000004", "Quần kaki cao cấp, chinos thanh lịch", 3),
        ("DM000017", "Quần short nam", "DM000004", "Quần short dạo phố, thể thao mùa hè", 4),
        ("DM000018", "Quần tây âu nam", "DM000004", "Quần âu may đo chuẩn form công sở", 5),
        # Cấp 3 Lá Nữ - Áo
        ("DM000019", "Áo gile & Croptop nữ", "DM000005", "Áo gile dạ, croptop phong cách sành điệu", 1),
        ("DM000020", "Áo kiểu & Blouses nữ", "DM000005", "Áo kiểu cách điệu, blouse thắt nơ nữ tính", 2),
        ("DM000021", "Áo len & Cardigan nữ", "DM000005", "Áo len dệt kim, cardigan tiểu thư", 3),
        ("DM000022", "Áo nỉ nữ", "DM000005", "Áo nỉ sweater form rộng cá tính", 4),
        ("DM000023", "Áo sơ mi nữ", "DM000005", "Áo sơ mi lụa, voan thanh lịch công sở", 5),
        ("DM000024", "Áo thun & Polo nữ", "DM000005", "Áo thun và polo nữ phom ôm tôn dáng", 6),
        # Cấp 3 Lá Nữ - Quần
        ("DM000025", "Quần jeans nữ", "DM000006", "Quần jeans cạp cao hack dáng thon dài", 1),
        ("DM000026", "Quần nỉ nữ", "DM000006", "Quần nỉ ống rộng thoải mái mặc nhà, dạo phố", 2),
        ("DM000027", "Quần short nữ", "DM000006", "Quần short jeans, kaki cá tính mùa hè", 3),
        ("DM000028", "Quần tây công sở nữ", "DM000006", "Quần âu xếp ly, ống suông công sở hiện đại", 4),
        # Cấp 3 Lá Nữ - Đầm & Váy
        ("DM000029", "Áo dài truyền thống & Cách tân", "DM000007", "Áo dài lụa thêu tơ tằm thanh tao", 1),
        ("DM000030", "Chân váy nữ", "DM000007", "Chân váy chữ A, xếp ly, đuôi cá tôn dáng", 2),
        ("DM000031", "Đầm liền thân", "DM000007", "Đầm xòe, đầm suông, đầm dạ hội quý phái", 3),
    ]

    # Map category path -> category ID
    CAT_MAP = {
        "Male/AoNam/AoGile_Vest": "DM000008",
        "Male/AoNam/AoLen_Cardigan": "DM000009",
        "Male/AoNam/AoNi_Hoodie": "DM000010",
        "Male/AoNam/AoPolo": "DM000011",
        "Male/AoNam/AoSoMi": "DM000012",
        "Male/AoNam/AoThun": "DM000013",
        "Male/QuanNam/QuanJeans": "DM000014",
        "Male/QuanNam/QuanJogger_Sport": "DM000015",
        "Male/QuanNam/QuanKaki_Chinos": "DM000016",
        "Male/QuanNam/QuanShort": "DM000017",
        "Male/QuanNam/QuanTay_Au": "DM000018",
        "Female/AoNu/AoGile_Croptop": "DM000019",
        "Female/AoNu/AoKieu_Blouses": "DM000020",
        "Female/AoNu/AoLen_Cardigan": "DM000021",
        "Female/AoNu/AoNi": "DM000022",
        "Female/AoNu/AoSoMi": "DM000023",
        "Female/AoNu/AoThun_Polo": "DM000024",
        "Female/QuanNu/QuanJeans": "DM000025",
        "Female/QuanNu/QuanNi": "DM000026",
        "Female/QuanNu/QuanShort": "DM000027",
        "Female/QuanNu/QuanTay_CongSo": "DM000028",
        "Female/Dam_Vay/AoDai": "DM000029",
        "Female/Dam_Vay/ChanVay": "DM000030",
        "Female/Dam_Vay/DamLien": "DM000031",
    }

    # ---------------------------------------------------------
    # BẢNG 6, 7, 8: SanPham, BienTheSanPham, HinhAnhSanPham
    # ---------------------------------------------------------
    md_files = sorted(list(garments_dir.rglob("product_analysis.md")))
    print(f"Found {len(md_files)} products to parse.")

    san_pham = []
    bien_the = []
    hinh_anh = []
    global_used_skus = set()

    bt_idx = 1
    ha_idx = 1

    for sp_idx, md_path in enumerate(md_files, 1):
        sp_id = f"SP{sp_idx:08d}"
        pdir = md_path.parent
        rel_cat = "/".join(md_path.relative_to(garments_dir).parts[:-2])
        dm_id = CAT_MAP.get(rel_cat, "DM000011")

        # Đọc description.txt
        desc_file = pdir / "description.txt"
        ten_sp = pdir.name
        mo_ta = ""
        chat_lieu = "Vải cao cấp thoáng khí, chống nhăn"
        if desc_file.exists():
            lines = desc_file.read_text(encoding="utf-8").splitlines()
            if lines and lines[0].strip():
                raw_title = lines[0].strip()
                # Clean title
                raw_title = re.sub(r"\s*-\s*YOFASO\s*SHOP.*$", "", raw_title, flags=re.IGNORECASE)
                ten_sp = raw_title.strip()
            mo_ta = "\n".join(lines[1:]).strip()

        # Đọc additionalInformation.txt
        add_file = pdir / "additionalInformation.txt"
        kieu_dang = "SUONG_VUA"
        loai_cap = "KHONG_AP_DUNG"
        chieu_dai = "AO_TIEU_CHUAN"
        hoa_tiet = "VAI_TRON"

        if add_file.exists():
            add_txt = add_file.read_text(encoding="utf-8")
            if "OM_SAT" in add_txt: kieu_dang = "OM_SAT"
            elif "RONG_OVERSIZE" in add_txt: kieu_dang = "RONG_OVERSIZE"
            elif "SUONG_VUA" in add_txt: kieu_dang = "SUONG_VUA"

            if "CAP_CAO" in add_txt: loai_cap = "CAP_CAO"
            elif "CAP_VUA" in add_txt: loai_cap = "CAP_VUA"
            elif "CAP_TRE" in add_txt: loai_cap = "CAP_TRE"

            if "AO_CROPTOP" in add_txt: chieu_dai = "AO_CROPTOP"
            elif "AO_DANG_DAI" in add_txt: chieu_dai = "AO_DANG_DAI"
            elif "NGAN_TREN_GOI" in add_txt: chieu_dai = "NGAN_TREN_GOI"
            elif "LUNG_NGANG_GOI" in add_txt: chieu_dai = "LUNG_NGANG_GOI"
            elif "MIDI_QUA_GOI" in add_txt: chieu_dai = "MIDI_QUA_GOI"
            elif "DAI_CHAM_GOT" in add_txt: chieu_dai = "DAI_CHAM_GOT"

            if "HOA_VAN_NHO" in add_txt: hoa_tiet = "HOA_VAN_NHO"
            elif "HOA_TO_BAN" in add_txt: hoa_tiet = "HOA_TO_BAN"
            elif "KE_SOC_TAM" in add_txt: hoa_tiet = "KE_SOC_TAM"
            elif "KE_CARO" in add_txt: hoa_tiet = "KE_CARO"
            elif "IN_HINH" in add_txt: hoa_tiet = "IN_HINH"

            m_mat = re.search(r"Chất liệu\s*:\s*([^\n\r]+)", add_txt)
            if m_mat:
                chat_lieu = m_mat.group(1).strip()

        # Đọc product_analysis.md
        md_content = md_path.read_text(encoding="utf-8")
        struct_data = {}
        m_json = re.search(r"```json\s*(\{.*?\})\s*```", md_content, re.DOTALL)
        if m_json:
            try:
                struct_data = json.loads(m_json.group(1))
            except Exception:
                pass

        db_map = struct_data.get("database_mapping", {})

        slug_match = re.search(r'catalog(?:-processing)?/products/([^/\s]+)/', md_content)
        if slug_match:
            slug = slug_match.group(1).strip()
        else:
            slug = pdir.name

        vton_cat = db_map.get("PhanLoaiVTON") or struct_data.get("vton_category")
        if not vton_cat:
            if "lower_body" in md_content or "Quan" in rel_cat or "ChanVay" in rel_cat:
                vton_cat = "lower_body"
            elif "dress" in md_content or "Dam" in rel_cat or "AoDai" in rel_cat:
                vton_cat = "dress"
            else:
                vton_cat = "upper_body"

        tieu_loai = db_map.get("TieuLoaiTrangPhuc") or struct_data.get("resolved_label") or struct_data.get("category") or "short_sleeved_shirt"

        mask_path = db_map.get("DuongDanMatNaVungThayDo")
        if not mask_path:
            mask_match = re.search(r'"DuongDanMatNaVungThayDo"\s*:\s*"([^"]+)"', md_content)
            if mask_match:
                mask_path = mask_match.group(1).strip()
            else:
                mask_path = f"catalog-processing/products/{slug}/{slug}--img01--v01--mask.png"

        raw_palette = db_map.get("BangMauTrichXuat")
        if raw_palette is not None:
            palette_json = json.dumps(raw_palette, ensure_ascii=False)
        else:
            palette_json = "[]"

        # Xử lý chất liệu nếu vượt quá 100 ký tự (VARCHAR(100))
        clean_chat_lieu = chat_lieu.strip()
        if len(clean_chat_lieu) > 100:
            clean_chat_lieu = clean_chat_lieu.replace(" / Lớp lót trong 100% Faux Sherpa Fleece (Lông cừu nhân tạo)", " / Lót 100% Lông cừu nhân tạo")
            if len(clean_chat_lieu) > 100:
                clean_chat_lieu = clean_chat_lieu[:97] + "..."

        san_pham.append({
            "MaSanPham": sp_id,
            "MaDanhMuc": dm_id,
            "TenSanPham": ten_sp,
            "MoTaChiTiet": mo_ta if mo_ta else f"Sản phẩm {ten_sp} thời trang cao cấp của YoFaSo Shop.",
            "ChatLieu": clean_chat_lieu,
            "KieuDang": kieu_dang,
            "LoaiCapQuan": loai_cap,
            "ChieuDaiTrangPhuc": chieu_dai,
            "LoaiHoaTiet": hoa_tiet,
            "PhanLoaiVTON": vton_cat,
            "TieuLoaiTrangPhuc": tieu_loai,
            "BangMauTrichXuat": palette_json,
            "DuongDanMatNaVungThayDo": mask_path,
            "TrangThaiSanPham": "DANG_KINH_DOANH",
            "slug": slug,
            "pdir": pdir
        })

        # -----------------------------------------------------
        # Tạo Biến thể & Hình ảnh cho sản phẩm
        # -----------------------------------------------------
        # Tìm danh sách ảnh người mẫu (--ref*--original.png) và ảnh sản phẩm trên móc treo (--img*--original.png)
        ref_images = sorted([f for f in pdir.iterdir() if f.is_file() and "--ref" in f.name and "--original.png" in f.name])
        orig_images = sorted([f for f in pdir.iterdir() if f.is_file() and "--img" in f.name and "--original.png" in f.name])
        if not orig_images and not ref_images:
            orig_images = sorted([f for f in pdir.glob("*.png") if f.is_file()])

        # Lấy thông tin màu sắc từ product_analysis table
        variant_colors = []
        for line in md_content.splitlines():
            m_col = re.search(r'\|\s*\d+\s*\|.*?\|\s*([^|#]+(#[0-9A-Fa-f]{6})[^|]*)\|', line)
            if m_col:
                raw_col = m_col.group(1).strip()
                # Parse color name & hex, strip backticks and quotes
                c_name = re.sub(r'[`*"\']', '', raw_col.split("#")[0]).strip()
                hexes = re.findall(r'#[0-9A-Fa-f]{6}', raw_col)
                hex_str = ",".join(hexes) if hexes else "#000000"
                variant_colors.append((c_name or "Mặc định", hex_str))

        if not variant_colors:
            variant_colors = [("Màu tiêu chuẩn", "#1A1A1A")]

        # Tạo ảnh sản phẩm (HinhAnhSanPham):
        # QUY TẮC:
        # Nếu có ảnh người mẫu (--ref*--original.png):
        #   - ref01: ThuTuHienThi = 1, LaAnhDaiDien = 1 (Làm thumbnail UI danh mục), DuongDanAnhBocNen = NULL
        #   - ref02, ref03...: ThuTuHienThi = 2, 3..., LaAnhDaiDien = 0, DuongDanAnhBocNen = NULL
        #   - Các ảnh chụp móc treo (--img*): nối tiếp ThuTuHienThi, LaAnhDaiDien = 0, DuongDanAnhBocNen = ...--cutout.png
        # Nếu không có ảnh ref:
        #   - img01: ThuTuHienThi = 1, LaAnhDaiDien = 1
        #   - img02...: ThuTuHienThi = 2..., LaAnhDaiDien = 0
        img_seq = 1
        if ref_images:
            for r_idx, ref_f in enumerate(ref_images):
                ha_id = f"HA{ha_idx:08d}"
                ha_idx += 1
                col_name, col_hex = variant_colors[r_idx % len(variant_colors)]
                hinh_anh.append({
                    "MaHinhAnh": ha_id,
                    "MaSanPham": sp_id,
                    "DuongDanAnhGoc": f"catalog/products/{slug}/{ref_f.name}",
                    "DuongDanAnhBocNen": None,
                    "MaMauSac": col_hex,
                    "LaAnhDaiDien": 1 if r_idx == 0 else 0,
                    "ThuTuHienThi": img_seq
                })
                img_seq += 1

        for img_num, img_f in enumerate(orig_images, 1):
            ha_id = f"HA{ha_idx:08d}"
            ha_idx += 1
            col_name, col_hex = variant_colors[(img_num - 1) % len(variant_colors)]
            cutout_name = img_f.name.replace("--original.png", "--cutout.png")
            is_thumb = 1 if (not ref_images and img_num == 1) else 0
            hinh_anh.append({
                "MaHinhAnh": ha_id,
                "MaSanPham": sp_id,
                "DuongDanAnhGoc": f"catalog/products/{slug}/{img_f.name}",
                "DuongDanAnhBocNen": f"catalog/products/{slug}/{cutout_name}",
                "MaMauSac": col_hex,
                "LaAnhDaiDien": is_thumb,
                "ThuTuHienThi": img_seq
            })
            img_seq += 1

        # Tạo biến thể SKU (BienTheSanPham): mỗi sản phẩm có 2-4 biến thể (màu + size)
        # Size mapping: Nam: M, L, XL | Nữ: S, M, L
        available_sizes = ["S", "M", "L", "XL"] if "Male" in str(pdir) else ["S", "M", "L", "XL"]

        # Giá cơ sở tùy loại
        base_price = 350000
        if "Vest" in rel_cat or "Dam" in rel_cat or "AoDai" in rel_cat:
            base_price = 690000
        elif "Jeans" in rel_cat or "Tay" in rel_cat:
            base_price = 450000
        elif "Polo" in rel_cat or "SoMi" in rel_cat:
            base_price = 390000
        elif "Thun" in rel_cat or "Short" in rel_cat:
            base_price = 250000

        # Tạo biến thể kết hợp màu và size, đảm bảo tuyệt đối không trùng (KichCo, TenMau)
        var_count = min(len(variant_colors), 3)
        sp_used_combos = set()

        for v_i in range(var_count):
            c_name, c_hex = variant_colors[v_i]
            c_name = re.sub(r'[`*"\']', '', c_name).strip() or "Màu tiêu chuẩn"

            # Tìm size chưa bị trùng với màu này trong cùng sản phẩm
            target_size = None
            for s_cand in available_sizes:
                if (c_name, s_cand) not in sp_used_combos:
                    target_size = s_cand
                    break

            if not target_size:
                # Nếu tất cả size đã được dùng với tên màu này, đổi tên màu để phân biệt
                c_name = f"{c_name} Mẫu {v_i+1}"
                target_size = available_sizes[v_i % len(available_sizes)]

            sp_used_combos.add((c_name, target_size))

            bt_id = f"BT{bt_idx:08d}"
            bt_idx += 1

            clean_col = re.sub(r'[^a-zA-Z0-9]', '', c_name)[:6].upper() or "DEF"
            slug_short = slug[:10].replace("-", "").upper()

            # Sinh SKU đảm bảo duy nhất 100% trên toàn hệ thống
            base_sku = f"SKU-{slug_short}-{clean_col}-{target_size}-{sp_idx:03d}"
            sku = base_sku
            sku_counter = 1
            while sku in global_used_skus:
                sku_counter += 1
                sku = f"{base_sku}-V{sku_counter:02d}"
            global_used_skus.add(sku)

            gia_ban = base_price + (v_i * 20000)
            gia_km = (gia_ban - 40000) if (sp_idx % 3 == 0) else None
            ton_kho = 25 + ((sp_idx * 7 + v_i * 13) % 45)
            ke = f"KE-{'M' if 'Male' in str(pdir) else 'F'}{(sp_idx % 20)+1:02d}"

            bien_the.append({
                "MaBienThe": bt_id,
                "MaSanPham": sp_id,
                "MaQuanLyHangHoa": sku,
                "KichCo": target_size,
                "TenMau": c_name[:50],
                "MaMauSac": c_hex[:100],
                "GiaNiemYet": gia_ban,
                "GiaKhuyenMai": gia_km,
                "SoLuongTonKho": ton_kho,
                "ViTriGiaKe": ke,
                "TrangThai": "DANG_BAN"
            })

    print(f"Generated {len(san_pham)} SanPham, {len(bien_the)} BienThe, {len(hinh_anh)} HinhAnh.")

    # ---------------------------------------------------------
    # BẢNG 9 & 10: PhienThuDoAo & GoiYPhoiDo (5 phiên mẫu chuẩn)
    # ---------------------------------------------------------
    phien_thu_do = []
    goi_y_phoi_do = []
    
    # 5 phiên thử đồ ảo VTON chuẩn mực khớp chính xác sản phẩm thực tế
    sample_tryons = [
        # TD00000001: Khách Mai Anh thử Áo polo nữ cổ sen (SP00000025, BT00000049)
        ("TD00000001", "ND00000002", "BT00000049", "HALF_BODY", 
         "tryon/TD00000001/35220f68--person.png", "tryon/TD00000001/0dab1284--result.png",
         92, 95, 90, 90, 0, "1/3-2/3", "RECOMMENDED", 
         "Áo polo cổ sen phom ôm vừa vặn, màu đen phối viền thanh lịch, tôn dáng cổ thon thả và làn da sáng."),
        # TD00000002: Khách Hoàng Long thử Áo kiểu cộc tay nữ tặng bạn gái (SP00000008, BT00000016)
        ("TD00000002", "ND00000003", "BT00000016", "HALF_BODY", 
         "tryon/TD00000002/f981c10e--person.png", "tryon/TD00000002/a1fe483d--result.png",
         88, 85, 90, 90, 0, "2/3-1/3", "RECOMMENDED", 
         "Form áo kiểu cộc tay thanh thoát, gam màu đen sang trọng, rất thích hợp diện đi làm hoặc gặp gỡ đối tác."),
        # TD00000003: Khách Quỳnh Chi chọn Quần jeans loose nam tặng bạn trai (SP00000097, BT00000217)
        ("TD00000003", "ND00000004", "BT00000217", "FULL_BODY", 
         "tryon/TD00000003/e7a30bdd--person.png", "tryon/TD00000003/f18df779--result.png",
         96, 98, 95, 95, 0, "1/3-2/3", "RECOMMENDED", 
         "Form quần jeans ống suông Loose thoải mái, màu xanh denim trẻ trung, hack chiều cao và tạo nét phóng khoáng."),
        # TD00000004: Khách Minh Quân thử Áo thun nam dệt liền (SP00000093, BT00000207)
        ("TD00000004", "ND00000005", "BT00000207", "HALF_BODY", 
         "tryon/TD00000004/adec6433--person.png", "tryon/TD00000004/8e456c25--result.png",
         84, 80, 85, 90, 0, "1/3-2/3", "ACCEPTABLE", 
         "Chất liệu dệt liền mềm mại thoáng khí, tone màu nâu thời thượng, phom suông che khuyết điểm cơ thể tối ưu."),
        # TD00000005: Khách Admin / Vãng lai thử Đầm xuân hè dáng A (SP00000041, BT00000089)
        ("TD00000005", "ND00000001", "BT00000089", "FULL_BODY", 
         "tryon/TD00000005/b9e36fe2--person.png", "tryon/TD00000005/41a7a05d--result.png",
         90, 92, 90, 88, 0, "1/3-2/3", "RECOMMENDED", 
         "Đầm dáng A nơ cổ thanh nhã, tone màu be quý phái, chiết eo tôn vóc dáng đồng hồ cát và hack chiều cao hoàn hảo."),
    ]
    
    for s in sample_tryons:
        sid, uid, btid, frame, p_key, r_key, s_total, s_prop, s_col, s_pat, p_pen, rule, rank, cmt = s
        phien_thu_do.append({
            "MaPhienThuDo": sid,
            "MaNguoiDung": uid,
            "MaBienThe": btid,
            "LoaiKhungAnhNguoi": frame,
            "DuongDanAnhNguoiDung": p_key,
            "DuongDanAnhKetQuaVTON": r_key,
            "TrangThaiPhien": "THANH_CONG",
            "ThoiGianXuLy": 8.45,
            "ThongBaoLoi": None,
            "DiemTongThe": s_total,
            "DiemTiLeVocDang": s_prop,
            "DiemHoaHopMauSac": s_col,
            "DiemCanBangHoaTiet": s_pat,
            "DiemTruPhat": p_pen,
            "QuyTacTiLeApDung": rule,
            "XepHangThamMy": rank,
            "LoiNhanXet": cmt
        })
        
    # 10 Gợi ý phối đồ bán kèm Smart Cross-selling đối ứng chuẩn mực từ allScoring.md & ScoringTable.xlsx
    sample_suggestions = [
        # TD00000001 (Áo polo nữ đen BT00000049): Gợi ý chân váy xếp ly chữ A be (BT00000079) & Chân váy denim (BT00000075)
        ("TD00000001", "BT00000079", 96, "Phối áo polo đen cùng chân váy chữ A màu be tạo tỉ lệ vàng 1/3-2/3, tương phản sáng tối kinh điển giúp kéo dài đôi chân và tôn eo thon.", "DA_THEM_GIO_HANG"),
        ("TD00000001", "BT00000075", 93, "Sự kết hợp năng động giữa áo polo cổ sen và chân váy denim xếp ly mang lại diện mạo trẻ trung, hiện đại cho buổi dạo phố.", "CHUA_XEM"),
        # TD00000002 (Áo kiểu cộc tay đen BT00000016): Gợi ý chân váy lưng thun nâu (BT00000083) & Chân váy xếp ly nylon navy (BT00000076)
        ("TD00000002", "BT00000083", 95, "Quy tắc phối màu trung tính tương đồng (Đen - Nâu) kết hợp phom ôm trên suông dưới tạo tổng thể sang trọng và thanh lịch.", "CHUA_XEM"),
        ("TD00000002", "BT00000076", 92, "Phom áo cộc tay gọn gàng phối cùng chân váy xếp ly rũ nhẹ giúp cân bằng hình khối cơ thể theo tỉ lệ vàng thời trang.", "DA_THU_THEM"),
        # TD00000003 (Quần jeans loose nam BT00000217): Gợi ý Áo polo kẻ trắng (BT00000178) & Áo sơ mi cộc tay kem (BT00000189)
        ("TD00000003", "BT00000178", 95, "Quần jeans loose phối cùng áo polo trắng bo viền tạo độ tương phản sáng tối sắc sảo, nâng tầm phong cách smart-casual lịch lãm.", "CHUA_XEM"),
        ("TD00000003", "BT00000189", 93, "Bộ đôi áo sơ mi cộc tay và quần jeans rộng phóng khoáng mang đậm cảm hứng dạo phố mùa hè, cân bằng hoàn hảo cấu trúc vóc dáng.", "CHUA_XEM"),
        # TD00000004 (Áo thun dệt liền nam nâu BT00000207): Gợi ý Quần jeans Barrel trắng (BT00000220) & Quần jeans baggy trắng (BT00000223)
        ("TD00000004", "BT00000220", 96, "Tông màu nâu đất kết hợp quần jeans trắng ngà tạo sự hòa hợp màu sắc êm dịu, vóc dáng 1/3-2/3 giúp đôi chân trông dài và thẳng hơn.", "CHUA_XEM"),
        ("TD00000004", "BT00000223", 92, "Áo thun suông phối cùng quần jeans baggy tạo nên set đồ streetwear năng động, thoải mái tối đa khi di chuyển.", "CHUA_XEM"),
        # TD00000005 (Đầm xuân hè dáng A be BT00000089): Gợi ý Áo gile tuytxi đen khoác layer (BT00000001) & Áo sơ mi croptop trắng khoác ngoài (BT00000006)
        ("TD00000005", "BT00000001", 95, "Phối layer áo gile đen khoác ngoài đầm be tạo hiệu ứng chiều sâu đa tầng, tôn vòng eo và định hình phong cách nữ tổng tài hiện đại.", "CHUA_XEM"),
        ("TD00000005", "BT00000006", 92, "Thiết kế croptop khoác ngoài tạo điểm nhấn cúp ngực, tôn vinh nét nữ tính thanh tao và bảo vệ làn da dưới nắng hè.", "CHUA_XEM"),
    ]
    
    for gy_idx, g in enumerate(sample_suggestions, 1):
        goi_y_phoi_do.append({
            "MaGoiY": f"GY{gy_idx:08d}",
            "MaPhienThuDo": g[0],
            "MaBienTheGoiY": g[1],
            "DiemSauKhiPhoiKem": g[2],
            "LyDoGoiY": g[3],
            "TrangThaiTuongTac": g[4]
        })

    # ---------------------------------------------------------
    # BẢNG 11: KhuyenMai (6 voucher giá trị cao)
    # ---------------------------------------------------------
    khuyen_mai = [
        ("KM000001", "YOFASO10", "Ưu đãi 10% đơn hàng đầu tiên", "PHAN_TRAM", 10.00, 300000.00, 100000.00, 500, 48, "2026-01-01 00:00:00", "2026-12-31 23:59:59", "DANG_DIEN_RA"),
        ("KM000002", "FREESHIP50", "Miễn phí vận chuyển toàn quốc", "MIEN_PHI_VAN_CHUYEN", 30000.00, 250000.00, 30000.00, 1000, 126, "2026-01-01 00:00:00", "2026-12-31 23:59:59", "DANG_DIEN_RA"),
        ("KM000003", "GIAM50K", "Giảm ngay 50.000đ cho đơn từ 500.000đ", "TIEN_MAT_TRUC_TIEP", 50000.00, 500000.00, 50000.00, 300, 89, "2026-01-01 00:00:00", "2026-12-31 23:59:59", "DANG_DIEN_RA"),
        ("KM000004", "VIPGOLD15", "Đặc quyền tri ân hội viên Vàng & Kim Cương", "PHAN_TRAM", 15.00, 600000.00, 200000.00, 200, 35, "2026-01-01 00:00:00", "2026-12-31 23:59:59", "DANG_DIEN_RA"),
        ("KM000005", "SINHNHATYOFASO", "Quà tặng sinh nhật tháng vàng", "PHAN_TRAM", 20.00, 400000.00, 250000.00, 100, 12, "2026-05-01 00:00:00", "2026-05-31 23:59:59", "TAM_DUNG"),
        ("KM000006", "HEVTON2026", "Lễ hội thử đồ ảo mùa hè VTON", "TIEN_MAT_TRUC_TIEP", 70000.00, 700000.00, 70000.00, 400, 95, "2026-06-01 00:00:00", "2026-08-31 23:59:59", "DANG_DIEN_RA"),
    ]

    # ---------------------------------------------------------
    # BẢNG 12 & 13: GioHang & ChiTietGioHang (5 giỏ hàng khách)
    # ---------------------------------------------------------
    gio_hang = []
    chi_tiet_gio_hang = []
    ctgh_idx = 1
    
    for c_i in range(2, 7):
        uid = f"ND{c_i:08d}"
        gh_id = f"GH{c_i-1:08d}"
        gio_hang.append((gh_id, uid))
        
        # 1-2 sản phẩm trong giỏ
        chi_tiet_gio_hang.append({
            "MaChiTietGioHang": f"CTGH{ctgh_idx:06d}",
            "MaGioHang": gh_id,
            "MaBienThe": bien_the[(c_i * 7) % len(bien_the)]["MaBienThe"],
            "SoLuong": 1
        })
        ctgh_idx += 1
        if c_i % 2 == 0:
            chi_tiet_gio_hang.append({
                "MaChiTietGioHang": f"CTGH{ctgh_idx:06d}",
                "MaGioHang": gh_id,
                "MaBienThe": bien_the[(c_i * 11) % len(bien_the)]["MaBienThe"],
                "SoLuong": 2
            })
            ctgh_idx += 1

    # ---------------------------------------------------------
    # BẢNG 14 & 15: DonHang & ChiTietDonHang (12 đơn hàng đa kênh)
    # ---------------------------------------------------------
    # Cấu hình 18 dòng chi tiết đơn hàng tương ứng 12 đơn:
    # (MaDonHang, MaBienThe, SoLuong, MaPhienThuDo)
    # Khớp chính xác với sản phẩm đã thử đồ VTON (TD00000001 -> BT00000049, TD00000002 -> BT00000016, TD00000005 -> BT00000089)
    order_items_blueprint = [
        # DH00000001: Mai Anh mua online COD, CÓ VTON O2O (từ TD00000001 thử BT00000049) + Mua thêm BT00000018
        ("DH00000001", "BT00000049", 1, "TD00000001"),
        ("DH00000001", "BT00000018", 1, None),
        # DH00000002: Hoàng Long mua online VietQR, CÓ VTON O2O (từ TD00000002 thử BT00000016)
        ("DH00000002", "BT00000016", 1, "TD00000002"),
        # DH00000003: Quỳnh Chi mua tại quầy POS Lan Phương, tiền mặt
        ("DH00000003", "BT00000016", 1, None),
        ("DH00000003", "BT00000032", 1, None),
        # DH00000004: Khách vãng lai POS Đức Thịnh, VietQR
        ("DH00000004", "BT00000021", 1, None),
        # DH00000005: Quỳnh Chi mua online, đang giao hàng
        ("DH00000005", "BT00000026", 1, None),
        ("DH00000005", "BT00000046", 1, None),
        # DH00000006: Minh Quân mua online, đang giao hàng
        ("DH00000006", "BT00000031", 1, None),
        # DH00000007: Thu Thảo mua online, đang đóng gói kho
        ("DH00000007", "BT00000036", 1, None),
        ("DH00000007", "BT00000060", 1, None),
        # DH00000008: Mai Anh mua online, chờ xác nhận
        ("DH00000008", "BT00000041", 1, None),
        # DH00000009: Hoàng Long POS Bích Ngọc, tiền mặt
        ("DH00000009", "BT00000046", 1, None),
        ("DH00000009", "BT00000074", 1, None),
        # DH00000010: Khách vãng lai mua tại quầy POS sau khi thử VTON (từ TD00000005 thử BT00000089)
        ("DH00000010", "BT00000089", 1, "TD00000005"),
        # DH00000011: Minh Quân đặt online bị hủy do đổi ý
        ("DH00000011", "BT00000056", 1, None),
        ("DH00000011", "BT00000088", 1, None),
        # DH00000012: Thu Thảo mua online tuần trước, đã hoàn tất
        ("DH00000012", "BT00000061", 1, None),
    ]

    # Map tra cứu nhanh BienThe & SanPham
    bt_dict = {b["MaBienThe"]: b for b in bien_the}
    sp_dict = {s["MaSanPham"]: s for s in san_pham}

    chi_tiet_don_hang = []
    ctdh_idx = 1
    for item in order_items_blueprint:
        dh_id, bt_id, qty, vton_ref = item
        bt_obj = bt_dict[bt_id]
        sp_obj = sp_dict[bt_obj["MaSanPham"]]
        
        chi_tiet_don_hang.append({
            "MaChiTietDonHang": f"CTDH{ctdh_idx:06d}",
            "MaDonHang": dh_id,
            "MaBienThe": bt_id,
            "TenSanPhamLucDatHang": sp_obj["TenSanPham"],
            "TenMauLucDatHang": bt_obj["TenMau"],
            "KichCoLucDatHang": bt_obj["KichCo"],
            "SoLuong": qty,
            "DonGiaLucDatHang": bt_obj["GiaNiemYet"],
            "MaPhienThuDo": vton_ref
        })
        ctdh_idx += 1

    don_hang = []
    orders_setup = [
        # (dh_id, uid, channel, cashier, shipper, addr, receiver, phone, full_addr, discount, ship_fee, p_method, p_status, ext_tx, o_status, cod_amt, km_id)
        ("DH00000001", "ND00000002", "ONLINE_APP", None, "ND00000013", "DC00000002", "Trần Thị Mai Anh", "0912345678", "Số 18, Ngõ 45 Hoàng Hoa Thám, Hà Nội", 30000.00, 30000.00, "COD_TIEN_MAT_KHI_NHAN", "DA_THANH_TOAN", None, "HOAN_TAT", 780000.00, "KM000002"),
        ("DH00000002", "ND00000003", "ONLINE_APP", None, "ND00000014", "DC00000003", "Lê Hoàng Long", "0923456789", "Tòa Landmark 81, TP. Hồ Chí Minh", 35000.00, 0.00, "VIETQR_CHUYEN_KHOAN_ONLINE", "DA_THANH_TOAN", "FT261028001", "HOAN_TAT", 0.00, "KM000001"),
        ("DH00000003", "ND00000004", "TAI_QUAY_POS", "ND00000007", None, None, "Phạm Quỳnh Chi", "0934567890", None, 0.00, 0.00, "TIEN_MAT_TAI_QUAY", "DA_THANH_TOAN", None, "HOAN_TAT", 0.00, None),
        ("DH00000004", None, "TAI_QUAY_POS", "ND00000008", None, None, "Khách vãng lai", "0900000001", None, 0.00, 0.00, "VIETQR_TAI_QUAY", "DA_THANH_TOAN", "FT261028002", "HOAN_TAT", 0.00, None),
        ("DH00000005", "ND00000004", "ONLINE_APP", None, "ND00000013", "DC00000004", "Phạm Quỳnh Chi", "0934567890", "Số 88 Lê Duẩn, Đà Nẵng", 76000.00, 25000.00, "COD_TIEN_MAT_KHI_NHAN", "CHUA_THANH_TOAN", None, "DANG_GIAO_HANG", 709000.00, "KM000001"),
        ("DH00000006", "ND00000005", "ONLINE_APP", None, "ND00000014", "DC00000005", "Vũ Minh Quân", "0945678901", "Số 250 Lạch Tray, Hải Phòng", 0.00, 30000.00, "VIETQR_CHUYEN_KHOAN_ONLINE", "DA_THANH_TOAN", "FT261028003", "DANG_GIAO_HANG", 0.00, None),
        ("DH00000007", "ND00000006", "ONLINE_APP", None, None, "DC00000006", "Hoàng Thu Thảo", "0956789012", "Số 34 Đường 30/4, Cần Thơ", 30000.00, 0.00, "VIETQR_CHUYEN_KHOAN_ONLINE", "DA_THANH_TOAN", "FT261028004", "DANG_DONG_GOI", 0.00, "KM000002"),
        ("DH00000008", "ND00000002", "ONLINE_APP", None, None, "DC00000002", "Trần Thị Mai Anh", "0912345678", "Số 18, Ngõ 45 Hoàng Hoa Thám, Hà Nội", 0.00, 30000.00, "COD_TIEN_MAT_KHI_NHAN", "CHUA_THANH_TOAN", None, "CHO_XAC_NHAN", 420000.00, None),
        ("DH00000009", "ND00000003", "TAI_QUAY_POS", "ND00000009", None, None, "Lê Hoàng Long", "0923456789", None, 100000.00, 0.00, "TIEN_MAT_TAI_QUAY", "DA_THANH_TOAN", None, "HOAN_TAT", 0.00, "KM000001"),
        ("DH00000010", None, "TAI_QUAY_POS", "ND00000007", None, None, "Khách mua sau khi thử VTON", "0911223344", None, 0.00, 0.00, "VIETQR_TAI_QUAY", "DA_THANH_TOAN", "FT261028005", "HOAN_TAT", 0.00, None),
        ("DH00000011", "ND00000005", "ONLINE_APP", None, None, "DC00000005", "Vũ Minh Quân", "0945678901", "Số 250 Lạch Tray, Hải Phòng", 0.00, 30000.00, "COD_TIEN_MAT_KHI_NHAN", "CHUA_THANH_TOAN", None, "DA_HUY", 0.00, None),
        ("DH00000012", "ND00000006", "ONLINE_APP", None, "ND00000015", "DC00000006", "Hoàng Thu Thảo", "0956789012", "Số 34 Đường 30/4, Cần Thơ", 30000.00, 0.00, "VIETQR_CHUYEN_KHOAN_ONLINE", "DA_THANH_TOAN", "FT261028006", "HOAN_TAT", 0.00, "KM000002"),
    ]
    
    for o in orders_setup:
        dh_id = o[0]
        don_hang.append({
            "MaDonHang": dh_id,
            "MaNguoiDung": o[1],
            "KenhBanHang": o[2],
            "MaThuNgan": o[3],
            "MaNguoiGiaoHang": o[4],
            "MaDiaChiGiaoHang": o[5],
            "TenNguoiNhan": o[6],
            "SoDienThoaiNhan": o[7],
            "DiaChiGiaoChiTiet": o[8],
            "TienGiamGia": o[9],
            "PhiVanChuyen": o[10],
            "PhuongThucThanhToan": o[11],
            "TrangThaiThanhToan": o[12],
            "MaGiaoDichNgoai": o[13],
            "ThoiGianThanhToan": "2026-10-28 10:15:00" if o[12] == "DA_THANH_TOAN" else None,
            "TrangThaiDonHang": o[14],
            "SoTienThuHo": o[15],
            "SoLanGiao": 1 if o[14] in ("DANG_GIAO_HANG", "HOAN_TAT") else 0,
            "GhiChuDonHang": "Giao giờ hành chính, gọi trước khi giao" if o[2] == "ONLINE_APP" else "Mua trực tiếp tại cửa hàng",
            "MaKhuyenMai": o[16]
        })

    # ---------------------------------------------------------
    # BẢNG 16: BienDongKho (Sổ cái kho kết nối chặt với Đơn hàng)
    # ---------------------------------------------------------
    bien_dong_kho = []
    bdk_idx = 1
    
    # 1. Các giao dịch nhập kho ban đầu cho 10 biến thể tiêu biểu
    initial_bt_ids = ["BT00000004", "BT00000007", "BT00000010", "BT00000013", "BT00000016", 
                      "BT00000019", "BT00000022", "BT00000025", "BT00000028", "BT00000031"]
    for i, btid in enumerate(initial_bt_ids, 1):
        bt = bt_dict[btid]
        bien_dong_kho.append({
            "MaBienDong": f"BDK{bdk_idx:08d}",
            "MaBienThe": bt["MaBienThe"],
            "LoaiBienDong": "NHAP_KHO_BO_SUNG",
            "SoLuongThayDoi": 50,
            "SoLuongTonTruoc": bt["SoLuongTonKho"] - 50 if bt["SoLuongTonKho"] > 50 else 0,
            "SoLuongTonSau": bt["SoLuongTonKho"],
            "MaThamChieu": f"PNK261028_{i:02d}",
            "MaNhanVienThucHien": "ND00000010",
            "GhiChu": "Nhập kho bổ sung đầu kỳ từ xưởng may YoFaSo",
            "NgayThucHien": "2026-10-25 08:30:00"
        })
        bdk_idx += 1

    # 2. Các giao dịch xuất kho trừ hàng theo Đơn hàng
    for ctdh in chi_tiet_don_hang:
        dh_obj = next(d for d in don_hang if d["MaDonHang"] == ctdh["MaDonHang"])
        bt_obj = bt_dict[ctdh["MaBienThe"]]
        
        if dh_obj["TrangThaiDonHang"] in ("HOAN_TAT", "DANG_GIAO_HANG"):
            l_bd = "BAN_TAI_QUAY" if dh_obj["KenhBanHang"] == "TAI_QUAY_POS" else "XUAT_KHO_GIAO_HANG"
            nv = dh_obj["MaThuNgan"] if dh_obj["KenhBanHang"] == "TAI_QUAY_POS" else "ND00000011"
            
            bien_dong_kho.append({
                "MaBienDong": f"BDK{bdk_idx:08d}",
                "MaBienThe": ctdh["MaBienThe"],
                "LoaiBienDong": l_bd,
                "SoLuongThayDoi": -ctdh["SoLuong"],
                "SoLuongTonTruoc": bt_obj["SoLuongTonKho"] + ctdh["SoLuong"],
                "SoLuongTonSau": bt_obj["SoLuongTonKho"],
                "MaThamChieu": dh_obj["MaDonHang"],
                "MaNhanVienThucHien": nv,
                "GhiChu": f"Trừ kho tự động theo đơn hàng {dh_obj['MaDonHang']} ({dh_obj['KenhBanHang']})",
                "NgayThucHien": "2026-10-28 11:20:00"
            })
            bdk_idx += 1
            
        elif dh_obj["TrangThaiDonHang"] == "DANG_DONG_GOI":
            bien_dong_kho.append({
                "MaBienDong": f"BDK{bdk_idx:08d}",
                "MaBienThe": ctdh["MaBienThe"],
                "LoaiBienDong": "GIU_HANG_ONLINE",
                "SoLuongThayDoi": -ctdh["SoLuong"],
                "SoLuongTonTruoc": bt_obj["SoLuongTonKho"] + ctdh["SoLuong"],
                "SoLuongTonSau": bt_obj["SoLuongTonKho"],
                "MaThamChieu": dh_obj["MaDonHang"],
                "MaNhanVienThucHien": "ND00000012",
                "GhiChu": f"Giữ hàng đóng gói cho đơn online {dh_obj['MaDonHang']}",
                "NgayThucHien": "2026-10-28 14:10:00"
            })
            bdk_idx += 1

    # ---------------------------------------------------------
    # BẢNG 17: CaLamViecThuNgan (5 ca làm việc Web POS)
    # ---------------------------------------------------------
    ca_lam_viec = [
        ("CA000001", "ND00000007", "2026-10-27 08:00:00", "2026-10-27 16:00:00", 2000000.00, 5680000.00, "DA_PHE_DUYET_BAN_GIAO", "Ca sáng doanh thu ổn định, tiền mặt khớp 100% két"),
        ("CA000002", "ND00000008", "2026-10-27 16:00:00", "2026-10-27 22:00:00", 2000000.00, 8450000.00, "DA_PHE_DUYET_BAN_GIAO", "Ca tối lượng khách thử đồ VTON tại quầy đông, khớp két"),
        ("CA000003", "ND00000007", "2026-10-28 08:00:00", "2026-10-28 16:00:00", 2000000.00, 4890000.00, "DA_DONG_CHO_DUYET", "Đã kết ca, chờ quản lý admin phê duyệt bàn giao"),
        ("CA000004", "ND00000008", "2026-10-28 16:00:00", None, 2000000.00, None, "DANG_MO", "Ca chiều tối đang diễn ra bình thường"),
        ("CA000005", "ND00000009", "2026-10-26 08:00:00", "2026-10-26 16:00:00", 2000000.00, 6200000.00, "DA_PHE_DUYET_BAN_GIAO", "Ca mở tuần thuận lợi, không phát sinh chênh lệch tiền"),
    ]

    # ---------------------------------------------------------
    # BẢNG 18: DanhGiaSanPham (Khách Verified Buyer, HinhAnhThucTe = NULL)
    # ---------------------------------------------------------
    # Gắn đúng MaSanPham với sản phẩm trong ChiTietDonHang của đơn hàng HOAN_TAT
    danh_gia = [
        {
            "MaDanhGia": "DG00000001",
            "MaSanPham": "SP00000025",  # BT00000049 trong CTDH000001
            "MaNguoiDung": "ND00000002",
            "MaChiTietDonHang": "CTDH000001",
            "SoSao": 5,
            "NoiDungBinhLuan": "Áo polo chất vải Silk Touch sờ vào cực kỳ mướt tay, co giãn 4 chiều mặc rất mát và tôn dáng. Thử đồ VTON trên app thấy vừa vặn y như thật!",
            "HinhAnhThucTe": None,
            "PhanHoiTuShop": "Dạ YoFaSo Shop chân thành cảm ơn chị Mai Anh đã tin tưởng và đánh giá 5 sao cho shop ạ!",
            "TrangThaiDuyet": "DA_HIEN_THI"
        },
        {
            "MaDanhGia": "DG00000002",
            "MaSanPham": "SP00000008",  # BT00000016 trong CTDH000003
            "MaNguoiDung": "ND00000003",
            "MaChiTietDonHang": "CTDH000003",
            "SoSao": 5,
            "NoiDungBinhLuan": "Giao hàng hỏa tốc trong 24h. Form áo kiểu cổ polo đường may sắc sảo, nút đính chắc chắn, mua tặng bạn gái mặc đi làm ai cũng khen lịch lãm.",
            "HinhAnhThucTe": None,
            "PhanHoiTuShop": "Cảm ơn anh Long, chúc anh luôn tự tin và thành công cùng trang phục YoFaSo!",
            "TrangThaiDuyet": "DA_HIEN_THI"
        },
        {
            "MaDanhGia": "DG00000003",
            "MaSanPham": "SP00000008",  # BT00000016 trong CTDH000004
            "MaNguoiDung": "ND00000004",
            "MaChiTietDonHang": "CTDH000004",
            "SoSao": 5,
            "NoiDungBinhLuan": "Mua tại cửa hàng được bạn thu ngân hỗ trợ quét QR VTON trải nghiệm thử đồ rất thích. Chất vải dày dặn, giặt máy không bị xù lông.",
            "HinhAnhThucTe": None,
            "PhanHoiTuShop": "Dạ shop cảm ơn chị Quỳnh Chi, hẹn gặp lại chị trong những bộ sưu tập mới nhất ạ!",
            "TrangThaiDuyet": "DA_HIEN_THI"
        },
        {
            "MaDanhGia": "DG00000004",
            "MaSanPham": "SP00000022",  # BT00000046 trong CTDH000013
            "MaNguoiDung": "ND00000003",
            "MaChiTietDonHang": "CTDH000013",
            "SoSao": 4,
            "NoiDungBinhLuan": "Áo sơ mi form đứng, chiều dài tay áo vừa đẹp. Điểm trừ nhẹ là hộp giao hàng hơi móp chút xíu do vận chuyển nhưng sản phẩm bên trong nguyên vẹn.",
            "HinhAnhThucTe": None,
            "PhanHoiTuShop": "Dạ shop xin ghi nhận góp ý và sẽ làm việc kỹ hơn với khâu vận chuyển để phục vụ anh chu đáo hơn ạ!",
            "TrangThaiDuyet": "DA_HIEN_THI"
        },
        {
            "MaDanhGia": "DG00000005",
            "MaSanPham": "SP00000029",  # BT00000061 trong CTDH000018
            "MaNguoiDung": "ND00000006",
            "MaChiTietDonHang": "CTDH000018",
            "SoSao": 5,
            "NoiDungBinhLuan": "Áo thun cotton thiết kế rất thoải mái, chất vải mềm rũ tự nhiên, mặc lên tôn dáng cực kỳ. Rất đáng tiền!",
            "HinhAnhThucTe": None,
            "PhanHoiTuShop": "Cảm ơn chị Thảo đã yêu mến thiết kế của YoFaSo ạ!",
            "TrangThaiDuyet": "DA_HIEN_THI"
        },
    ]

    # ---------------------------------------------------------
    # BẢNG 19: ThongBao (16 thông báo đa kênh phong phú)
    # ---------------------------------------------------------
    thong_bao = [
        ("TB00000001", "ND00000002", "Đơn hàng DH00000001 đã giao thành công", "Kiện hàng của quý khách đã được giao thành công. Hãy đánh giá sản phẩm để nhận 50 điểm thưởng tích lũy nhé!", "DON_HANG", "/orders/DH00000001", 1),
        ("TB00000002", "ND00000002", "Kết quả thử đồ VTON đã sẵn sàng!", "Phiên thử đồ TD00000001 của bạn đã được OOTDiffusion xử lý xong với điểm thẩm mỹ 92/100 (RECOMMENDED). Xem ngay!", "THU_DO_VTON", "/vton/TD00000001", 1),
        ("TB00000003", "ND00000003", "Thanh toán VietQR đơn hàng DH00000002 thành công", "Hệ thống đã nhận được số tiền 420.000đ qua VietQR. Cửa hàng đang tiến hành đóng gói kiện hàng.", "DON_HANG", "/orders/DH00000002", 1),
        ("TB00000004", "ND00000003", "Kết quả thử đồ VTON TD00000002", "Gợi ý phối đồ set mới giúp nâng điểm lên 95/100. Khám phá ngay set đồ phối kèm!", "THU_DO_VTON", "/vton/TD00000002", 0),
        ("TB00000005", "ND00000004", "Đơn hàng DH00000005 đang trên đường giao", "Shipper Phan Quốc Bảo (0927654321) đang giao kiện hàng đến địa chỉ của bạn.", "DON_HANG", "/orders/DH00000005", 0),
        ("TB00000006", "ND00000005", "Chào mừng thành viên mới YoFaSo!", "Tặng bạn voucher YOFASO10 giảm ngay 10% cho đơn hàng thời trang đầu tiên.", "KHUYEN_MAI", "/promotions", 1),
        ("TB00000007", "ND00000006", "Đơn hàng DH00000007 đã được xác nhận", "Nhân viên kho đang nhặt hàng theo phiếu xuất kho và chuẩn bị đóng gói.", "DON_HANG", "/orders/DH00000007", 0),
        ("TB00000008", "ND00000010", "[CẢNH BÁO TỒN KHO] SKU-SM01 sắp chạm ngưỡng an toàn", "Biến thể SKU-SM01-DEN-M chỉ còn 5 sản phẩm khả dụng trong kho. Đề xuất tạo phiếu nhập hàng bổ sung.", "CANH_BAO_KHO", "/admin/inventory", 0),
        ("TB00000009", "ND00000010", "Yêu cầu nhặt hàng cho 3 đơn hàng mới", "Có 3 đơn hàng online mới cần xuất kho đóng gói trong ca làm việc hôm nay.", "CANH_BAO_KHO", "/admin/fulfillment", 1),
        ("TB00000010", "ND00000013", "Bạn có 2 kiện hàng mới cần giao", "Kiện hàng đơn DH00000005 đã được bàn giao từ kho, sẵn sàng vận chuyển.", "DON_HANG", "/shipper/tasks", 0),
        ("TB00000011", "ND00000007", "Nhắc nhở kết ca làm việc CA000003", "Vui lòng kiểm đếm tiền mặt thực tế trong két và nhập biên bản bàn giao ca.", "HE_THONG", "/pos/shift", 1),
        ("TB00000012", "ND00000001", "[ADMIN] Báo cáo doanh thu ngày 28/10", "Hôm nay ghi nhận 12 đơn hàng phát sinh, tổng doanh thu đạt 5.480.000đ.", "HE_THONG", "/admin/reports", 0),
        ("TB00000013", "ND00000001", "[ADMIN] Cần phê duyệt kết ca CA000003", "Thu ngân Đặng Lan Phương đã nộp báo cáo kết ca với 4.890.000đ tiền mặt trong két.", "HE_THONG", "/admin/shifts", 0),
        ("TB00000014", "ND00000004", "Ưu đãi thăng hạng thẻ KIM CƯƠNG", "Chúc mừng quý khách đã đạt hạng thẻ Kim Cương với nhiều đặc quyền giảm giá 15% độc quyền!", "KHUYEN_MAI", "/member/tier", 1),
        ("TB00000015", "ND00000005", "Đơn hàng DH00000011 đã được hủy theo yêu cầu", "Đơn hàng của bạn đã hủy thành công. Hy vọng được phục vụ quý khách trong lần mua sắm tiếp theo.", "DON_HANG", "/orders/DH00000011", 1),
    ]

    # ---------------------------------------------------------
    # SINH NỘI DUNG SQL TOÀN DIỆN
    # ---------------------------------------------------------
    sql_lines = []
    def add(line=""):
        sql_lines.append(line)

    add("-- =====================================================================================")
    add("-- HỆ THỐNG THƯƠNG MẠI ĐIỆN TỬ THỜI TRANG ĐA KÊNH YOFASO SHOP VTON")
    add("-- KỊCH BẢN INSERT DỮ LIỆU MẪU CHUẨN MỰC (SEED DATA SCRIPT CHO 19 BẢNG)")
    add("-- Hệ quản trị: MySQL 8.0+ | Database: yofasoshop_db_optimize | Engine: InnoDB")
    add("-- Quy chuẩn: Khớp 100% ERD_optimize.md, Cloudflare R2 và Sáp nhập hành chính 1/7/2025")
    add("-- =====================================================================================")
    add()
    add("USE yofasoshop_db_optimize;")
    add("SET FOREIGN_KEY_CHECKS = 0;")
    add()

    # 1. VaiTro
    add("-- -------------------------------------------------------------------------------------")
    add("-- BẢNG 1: VaiTro (5 vai trò chuẩn RBAC theo yêu cầu hệ thống)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO VaiTro (MaVaiTro, TenVaiTro, TenHienThi, MoTa, NgayTao) VALUES")
    vt_vals = [f"({escape_sql(r[0])}, {escape_sql(r[1])}, {escape_sql(r[2])}, {escape_sql(r[3])}, '2026-01-01 00:00:00')" for r in roles]
    add(",\n".join(vt_vals) + ";")
    add()

    # 2. NguoiDung
    add("-- -------------------------------------------------------------------------------------")
    add("-- BẢNG 2: NguoiDung (15 tài khoản gồm 1 Admin, 5 Khách hàng, 3 Thu ngân, 3 Thủ kho, 3 Shipper)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO NguoiDung (MaNguoiDung, TenDangNhap, MatKhau, HoVaTen, SoDienThoai, Email, AnhDaiDien, DiemTichLuy, HangThanhVien, NgaySinh, GioiTinh, TrangThaiHoatDong, NgayTao) VALUES")
    nd_vals = []
    for u in users_data:
        mnd = u[0]
        avatar = user_map.get(mnd, {}).get("AnhDaiDien", f"avatars/{mnd}/avatar-default.png")
        nd_vals.append(f"({escape_sql(u[0])}, {escape_sql(u[1])}, {escape_sql(u[2])}, {escape_sql(u[3])}, {escape_sql(u[4])}, {escape_sql(u[5])}, {escape_sql(avatar)}, {u[6]}, '{u[7]}', '{u[8]}', '{u[9]}', 'HOAT_DONG', '2026-01-01 08:00:00')")
    add(",\n".join(nd_vals) + ";")
    add()

    # 3. NguoiDung_VaiTro
    add("-- -------------------------------------------------------------------------------------")
    add("-- BẢNG 3: NguoiDung_VaiTro (Liên kết phân quyền người dùng)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO NguoiDung_VaiTro (MaNguoiDung, MaVaiTro, NgayGan) VALUES")
    ndvt_vals = [f"({escape_sql(u[0])}, {escape_sql(u[10])}, '2026-01-01 08:00:00')" for u in users_data]
    add(",\n".join(ndvt_vals) + ";")
    add()

    # 4. SoDiaChi
    add("-- -------------------------------------------------------------------------------------")
    add("-- BẢNG 4: SoDiaChi (15 địa chỉ chuẩn hóa sau sáp nhập 1/7/2025 từ tinhthanhpho.com)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO SoDiaChi (MaDiaChi, MaNguoiDung, TenNguoiNhan, SoDienThoaiNhan, SoNhaTenDuong, PhuongXa, TinhThanhPho, LaMacDinh) VALUES")
    dc_vals = [f"({escape_sql(d['MaDiaChi'])}, {escape_sql(d['MaNguoiDung'])}, {escape_sql(d['TenNguoiNhan'])}, {escape_sql(d['SoDienThoaiNhan'])}, {escape_sql(d['SoNhaTenDuong'])}, {escape_sql(d['PhuongXa'])}, {escape_sql(d['TinhThanhPho'])}, {d['LaMacDinh']})" for d in so_dia_chi]
    add(",\n".join(dc_vals) + ";")
    add()

    # 5. DanhMuc
    add("-- -------------------------------------------------------------------------------------")
    add("-- BẢNG 5: DanhMuc (Cây phân cấp 31 danh mục thời trang Nam, Nữ, Đầm váy)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO DanhMuc (MaDanhMuc, TenDanhMuc, MaDanhMucCha, MoTa, ThuTuHienThi, TrangThai) VALUES")
    dm_vals = [f"({escape_sql(d[0])}, {escape_sql(d[1])}, {escape_sql(d[2])}, {escape_sql(d[3])}, {d[4]}, 'HIEN_THI')" for d in danh_muc]
    add(",\n".join(dm_vals) + ";")
    add()

    # 6. SanPham (121 sản phẩm thực tế)
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 6: SanPham ({len(san_pham)} sản phẩm thực tế bóc tách từ kho AI Service)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO SanPham (MaSanPham, MaDanhMuc, TenSanPham, MoTaChiTiet, ChatLieu, KieuDang, LoaiCapQuan, ChieuDaiTrangPhuc, LoaiHoaTiet, PhanLoaiVTON, TieuLoaiTrangPhuc, BangMauTrichXuat, DuongDanMatNaVungThayDo, TrangThaiSanPham, NgayTao) VALUES")
    sp_vals = []
    for sp in san_pham:
        sp_vals.append(
            f"({escape_sql(sp['MaSanPham'])}, {escape_sql(sp['MaDanhMuc'])}, {escape_sql(sp['TenSanPham'])}, "
            f"{escape_sql(sp['MoTaChiTiet'])}, {escape_sql(sp['ChatLieu'])}, '{sp['KieuDang']}', "
            f"'{sp['LoaiCapQuan']}', '{sp['ChieuDaiTrangPhuc']}', '{sp['LoaiHoaTiet']}', "
            f"'{sp['PhanLoaiVTON']}', {escape_sql(sp['TieuLoaiTrangPhuc'])}, "
            f"{escape_sql(sp['BangMauTrichXuat'])}, {escape_sql(sp['DuongDanMatNaVungThayDo'])}, "
            f"'{sp['TrangThaiSanPham']}', '2026-10-15 00:00:00')"
        )
    add(",\n".join(sp_vals) + ";")
    add()

    # 7. BienTheSanPham (~360 biến thể)
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 7: BienTheSanPham ({len(bien_the)} SKU biến thể kích cỡ, màu sắc, tồn kho Single Pool)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO BienTheSanPham (MaBienThe, MaSanPham, MaQuanLyHangHoa, KichCo, TenMau, MaMauSac, GiaNiemYet, GiaKhuyenMai, SoLuongTonKho, ViTriGiaKe, TrangThai) VALUES")
    bt_vals = []
    for bt in bien_the:
        km_str = f"{bt['GiaKhuyenMai']:.2f}" if bt['GiaKhuyenMai'] is not None else "NULL"
        bt_vals.append(
            f"({escape_sql(bt['MaBienThe'])}, {escape_sql(bt['MaSanPham'])}, {escape_sql(bt['MaQuanLyHangHoa'])}, "
            f"'{bt['KichCo']}', {escape_sql(bt['TenMau'])}, {escape_sql(bt['MaMauSac'])}, "
            f"{bt['GiaNiemYet']:.2f}, {km_str}, {bt['SoLuongTonKho']}, {escape_sql(bt['ViTriGiaKe'])}, '{bt['TrangThai']}')"
        )
    add(",\n".join(bt_vals) + ";")
    add()

    # 8. HinhAnhSanPham (~400 ảnh)
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 8: HinhAnhSanPham ({len(hinh_anh)} ảnh gốc móc treo & ảnh bóc nền sạch phông trên R2)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO HinhAnhSanPham (MaHinhAnh, MaSanPham, DuongDanAnhGoc, DuongDanAnhBocNen, MaMauSac, LaAnhDaiDien, ThuTuHienThi) VALUES")
    ha_vals = []
    for ha in hinh_anh:
        ha_vals.append(
            f"({escape_sql(ha['MaHinhAnh'])}, {escape_sql(ha['MaSanPham'])}, {escape_sql(ha['DuongDanAnhGoc'])}, "
            f"{escape_sql(ha['DuongDanAnhBocNen'])}, {escape_sql(ha['MaMauSac'])}, {ha['LaAnhDaiDien']}, {ha['ThuTuHienThi']})"
        )
    add(",\n".join(ha_vals) + ";")
    add()

    # 9. PhienThuDoAo
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 9: PhienThuDoAo ({len(phien_thu_do)} phiên thử đồ ảo VTON kèm điểm thẩm mỹ 3 tiêu chí)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO PhienThuDoAo (MaPhienThuDo, MaNguoiDung, MaBienThe, LoaiKhungAnhNguoi, DuongDanAnhNguoiDung, DuongDanAnhKetQuaVTON, TrangThaiPhien, ThoiGianXuLy, ThongBaoLoi, DiemTongThe, DiemTiLeVocDang, DiemHoaHopMauSac, DiemCanBangHoaTiet, DiemTruPhat, QuyTacTiLeApDung, XepHangThamMy, LoiNhanXet, ThoiGianKhoiTao, ThoiGianHoanTat) VALUES")
    ptd_vals = []
    for p in phien_thu_do:
        ptd_vals.append(
            f"({escape_sql(p['MaPhienThuDo'])}, {escape_sql(p['MaNguoiDung'])}, {escape_sql(p['MaBienThe'])}, "
            f"'{p['LoaiKhungAnhNguoi']}', {escape_sql(p['DuongDanAnhNguoiDung'])}, {escape_sql(p['DuongDanAnhKetQuaVTON'])}, "
            f"'{p['TrangThaiPhien']}', {p['ThoiGianXuLy']}, {escape_sql(p['ThongBaoLoi'])}, "
            f"{p['DiemTongThe']}, {p['DiemTiLeVocDang']}, {p['DiemHoaHopMauSac']}, {p['DiemCanBangHoaTiet']}, "
            f"{p['DiemTruPhat']}, {escape_sql(p['QuyTacTiLeApDung'])}, '{p['XepHangThamMy']}', {escape_sql(p['LoiNhanXet'])}, "
            f"'2026-10-28 09:30:00', '2026-10-28 09:30:08')"
        )
    add(",\n".join(ptd_vals) + ";")
    add()

    # 10. GoiYPhoiDo
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 10: GoiYPhoiDo ({len(goi_y_phoi_do)} gợi ý phối đồ bán kèm Smart Cross-selling điểm >= 90)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO GoiYPhoiDo (MaGoiY, MaPhienThuDo, MaBienTheGoiY, DiemSauKhiPhoiKem, LyDoGoiY, TrangThaiTuongTac) VALUES")
    gy_vals = []
    for g in goi_y_phoi_do:
        gy_vals.append(
            f"({escape_sql(g['MaGoiY'])}, {escape_sql(g['MaPhienThuDo'])}, {escape_sql(g['MaBienTheGoiY'])}, "
            f"{g['DiemSauKhiPhoiKem']}, {escape_sql(g['LyDoGoiY'])}, '{g['TrangThaiTuongTac']}')"
        )
    add(",\n".join(gy_vals) + ";")
    add()

    # 11. KhuyenMai
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 11: KhuyenMai ({len(khuyen_mai)} chương trình khuyến mãi & voucher)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO KhuyenMai (MaKhuyenMai, MaPhieuGiamGia, TenChuongTrinh, LoaiGiamGia, GiaTriGiam, GiaTriDonHangToiThieu, GiamToiDa, SoLuongPhatHanh, SoLuongDaDung, NgayBatDau, NgayKetThuc, TrangThai) VALUES")
    km_vals = []
    for k in khuyen_mai:
        gt_max = f"{k[6]:.2f}" if k[6] is not None else "NULL"
        km_vals.append(
            f"({escape_sql(k[0])}, {escape_sql(k[1])}, {escape_sql(k[2])}, '{k[3]}', {k[4]:.2f}, "
            f"{k[5]:.2f}, {gt_max}, {k[7]}, {k[8]}, '{k[9]}', '{k[10]}', '{k[11]}')"
        )
    add(",\n".join(km_vals) + ";")
    add()

    # 12. GioHang
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 12: GioHang ({len(gio_hang)} giỏ hàng mua sắm online trên Mobile App)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO GioHang (MaGioHang, MaNguoiDung, NgayTao, NgayCapNhat) VALUES")
    gh_vals = [f"({escape_sql(g[0])}, {escape_sql(g[1])}, '2026-10-20 09:00:00', '2026-10-28 10:00:00')" for g in gio_hang]
    add(",\n".join(gh_vals) + ";")
    add()

    # 13. ChiTietGioHang
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 13: ChiTietGioHang ({len(chi_tiet_gio_hang)} dòng sản phẩm trong giỏ hàng)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO ChiTietGioHang (MaChiTietGioHang, MaGioHang, MaBienThe, SoLuong, NgayThem) VALUES")
    ctgh_vals = [f"({escape_sql(c['MaChiTietGioHang'])}, {escape_sql(c['MaGioHang'])}, {escape_sql(c['MaBienThe'])}, {c['SoLuong']}, '2026-10-28 09:00:00')" for c in chi_tiet_gio_hang]
    add(",\n".join(ctgh_vals) + ";")
    add()

    # 14. DonHang
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 14: DonHang ({len(don_hang)} đơn hàng đa kênh: Online App & Bán lẻ POS)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO DonHang (MaDonHang, MaNguoiDung, KenhBanHang, MaThuNgan, MaNguoiGiaoHang, MaDiaChiGiaoHang, TenNguoiNhan, SoDienThoaiNhan, DiaChiGiaoChiTiet, TienGiamGia, PhiVanChuyen, PhuongThucThanhToan, TrangThaiThanhToan, MaGiaoDichNgoai, ThoiGianThanhToan, TrangThaiDonHang, SoTienThuHo, SoLanGiao, GhiChuDonHang, MaKhuyenMai, NgayTao, NgayCapNhat) VALUES")
    dh_vals = []
    for d in don_hang:
        dh_vals.append(
            f"({escape_sql(d['MaDonHang'])}, {escape_sql(d['MaNguoiDung'])}, '{d['KenhBanHang']}', "
            f"{escape_sql(d['MaThuNgan'])}, {escape_sql(d['MaNguoiGiaoHang'])}, {escape_sql(d['MaDiaChiGiaoHang'])}, "
            f"{escape_sql(d['TenNguoiNhan'])}, {escape_sql(d['SoDienThoaiNhan'])}, {escape_sql(d['DiaChiGiaoChiTiet'])}, "
            f"{d['TienGiamGia']:.2f}, {d['PhiVanChuyen']:.2f}, '{d['PhuongThucThanhToan']}', "
            f"'{d['TrangThaiThanhToan']}', {escape_sql(d['MaGiaoDichNgoai'])}, {escape_sql(d['ThoiGianThanhToan'])}, "
            f"'{d['TrangThaiDonHang']}', {d['SoTienThuHo']:.2f}, {d['SoLanGiao']}, {escape_sql(d['GhiChuDonHang'])}, "
            f"{escape_sql(d['MaKhuyenMai'])}, '2026-10-28 10:00:00', '2026-10-28 10:30:00')"
        )
    add(",\n".join(dh_vals) + ";")
    add()

    # 15. ChiTietDonHang
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 15: ChiTietDonHang ({len(chi_tiet_don_hang)} chi tiết dòng hàng kèm liên kết VTON O2O)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO ChiTietDonHang (MaChiTietDonHang, MaDonHang, MaBienThe, TenSanPhamLucDatHang, TenMauLucDatHang, KichCoLucDatHang, SoLuong, DonGiaLucDatHang, MaPhienThuDo) VALUES")
    ctdh_vals = []
    for c in chi_tiet_don_hang:
        ctdh_vals.append(
            f"({escape_sql(c['MaChiTietDonHang'])}, {escape_sql(c['MaDonHang'])}, {escape_sql(c['MaBienThe'])}, "
            f"{escape_sql(c['TenSanPhamLucDatHang'])}, {escape_sql(c['TenMauLucDatHang'])}, '{c['KichCoLucDatHang']}', "
            f"{c['SoLuong']}, {c['DonGiaLucDatHang']:.2f}, {escape_sql(c['MaPhienThuDo'])})"
        )
    add(",\n".join(ctdh_vals) + ";")
    add()

    # 16. BienDongKho
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 16: BienDongKho ({len(bien_dong_kho)} dòng sổ cái biến động kho khớp chặt với đơn hàng)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO BienDongKho (MaBienDong, MaBienThe, LoaiBienDong, SoLuongThayDoi, SoLuongTonTruoc, SoLuongTonSau, MaThamChieu, MaNhanVienThucHien, GhiChu, NgayThucHien) VALUES")
    bdk_vals = []
    for b in bien_dong_kho:
        bdk_vals.append(
            f"({escape_sql(b['MaBienDong'])}, {escape_sql(b['MaBienThe'])}, '{b['LoaiBienDong']}', "
            f"{b['SoLuongThayDoi']}, {b['SoLuongTonTruoc']}, {b['SoLuongTonSau']}, "
            f"{escape_sql(b['MaThamChieu'])}, {escape_sql(b['MaNhanVienThucHien'])}, "
            f"{escape_sql(b['GhiChu'])}, '{b['NgayThucHien']}')"
        )
    add(",\n".join(bdk_vals) + ";")
    add()

    # 17. CaLamViecThuNgan
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 17: CaLamViecThuNgan ({len(ca_lam_viec)} ca làm việc POS Web)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO CaLamViecThuNgan (MaCaLamViec, MaThuNgan, ThoiGianMoCa, ThoiGianDongCa, TienMatDauCa, TienMatKiemDemThucTe, TrangThaiCa, GhiChuKetCa) VALUES")
    ca_vals = []
    for c in ca_lam_viec:
        dong_ca = f"'{c[3]}'" if c[3] is not None else "NULL"
        kiem_dem = f"{c[5]:.2f}" if c[5] is not None else "NULL"
        ca_vals.append(
            f"({escape_sql(c[0])}, {escape_sql(c[1])}, '{c[2]}', {dong_ca}, "
            f"{c[4]:.2f}, {kiem_dem}, '{c[6]}', {escape_sql(c[7])})"
        )
    add(",\n".join(ca_vals) + ";")
    add()

    # 18. DanhGiaSanPham
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 18: DanhGiaSanPham ({len(danh_gia)} đánh giá Verified Buyer, HinhAnhThucTe = NULL)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO DanhGiaSanPham (MaDanhGia, MaSanPham, MaNguoiDung, MaChiTietDonHang, SoSao, NoiDungBinhLuan, HinhAnhThucTe, PhanHoiTuShop, TrangThaiDuyet, NgayDanhGia) VALUES")
    dg_vals = []
    for d in danh_gia:
        img_val = escape_sql(d["HinhAnhThucTe"])
        dg_vals.append(
            f"({escape_sql(d['MaDanhGia'])}, {escape_sql(d['MaSanPham'])}, {escape_sql(d['MaNguoiDung'])}, "
            f"{escape_sql(d['MaChiTietDonHang'])}, {d['SoSao']}, {escape_sql(d['NoiDungBinhLuan'])}, "
            f"{img_val}, {escape_sql(d['PhanHoiTuShop'])}, '{d['TrangThaiDuyet']}', '2026-10-28 14:00:00')"
        )
    add(",\n".join(dg_vals) + ";")
    add()

    # 19. ThongBao
    add("-- -------------------------------------------------------------------------------------")
    add(f"-- BẢNG 19: ThongBao ({len(thong_bao)} thông báo in-app đa kênh)")
    add("-- -------------------------------------------------------------------------------------")
    add("INSERT INTO ThongBao (MaThongBao, MaNguoiDung, TieuDe, NoiDung, LoaiThongBao, LienKetDieuHuong, DaDoc, NgayTao) VALUES")
    tb_vals = []
    for t in thong_bao:
        tb_vals.append(
            f"({escape_sql(t[0])}, {escape_sql(t[1])}, {escape_sql(t[2])}, {escape_sql(t[3])}, "
            f"'{t[4]}', {escape_sql(t[5])}, {t[6]}, '2026-10-28 10:00:00')"
        )
    add(",\n".join(tb_vals) + ";")
    add()

    add("SET FOREIGN_KEY_CHECKS = 1;")
    add("-- =====================================================================================")
    add("-- KẾT THÚC KỊCH BẢN INSERT DỮ LIỆU MẪU YOFASOSHOP")
    add("-- =====================================================================================")

    # Ghi file SQL độc lập
    output_sql_path = repo_root / "yofasoshop_seed_data.sql"
    output_sql_path.write_text("\n".join(sql_lines), encoding="utf-8")
    print(f"Generated standalone SQL file: {output_sql_path}")
    print(f"Total lines: {len(sql_lines)}")

    return output_sql_path, {
        "roles": len(roles),
        "users": len(users_data),
        "addresses": len(so_dia_chi),
        "categories": len(danh_muc),
        "products": len(san_pham),
        "variants": len(bien_the),
        "images": len(hinh_anh),
        "tryon": len(phien_thu_do),
        "cross_sell": len(goi_y_phoi_do),
        "promotions": len(khuyen_mai),
        "carts": len(gio_hang),
        "cart_items": len(chi_tiet_gio_hang),
        "orders": len(don_hang),
        "order_items": len(chi_tiet_don_hang),
        "inventory_ledger": len(bien_dongKho := bien_dong_kho),
        "cashier_shifts": len(ca_lam_viec),
        "reviews": len(danh_gia),
        "notifications": len(thong_bao),
    }


if __name__ == "__main__":
    main()
