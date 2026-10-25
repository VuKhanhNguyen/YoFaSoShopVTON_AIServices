# DỮ LIỆU MẪU KIỂM THỬ TUẦN 4 - YOFASOSHOP_VTON
Thư mục lưu trữ 20-30 mẫu ảnh thời trang thực tế thu thập từ web và ảnh người mẫu phục vụ chạy thử nghiệm bóc tách phông nền và ghép đồ ảo IDM-VTON (Tuần 4: 21/09 - 27/09/2026).

---

## 1. Cấu trúc thư mục chuẩn

```text
data/test_samples/
├── garments/                     # Ảnh sản phẩm thời trang của shop (20 - 30 ảnh)
│   ├── tops/                     # Áo (áo thun, sơ mi, áo khoác, áo len...)
│   ├── bottoms/                  # Quần & Chân váy (quần jean, âu, kaki, short, chân váy chữ A...)
│   └── dresses/                  # Đầm liền thân (đầm suông, đầm xòe, đầm dạ hội...)
│
├── persons/                      # Ảnh người mẫu thử đồ (5 - 10 ảnh)
│   ├── model_female_01.jpg       # Người mẫu nữ đứng thẳng, rõ dáng
│   ├── model_male_01.jpg         # Người mẫu nam đứng thẳng, rõ dáng
│   └── ...
│
└── results/                      # Kết quả sinh ra từ các script AI
    ├── segmented/                # Ảnh sản phẩm đã bóc tách nền (PNG trong suốt RGBA)
    └── tryon/                    # Ảnh thành phẩm thử đồ ảo từ IDM-VTON
```

---

## 2. Tiêu chuẩn thu thập ảnh sản phẩm (Garment Images - 20 đến 30 mẫu)

Để mô hình **DeepFashion2 (YOLOv8-seg)** bóc tách nền sạch sẽ và **IDM-VTON** uốn nếp vải chuẩn nhất, ảnh thu thập cần tuân thủ:

1. **Góc chụp:** Chụp chính diện (Front-view), trải phẳng (flat-lay) hoặc mặc trên ma-nơ-canh / treo móc. Tránh ảnh góc nghiêng quá 30 độ.
2. **Hậu cảnh (Background):** Nền đơn sắc (trắng, xám nhạt, be) tương tự ảnh studio chuẩn của các hãng thời trang.
3. **Độ phân giải:** Tối thiểu $768 \times 1024$ pixel (tỉ lệ 3:4 - đúng chuẩn đầu vào của IDM-VTON).
4. **Phân bổ danh mục tối thiểu cho YoFaSo Shop:**
   - **Tops (10 - 12 mẫu):** Áo thun trơn, áo thun in họa tiết, áo sơ mi dài tay, áo sơ mi cộc tay, áo khoác blazer/jacket.
   - **Bottoms (8 - 10 mẫu):** Quần âu công sở, quần jeans dài, quần short kaki, chân váy dài, chân váy ngắn.
   - **Dresses (4 - 6 mẫu):** Đầm suông liền thân, đầm hoa xòe.
5. **Nguồn thu thập tham khảo:** Zara, Uniqlo, Coolmate, Routine, H&M, Shopee Mall các local brand Việt Nam.

---

## 3. Tiêu chuẩn thu thập ảnh người mẫu (Person Images - 5 đến 10 mẫu)

1. **Dáng đứng:** Toàn thân (full-body) hoặc 3/4 thân trên, đứng thẳng nhìn thẳng vào ống kính.
2. **Tư thế:** Hai tay buông nhẹ hoặc chống hông nhẹ, hai bàn tay tách rời thân mình (không bắt chéo tay trước ngực, không đút tay vào túi quần sâu) để mô hình Human Parsing và OpenPose bắt chuẩn 18 khớp xương.
3. **Trang phục ban đầu của mẫu:** Mặc trang phục gọn gàng, đơn giản (ví dụ: áo thun ôm và quần short/legging) giúp mô hình dễ dàng xóa đồ cũ mà không làm mất diện tích tay chân.
4. **Ánh sáng:** Rõ nét, không bị đổ bóng quá đậm lên mặt và cơ thể.
