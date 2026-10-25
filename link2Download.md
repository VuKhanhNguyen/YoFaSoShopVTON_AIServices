# HƯỚNG DẪN TẢI & THIẾT LẬP TOÀN BỘ TRỌNG SỐ, MÔ HÌNH VÀ DỮ LIỆU (MASTER DOWNLOAD GUIDE)

**Dự án:** YoFaSoShop_VTON (Ứng dụng TMĐT Thời trang tích hợp Thử đồ ảo & Chấm điểm Thẩm mỹ)
**Dịch vụ AI:** `YoFaSoShop_AIService`
**Đối tượng sử dụng:** Giảng viên hướng dẫn (ThS. Phạm Thị Trà My), Hội đồng chấm đồ án tốt nghiệp, và Developers khi clone mã nguồn về máy mới.

---

## 📌 LÝ DO CẦN FILE HƯỚNG DẪN NÀY

Trong tệp `.gitignore` của dự án, toàn bộ các tệp trọng số (`*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `*.pkl`) và tập dữ liệu thô (`data/raw/`) đều bị chặn đẩy lên GitHub vì dung lượng vượt quá giới hạn (hơn **30 GB**).
Do đó, khi **Giảng viên hoặc Developer clone repository từ GitHub về máy**, thư mục dự án sẽ chỉ có mã nguồn code. Để hệ thống khởi chạy và suy luận được, bạn cần tải các tệp mô hình và dữ liệu theo hướng dẫn chi tiết dưới đây.

---

## 🧭 1. BẢNG TỔNG HỢP TOÀN BỘ TÀI NGUYÊN CẦN TẢI (4 NHÓM)

### 🔹 NHÓM 1: LÕI THỬ ĐỒ ẢO THEO CATEGORY (IDM-VTON, OOTDIFFUSION & PREPROCESSING)

| Tên Mô Hình                                                            | File / Thư mục cần tải                                                                                                              | Dung lượng | Link tải chính thức & Mirror                                                                                                                                                                                                              | Thư mục lưu trữ đích trong`YoFaSoShop_AIService/` | Vai trò trong hệ thống                                                                                                                                |
| :------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------- | :-----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **YOLOv8s-seg (DeepFashion2)**                                      | `deepfashion2_yolov8s-seg.pt`                                                                                                         |   ~23.8 MB   | •[HuggingFace Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/blob/main/deepfashion2_yolov8s-seg.pt)• [Direct Download](https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt)                          | `weights/`                                              | Nhận diện 13 loại trang phục, phân đoạn và bóc tách sạch nền ảnh sản phẩm của shop                                                       |
| **DensePose (Meta Detectron2)**                                     | `model_final_162be9.pkl`                                                                                                              |    ~255 MB    | •[Meta Research CDN](https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl)• [Mirror HF yisol/IDM-VTON](https://huggingface.co/yisol/IDM-VTON/resolve/main/densepose/model_final_162be9.pkl) | `api/models/idm-vton/ckpt/densepose/`                   | Khớp lưới 3D bề mặt cơ thể (IUV mapping), uốn nếp vải theo dáng đứng người mẫu                                                           |
| **Human Parsing (ATR ONNX)**                                        | `parsing_atr.onnx`                                                                                                                    |    ~266 MB    | •[HF yisol/IDM-VTON](https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx)                                                                                                                                       | `api/models/idm-vton/ckpt/humanparsing/`                | Tách vùng mặt, tóc, cổ, da tay và đồ cũ để tạo mặt nạ Agnostic Mask (bảo tồn 100% da mặt)                                               |
| **OpenPose (CMU Pose)**                                             | `body_pose_model.pth`                                                                                                                 |    ~209 MB    | •[HF yisol/IDM-VTON](https://huggingface.co/yisol/IDM-VTON/resolve/main/openpose/ckpts/body_pose_model.pth)                                                                                                                                  | `api/models/idm-vton/ckpt/openpose/ckpts/`              | Định vị 18 điểm khớp xương cơ thể để nhận diện tư thế đứng                                                                             |
| **IDM-VTON Diffusion Pipeline** *(Lõi sinh ảnh thử đồ SDXL)* | Trọn bộ thư mục:`unet/`, `unet_encoder/`, `vae/`, `image_encoder/`, `text_encoder/`, `scheduler/`, `model_index.json` |    ~20 GB    | •[HF yisol/IDM-VTON](https://huggingface.co/yisol/IDM-VTON/tree/main)*(Xem lệnh tải nhanh bằng `huggingface-cli` ở Mục 3)*                                                                                                          | `api/models/idm-vton/ckpt/idm_vton_weights/`            | Mạng khuếch tán Denoising UNet + GarmentNet uốn chi tiết hoa văn vải lên người mẫu                                                            |
| **OOTDiffusion Space snapshot** *(tùy chọn self-host)*          | `checkpoints/`, `ootd/`, `preprocess/`, `run/`, `requirements.txt`                                                            | khoảng 15 GB | •[HF Space levihsu/OOTDiffusion](https://huggingface.co/spaces/levihsu/OOTDiffusion/tree/main)• [GitHub upstream](https://github.com/levihsu/OOTDiffusion)                                                                                   | `api/models/ootdiffusion/`                              | Backend DressCode cho`lower_body` và `dress`. Notebook gọi Space từ xa nên không bắt buộc tải; snapshot dùng khi triển khai worker riêng. |

> [!IMPORTANT]
> Không tải model repository riêng rồi kỳ vọng có ngay ứng dụng chạy hoàn chỉnh. Với mục tiêu self-host theo Space mà notebook đang gọi, snapshot **Space** là lựa chọn phù hợp vì chứa cả code, preprocessing và checkpoints. Token chỉ đặt trong biến môi trường `HF_TOKEN`, Colab Secrets hoặc cache của `hf auth login`; tuyệt đối không ghi token vào tài liệu/lệnh đã commit.

---

### 🔹 NHÓM 2: MÔ HÌNH PHÂN ĐOẠN & THUỘC TÍNH BỔ TRỢ (SUPPLEMENTARY & BACKBONE MODELS)

| Tên Mô Hình                                            | File cần tải                                                                                                   |  Dung lượng  | Link tải chính thức                                                                                                                                                                                                                                  | Thư mục lưu trữ đích trong`YoFaSoShop_AIService/` | Vai trò trong hệ thống                                                                                       |
| :-------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- | :-------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **Garment Segmentation U-Net (ResNet-50 Backbone)** | `model.safetensors`                                                                                            |     ~130 MB     | •[HF zyuzuguldu/garment-segmentation-unet-resnet50](https://huggingface.co/zyuzuguldu/garment-segmentation-unet-resnet50/resolve/main/model.safetensors)• [Trang chủ Model Card](https://huggingface.co/zyuzuguldu/garment-segmentation-unet-resnet50) | `api/models/unet_resnet50/`                             | Mô hình phân đoạn nhị phân chuyên sâu tách biệt quần áo khỏi nền (độ chính xác IoU = 0.8964) |
| **MMFashion ResNet-50 Attribute Model**             | `resnet50_attr_model_best_clean.pth` *(Bản gốc)*`resnet50_attr_model_best_fp16.pth` *(Bản FP16 nhẹ)* | ~1.82 GB~911 MB | •[MMFashion Model Zoo](https://mmfashion.readthedocs.io/en/latest/MODEL_ZOO/#attribute-prediction)• Checkpoint Google Drive đính kèm trong Model Zoo CUHK                                                                                           | `weights/`                                              | Mô hình nhận diện 1.000 thuộc tính trang phục đa nhãn (Multi-label Attribute Prediction)               |

---

### 🔹 NHÓM 3: CƠ SỞ TRI THỨC PHỐI ĐỒ & CHẤM ĐIỂM (KNOWLEDGE BASE CHO SCORING ENGINE)

| Tên Bộ Dữ Liệu                                                     | Thư mục / File cần tải                                                                                                                               | Link tải chính thức                                                                                                                                                                                                                           | Thư mục lưu trữ đích trong`YoFaSoShop_AIService/` | Vai trò trong hệ thống                                                                                                                                                                                          |
| :--------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FashionStylist Knowledge Base** *(Bộ quy tắc thời trang)* | Tải trọn bộ thư mục`Dataset/` gồm 3 thư mục con:• `Child/`• `Female/`• `Male/`*(Chứa các file `look_en.csv`, `label_en.csv`)* | •[GitHub recsys-benchmark/FashionStylist](https://github.com/recsys-benchmark/FashionStylist/tree/main/Dataset)• [Tài liệu quy tắc phối đồ (PDF)](https://github.com/recsys-benchmark/FashionStylist/blob/main/Supplementary_material.pdf) | `data/knowledge/`                                       | Chứa dữ liệu 1.000 outfit chuẩn mực và 4.637 món đồ phân tích chi tiết kiểu dáng, chất liệu, hoàn cảnh mặc. Phục vụ thuật toán chấm điểm và gợi ý mua kèm (Cross-selling) ở Tuần 5 |

---

### 🔹 NHÓM 4: TẬP DỮ LIỆU THÔ, NHÃN ĐÁNH GIÁ & TOOL TIỀN XỬ LÝ (OFFLINE REFERENCE & BENCHMARK)

| Tên Dữ Liệu / Script                                     | File / Thư mục cần tải                                                                                                                                            | Link tải chính thức                                                                                                      | Thư mục lưu trữ đích trong`YoFaSoShop_AIService/` | Vai trò trong hệ thống                                                                                                                                                                                                  |
| :---------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Script chuyển đổi nhãn DeepFashion2 sang COCO** | `deepfashion2_to_coco.py`                                                                                                                                           | •[GitHub switchablenorms/DeepFashion2](https://github.com/switchablenorms/DeepFashion2/blob/master/deepfashion2_to_coco.py) | `ml-training/preprocess/`                               | Script Python chuẩn của tác giả DeepFashion2 dùng chuyển đổi nhãn JSON sang chuẩn COCO JSON để train/đo mAP                                                                                                   |
| **DeepFashion2 Validation Dataset**                   | •`validation/` *(ảnh thời trang)*• `json_for_validation/` *(nhãn)*                                                                                       | •[GitHub switchablenorms/DeepFashion2](https://github.com/switchablenorms/DeepFashion2/)                                    | `data/raw/deepfashion2/`                                | **Rất quan trọng:** Cung cấp ảnh mẫu để test ở Tuần 4 và nhãn Ground Truth để chạy script đo chỉ số khoa học (**mAP@50**, **mAP@50:95**) ở Tuần 14 đưa vào báo cáo tốt nghiệp |
| **DeepFashion Attribute Annotations**                 | Thư mục`Anno/` gồm 5 file:• `list_attr_cloth.txt`• `list_category_cloth.txt`• `list_bbox.txt`• `list_category_img.txt`• `list_eval_partition.txt` | •[CUHK MMLab Attribute Benchmark](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html/)              | `data/raw/deepfashion_attr/Anno/`                       | Từ điển tra cứu 1.000 thuộc tính trang phục quốc tế (Fashion Taxonomy) và minh chứng nghiên cứu thực nghiệm làm dày cơ sở lý thuyết trong báo cáo                                                   |

---

## 🌳 2. SƠ ĐỒ CÂY THƯ MỤC TỔNG THỂ DỰ ÁN SAU KHI ĐẶT ĐẦY ĐỦ FILE

Sau khi tải và đưa toàn bộ các tệp vào dự án, cấu trúc bên trong `YoFaSoShop_AIService/` sẽ chuẩn xác 100% như sau:

```text
YoFaSoShop_AIService/
├── api/
│   ├── main.py                                      # File khởi chạy dịch vụ FastAPI
│   └── models/
│       ├── idm-vton/                                # [NHÓM 1: LÕI THỬ ĐỒ ẢO IDM-VTON]
│       │   └── ckpt/
│       │       ├── densepose/
│       │       │   └── model_final_162be9.pkl       (255 MB - Khớp lưới 3D cơ thể)
│       │       ├── humanparsing/
│       │       │   └── parsing_atr.onnx             (266 MB - Tách da mặt/tóc/đồ cũ)
│       │       ├── openpose/
│       │       │   └── ckpts/
│       │       │       └── body_pose_model.pth      (209 MB - 18 khớp xương)
│       │       └── idm_vton_weights/                (~20 GB - UNet, GarmentNet, VAE, Encoders)
│       │           ├── model_index.json
│       │           ├── unet/
│       │           ├── unet_encoder/
│       │           ├── vae/
│       │           ├── image_encoder/
│       │           ├── text_encoder/ & text_encoder_2/
│       │           ├── tokenizer/ & tokenizer_2/
│       │           └── scheduler/
│       │
│       ├── ootdiffusion/                            # [TÙY CHỌN: SELF-HOST LOWER-BODY/DRESS]
│       │   ├── checkpoints/
│       │   ├── ootd/
│       │   ├── preprocess/
│       │   ├── run/
│       │   └── requirements.txt
│       │
│       └── unet_resnet50/                           # [NHÓM 2: MÔ HÌNH U-NET BỔ TRỢ]
│           └── model.safetensors                    (130 MB - Tách nền quần áo IoU 0.8964)
│
├── weights/
│   ├── deepfashion2_yolov8s-seg.pt                  # [NHÓM 1] 23.8 MB - YOLOv8 cắt nền đồ shop
│   ├── resnet50_attr_model_best_clean.pth           # [NHÓM 2] 1.82 GB - Backbone nhận diện thuộc tính
│   └── resnet50_attr_model_best_fp16.pth            # [NHÓM 2] 911 MB - Bản nén FP16
│
├── data/
│   ├── knowledge/                                   # [NHÓM 3: TRI THỨC PHỐI ĐỒ FASHIONSTYLIST]
│   │   ├── Child/                                   (look_en.csv, label_en.csv)
│   │   ├── Female/                                  (look_en.csv, label_en.csv)
│   │   └── Male/                                    (look_en.csv, label_en.csv)
│   │
│   ├── raw/                                         # [NHÓM 4: DỮ LIỆU ĐÁNH GIÁ HỌC THUẬT & TỪ ĐIỂN]
│   │   ├── deepfashion2/
│   │   │   ├── validation/                          (Kho ảnh mẫu test thực tế)
│   │   │   └── json_for_validation/                 (Ground Truth đo mAP@50)
│   │   │
│   │   └── deepfashion_attr/
│   │       └── Anno/                                (Từ điển 1.000 thuộc tính & Bounding box)
│   │           ├── list_attr_cloth.txt
│   │           ├── list_category_cloth.txt
│   │           ├── list_bbox.txt
│   │           ├── list_category_img.txt
│   │           └── list_eval_partition.txt
│   │
│   └── test_samples/                                # [DỮ LIỆU TEST TUẦN 4]
│       ├── garments/ (tops, bottoms, dresses)
│       ├── persons/
│       └── results/ (segmented, tryon)
│
├── ml-training/
│   └── preprocess/
│       └── deepfashion2_to_coco.py                  # [NHÓM 4] Tool convert nhãn sang chuẩn COCO
│
├── scripts/
│   ├── download_weights.ps1                         # Script tự động tải các model nhỏ (< 1GB)
│   ├── download_idm_vton.py                         # Script tự động tải IDM-VTON từ Hugging Face
│   ├── download_ootdiffusion.py                     # Tải snapshot Space cho self-host
│   ├── download_ootdiffusion.ps1                    # Wrapper PowerShell
│   ├── download_ootdiffusion.sh                     # Wrapper Bash/Linux
│   ├── test_environment.py                          # Script kiểm tra tính toàn vẹn (Sanity Check)
│   ├── test_segmentation.py                         # Script chạy thử bóc tách nền sản phẩm
│   └── test_tryon_pipeline.py                       # Script chạy thử pipeline thử đồ ảo
│
└── link2Download.md                                 # Tài liệu tra cứu này
```

---

## ⚡ 3. HƯỚNG DẪN TẢI TỰ ĐỘNG BẰNG CÁC SCRIPT TIỆN ÍCH

### Bước 1: Tải các mô hình tiền xử lý & U-Net (< 1 GB)

Mở PowerShell tại thư mục gốc dự án và chạy:

```powershell
powershell -ExecutionPolicy Bypass -File YoFaSoShop_AIService/scripts/download_weights.ps1
```

Để tải thêm mô hình U-Net ResNet50 (`model.safetensors`):

```powershell
$unetDir = "YoFaSoShop_AIService/api/models/unet_resnet50"
New-Item -ItemType Directory -Force -Path $unetDir | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/zyuzuguldu/garment-segmentation-unet-resnet50/resolve/main/model.safetensors" -OutFile "$unetDir/model.safetensors"
```

### Bước 2: Tải trọn bộ lõi khuếch tán IDM-VTON (~20 GB)

Đảm bảo đã cài thư viện Hugging Face Hub:

```bash
pip install huggingface_hub
```

Chạy script Python tự động tải (có hỗ trợ tự động tiếp tục khi rớt mạng `resume_download=True`):

```bash
python YoFaSoShop_AIService/scripts/download_idm_vton.py
```

Hoặc tải trực tiếp bằng lệnh CLI:

```bash
huggingface-cli download yisol/IDM-VTON --exclude "densepose/*" "humanparsing/*" "openpose/*" "assets/*" --local-dir YoFaSoShop_AIService/api/models/idm-vton/ckpt/idm_vton_weights
```

### Bước 3: Tải OOTDiffusion để self-host (tùy chọn, khoảng 15 GB)

Notebook `week4_experiment_colab.ipynb` mặc định gọi Hugging Face Space nên có thể bỏ qua bước này. Chỉ tải khi có Linux/GPU worker riêng và đã đọc README/licence của upstream.

```powershell
# Xem kế hoạch, không tải
python YoFaSoShop_AIService/scripts/download_ootdiffusion.py --dry-run

# Tải snapshot chạy được của Space vào api/models/ootdiffusion/
python YoFaSoShop_AIService/scripts/download_ootdiffusion.py

# Tương đương bằng wrapper PowerShell
& YoFaSoShop_AIService/scripts/download_ootdiffusion.ps1
```

Trên Linux/Colab có thể chạy `bash YoFaSoShop_AIService/scripts/download_ootdiffusion.sh`. Script chỉ lấy `checkpoints/`, `ootd/`, `preprocess/`, `run/*.py`, `requirements.txt` và `README.md`; tự kiểm tra tối thiểu 20 GiB trống và có thể chạy lại để tận dụng cache.

### Bước 4: Tải tập dữ liệu FashionStylist (Knowledge Base)

Clone repo FashionStylist về một thư mục tạm, sau đó copy thư mục `Dataset/` vào `YoFaSoShop_AIService/data/knowledge/`:

```bash
git clone https://github.com/recsys-benchmark/FashionStylist.git temp_stylist
# Copy 3 thư mục Child, Female, Male vào data/knowledge/
cp -r temp_stylist/Dataset/* YoFaSoShop_AIService/data/knowledge/
# Xóa thư mục tạm
rm -rf temp_stylist
```

---

## 🛡️ 4. KIỂM THỬ TÍNH TOÀN VẸN (SANITY CHECK) SAU KHI TẢI

Sau khi đã tải và đặt các tệp vào đúng thư mục, hãy chạy script kiểm tra chốt chặn:

```bash
python YoFaSoShop_AIService/scripts/test_environment.py
```

### Tiêu chuẩn nghiệm thu đạt 100% Pass:

* `deepfashion2_yolov8s-seg.pt` : **22.75 MB** ➜ `[PASS] Nguyên vẹn`
* `model_final_162be9.pkl`      : **243.91 MB** ➜ `[PASS] Nguyên vẹn`
* `parsing_atr.onnx`            : **254.50 MB** ➜ `[PASS] Nguyên vẹn`
* `body_pose_model.pth`         : **199.57 MB** ➜ `[PASS] Nguyên vẹn`
* `idm_vton_weights/`           : **Đầy đủ 8 modules** ➜ `[PASS]`
* `ootdiffusion/` *(nếu self-host)*: Có `checkpoints/`, `ootd/`, `preprocess/`, `run/` ➜ `[PASS]`
* `model.safetensors` (U-Net)   : **124.3 MB** ➜ Sẵn sàng trong `api/models/unet_resnet50/`
* `data/knowledge/`             : Đầy đủ 3 bộ `Child`, `Female`, `Male` phục vụ thuật toán chấm điểm.

---

## 💻 5. YÊU CẦU PHẦN CỨNG & KHUYẾN NGHỊ THI CÔNG

* **Mô hình Nhẹ (YOLOv8, DensePose, Parsing, OpenPose, U-Net ResNet50):**
  * Tiêu thụ ít tài nguyên (< 2GB VRAM).
  * Chạy mượt mà trực tiếp trên Laptop cá nhân (RTX 3050 6GB VRAM) hoặc CPU Intel Core i7.
* **Mô hình Nặng (IDM-VTON và OOTDiffusion):**
  * Yêu cầu mặc định: **12GB – 16GB VRAM**.
  * **Trên Laptop RTX 3050 (6GB VRAM):** Bắt buộc kích hoạt chế độ `enable_model_cpu_offload()` và `enable_vae_tiling()` trong mã nguồn (đã tích hợp trong `scripts/test_tryon_pipeline.py`) để tránh lỗi `CUDA Out of Memory (OOM)`. Thời gian sinh ảnh khoảng 35–55 giây/ảnh.
  * **Trên Google Colab Free (GPU Tesla T4 16GB VRAM):** Nạp toàn bộ pipeline lên CUDA, thời gian sinh ảnh chỉ mất **4–6 giây/ảnh** (Khuyên dùng khi cần render nhanh số lượng lớn mẫu ảnh để nộp báo cáo đồ án).
  * **OOTDiffusion:** Upstream hướng dẫn Linux; snapshot local khoảng 15 GB và chưa có cam kết chạy ổn định trên RTX 3050 6GB. Với notebook Tuần 4, ưu tiên Space + `HF_TOKEN` trong Secrets; với demo ổn định, dùng GPU worker tự host/thuê riêng và theo dõi quota.
