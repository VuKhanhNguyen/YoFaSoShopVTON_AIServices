# YoFaSoShop · AI Service

> **30/10/2026:** AI không kết nối MySQL/SQLite, không tạo bảng. Queue/asset/heartbeat dùng JSON manifest trong RUNTIME_DIR, API và một worker chia sẻ volume trên một host. Backend giữ đúng 19 bảng nghiệp vụ. Xem [setupGuide](../YoFaSoShop_Docs/extra-info/setupGuide.md).

**FastAPI cho phân tích ảnh sản phẩm, kiểm tra ảnh người và thử đồ ảo bằng OOTDiffusion.**

AI Service xử lý ảnh và các tác vụ AI. Spring Boot sẽ phụ trách tài khoản, sản phẩm,
giỏ hàng, đơn hàng, tồn kho, điểm thẩm mỹ và gợi ý sản phẩm theo quy tắc nghiệp vụ.

> **Mới clone repo?** Chạy FastAPI ở chế độ giả lập trước theo mục 2.
> Bạn chưa cần thuê GPU hoặc tải model để kiểm tra API.
> Khi muốn sinh ảnh thử đồ thật, tiếp tục mục 3–5 trên máy GPU.

## Mục lục

- [1. Chuẩn bị và chọn cách chạy](#1-chuẩn-bị-và-chọn-cách-chạy)
- [2. Cài và chạy FastAPI trên laptop](#2-cài-và-chạy-fastapi-trên-laptop)
- [3. Tải code và checkpoint OOTDiffusion](#3-tải-code-và-checkpoint-ootdiffusion)
- [4. Chuẩn bị weights riêng của dự án](#4-chuẩn-bị-weights-riêng-của-dự-án)
- [5. Chạy model thật trên máy GPU](#5-chạy-model-thật-trên-máy-gpu)
- [6. Chạy bằng Docker](#6-chạy-bằng-docker)
- [7. Kiểm thử API](#7-kiểm-thử-api)
- [8. Cấu trúc thư mục và quản lý tài nguyên](#8-cấu-trúc-thư-mục-và-quản-lý-tài-nguyên)
- [9. Xử lý lỗi thường gặp](#9-xử-lý-lỗi-thường-gặp)

## 1. Chuẩn bị và chọn cách chạy

| Mục đích                                                               | Chế độ | Cần GPU/model?                         |
| ------------------------------------------------------------------------- | --------- | --------------------------------------- |
| Xem Swagger, kiểm tra upload, xác thực, job/status và lỗi đầu vào | `mock`  | Không                                  |
| Phân tích ảnh bằng detector, sinh ảnh thử đồ và hậu kiểm thật | `real`  | Cần weights; VTON cần GPU NVIDIA/CUDA |

**Dùng Python 3.10** theo môi trường dự án. Kiểm tra bằng `python --version`.
Các lệnh bên dưới được chạy từ thư mục `YoFaSoShop_AIService` trong repo đã clone.
Nếu terminal đang đứng ở thư mục gốc repo, chạy:

```bash
cd YoFaSoShop_AIService
```

Không cần tạo ứng dụng FastAPI mới: mã nguồn đã có tại [api/main.py](api/main.py),
các endpoint được đăng ký tại [api/application.py](api/application.py).

## 2. Cài và chạy FastAPI trên laptop

### Bước 1 — Tạo môi trường và cài thư viện

**Windows / PowerShell:**

```powershell
python -m venv .venv-api
.\.venv-api\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-api\Scripts\python.exe scripts/init_env.py
```

**Linux / macOS:**

```bash
python3.10 -m venv .venv-api
./.venv-api/bin/python -m pip install -r requirements-dev.txt
./.venv-api/bin/python scripts/init_env.py
```

`requirements-dev.txt` cài FastAPI cùng thư viện kiểm thử. `init_env.py` tạo `.env`
từ `.env.example`, sinh khóa nội bộ ngẫu nhiên và **giữ nguyên file nếu đã tồn tại**.
Không đưa `.env` thật lên GitHub.

<details>
<summary>Cách gọn hơn trên Windows</summary>

Script sau tạo venv, cài thư viện và tạo `.env` nếu chưa có:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1
```

Nếu cần chọn đúng Python 3.10, thêm `-Python` với đường dẫn Python thật trên máy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1 -Python "C:\duong-dan-python310\python.exe"
```

</details>

### Bước 2 — Kiểm tra `.env`

Mở `.env`, giữ khóa `AI_INTERNAL_API_KEY` đã tạo và kiểm tra:

```dotenv
APP_ENV=development
AI_MODE=mock
EMBEDDED_WORKER=true
EXPOSE_DOCS=true
POSE_WEIGHTS=./weights/yolo26s-pose.pt
```

Nếu từng chạy chế độ thật, tự đổi lại các dòng trên: script khởi tạo không ghi đè
cấu hình cũ. Mock chưa cần có file pose/model; worker giả lập chạy cùng ứng dụng.

### Bước 3 — Khởi động FastAPI

**Windows:**

```powershell
.\.venv-api\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Linux / macOS:**

```bash
./.venv-api/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Giữ terminal mở. Nhấn `Ctrl+C` khi muốn dừng server.

### Bước 4 — Mở Swagger và thử API

1. Mở [Swagger UI](http://127.0.0.1:8000/docs).
2. Nhấn **Authorize**.
3. Lấy giá trị `AI_INTERNAL_API_KEY` trong `.env`, dán vào ô API key rồi xác nhận.
4. Mở `GET /internal/v1/capabilities` → **Try it out** → **Execute**.
5. Mở [health/live](http://127.0.0.1:8000/health/live) để kiểm tra server đang sống.

Các endpoint nội bộ dùng header `X-Internal-API-Key`. Hướng dẫn Postman và các
request cụ thể nằm trong [apiTestingGuide.md](../YoFaSoShop_Docs/extra-info/apiTestingGuide.md).

> **Kết quả mock là giả lập.** Output có `simulated=true`; ảnh thử đồ là placeholder
> có chữ MOCK. Luồng này kiểm tra API/hàng đợi, chưa xác nhận chất lượng ảnh hay tốc độ GPU.

## 3. Tải code và checkpoint OOTDiffusion

### 3.1. Phân biệt các script

| Script trong `scripts/`     | Công việc                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `download_ootdiffusion.py`  | Tải bộ code chạy, preprocessing và checkpoint từ Hugging Face Space                                      |
| `download_ootdiffusion.ps1` | Gọi script Python trên Windows, chuyển tiếp các tham số                                                 |
| `download_ootdiffusion.sh`  | Gọi script Python trên Linux/Bash, chuyển tiếp các tham số                                              |
| `download_assets.py`        | Chỉ tải checkpoint OOTDiffusion và weight DeepFashion2 theo `.env`; yêu cầu code OOTDiffusion đã có |

Bản `.ps1`/`.sh` gọi `python` của terminal hiện tại. Nếu có nhiều Python, dùng
trực tiếp Python trong venv như bên dưới để tránh chạy nhầm môi trường.

Script yêu cầu ít nhất **20 GiB trống** trước khi tải; cần dự trù thêm chỗ cho Python, weights khác, Docker image và kết quả.

### 3.2. Tải trên Windows

Sau khi tạo venv ở mục 2:

```powershell
# Cài công cụ tải, chưa cài môi trường chạy model GPU
.\.venv-api\Scripts\python.exe -m pip install huggingface-hub==0.25.2

# Chỉ xem kế hoạch, không tải hoặc ghi dữ liệu
.\.venv-api\Scripts\python.exe scripts/download_ootdiffusion.py --dry-run

# Bắt đầu tải code và checkpoint
.\.venv-api\Scripts\python.exe scripts/download_ootdiffusion.py
```

Nếu đã kích hoạt đúng venv và `python` trỏ tới venv đó, có thể dùng bản PowerShell:

```powershell
.\scripts\download_ootdiffusion.ps1 -DryRun
.\scripts\download_ootdiffusion.ps1
```

### 3.3. Tải trên máy GPU Linux

Sau khi tạo và kích hoạt venv GPU ở mục 5.1:

```bash
python -m pip install huggingface-hub==0.25.2
python scripts/download_ootdiffusion.py --dry-run
python scripts/download_ootdiffusion.py
```

Hoặc dùng `bash scripts/download_ootdiffusion.sh`. Khi tải gián đoạn, chạy lại
cùng lệnh để công cụ tận dụng dữ liệu/cache đã tải.

### 3.4. Kết quả tải và cấu hình nâng cao

Nguồn mặc định: [Hugging Face Space · levihsu/OOTDiffusion](https://huggingface.co/spaces/levihsu/OOTDiffusion).
Script chọn các thành phần sau, không sao chép mọi file trong Space:

```text
api/models/ootdiffusion/
├── checkpoints/       # Trọng số model và tài nguyên liên quan
├── ootd/              # Code inference
├── preprocess/        # Code tiền xử lý
├── run/               # Các file được chọn từ run/*.py
├── requirements.txt   # Môi trường upstream để tham khảo
└── README.md          # Tài liệu upstream
```

**Tải xong chưa có nghĩa là model đã chạy được.** Script không tự cài dependencies,
kiểm tra CUDA hoặc chạy inference. Môi trường tích hợp của dự án nằm trong
`requirements-gpu.txt` và `Dockerfile.gpu`; tránh cài chồng tùy ý requirements upstream.

<details>
<summary>Tải sang ổ khác, cố định revision và dùng token</summary>

Ví dụ tải sang ổ khác trên Windows:

```powershell
.\.venv-api\Scripts\python.exe scripts/download_ootdiffusion.py --local-dir "E:\AI-models\ootdiffusion"
```

Sau đó sửa `.env` để API/worker tìm đúng thư mục:

```dotenv
OOTD_ROOT=E:/AI-models/ootdiffusion
```

Downloader này dùng `--local-dir`, **không tự đọc `OOTD_ROOT` trong `.env`**.
Revision mặc định là `main`; để cố định phiên bản, chạy bằng Python của venv:

```bash
python scripts/download_ootdiffusion.py --revision <commit-hoac-tag>
```

Thay `<commit-hoac-tag>` bằng giá trị thật trước khi chạy. Sau khi kiểm thử GPU,
nên ghi lại commit đã nghiệm thu và đặt `OOTD_REVISION` trong `.env` cùng giá trị
để `download_assets.py` tải checkpoint tương ứng. Hiện chưa chốt revision đã nghiệm thu GPU.

Nguồn công khai có thể tải khi chưa có token. Nếu cần xác thực, giữ `HF_TOKEN`
trong `.env` và chạy downloader qua lệnh nạp `.env` dưới đây (dùng Python của venv,
đứng trong thư mục AI Service):

```bash
python -c "from dotenv import load_dotenv; import runpy; load_dotenv('.env'); runpy.run_path('scripts/download_ootdiffusion.py', run_name='__main__')"
```

Bản downloader không tự nạp `.env`; nó chỉ đọc token từ biến môi trường hoặc dùng
thông tin đăng nhập Hugging Face đã lưu. Không ghi token vào README/lệnh chia sẻ.

Không tải đè lên thư mục upstream đã tự sửa code nếu chưa sao lưu phần sửa.
Tải từ nguồn không phục hồi các chỉnh sửa chỉ có trên máy bạn.

</details>

## 4. Chuẩn bị weights riêng của dự án

**Tải OOTDiffusion không đồng thời tải các weight riêng trong `weights/`.**

| File cần có                           | Vai trò                                                                |
| --------------------------------------- | ----------------------------------------------------------------------- |
| `weights/deepfashion2_yolov8s-seg.pt` | Phát hiện và phân vùng trang phục                                 |
| `weights/yolo26s-pose.pt`             | Phát hiện người và các điểm cơ thể theo lựa chọn hiện tại |

Nếu chủ repo đã lưu chúng bằng Git LFS, cài Git LFS rồi chạy trong repo đã clone:

```bash
git lfs install --local
git lfs pull
git lfs ls-files
```

Lệnh này chỉ tải file LFS thực sự đã được đưa lên. Nếu không có trên LFS, lấy từ
bản phân phối của chủ dự án rồi đặt đúng đường dẫn.

Khi code OOTDiffusion đã có và đã cài môi trường GPU, có thể tải checkpoint và
DeepFashion2 theo nguồn/revision trong `.env` bằng:

```bash
python scripts/download_assets.py
```

Script này **không tải YOLO26 pose**. Hai weight ResNet50 chưa được pipeline API
hiện tại nạp.

## 5. Chạy model thật trên máy GPU

> **Chưa nghiệm thu GPU thật.** Đây là quy trình theo code hiện tại; môi trường thư
> viện là cấu hình ứng viên. Tương thích YOLO26, OOTDiffusion, driver và VRAM phải
> được xác nhận bằng inference thật, không chỉ bằng việc tải đủ file.

### 5.1. Tạo môi trường Linux

Code AI Service phải có trên máy thuê. Mở terminal trong thư mục đó:

```bash
nvidia-smi
python3.10 --version
df -h .
python3.10 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
```

Nếu template thiếu Python 3.10 hoặc `venv`, cài thành phần còn thiếu theo hệ điều
hành/template trước. Mỗi terminal mới cần kích hoạt lại `.venv-gpu`.

Bộ PyTorch sau khớp với `Dockerfile.gpu` hiện tại; cần driver hỗ trợ bản CUDA này:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-gpu.txt
python -m pip check
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python scripts/init_env.py
```

Phải thấy `CUDA available: True` trước khi chạy VTON thật. Dùng venv riêng giúp tránh
trộn với PyTorch có sẵn trong template. Tải model/weights theo mục 3–4 nếu chưa có.

### 5.2. Chạy API và worker ở hai terminal

Cả hai terminal đều đứng trong `YoFaSoShop_AIService`.

**Terminal 1 — API nhận request:**

```bash
source .venv-gpu/bin/activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

**Terminal 2 — Worker xử lý job:**

```bash
source .venv-gpu/bin/activate
python -m workers.runner
```

API và worker phải đọc cùng `.env` và dùng chung `RUNTIME_DIR`. Queue dùng JSON manifest nguyên tử, không cần database hay migration.
Ảnh vẫn cần **cùng volume runtime**; không đặt API và worker lên hai host chia sẻ NFS.
Worker giữ khóa OS trên volume, một worker cho runtime. Redis/Celery chưa phải dependency.

```text
Upload ảnh → Nhận assetId → Tạo job → Nhận 202 + jobId
                                         ↓
                              Worker lấy job từ queue
                                         ↓
                              Model → Hậu kiểm → Lưu ảnh
                                         ↓
                         Gọi API trạng thái để lấy kết quả
```

Worker xử lý lần lượt, cache model trong tiến trình con và dừng tiến trình con khi
quá thời gian. Routing: `upper_body` → OOTDiffusion HD;
`lower_body` / `dress` → OOTDiffusion DC. Không dùng IDM-VTON.

### 5.3. Truy cập từ laptop

Dùng SSH tunnel để mở API máy thuê trên laptop trong khi server vẫn bind
`127.0.0.1`. Xem [setupGuide.md](../YoFaSoShop_Docs/extra-info/setupGuide.md)
về SSH, `scripts/connect_gpu.ps1` và cấu hình cổng. Nếu tunnel dùng cổng laptop
`18000`, mở `http://127.0.0.1:18000/docs`.

## 6. Chạy bằng Docker

Docker đóng gói code và dependencies. Model/weights/cache/runtime được gắn từ thư
mục máy chủ vào container, không nhét checkpoint vào image. Chuẩn bị `.env` và
model/weights trước khi chạy chế độ thật.

**Mock**, với `.env` ở mục 2:

```bash
docker compose up -d --build ai-api
docker compose logs -f ai-api
```

**API + worker thật**, trên máy có Docker Engine, NVIDIA driver và khả năng cấp GPU
cho container:

```bash
docker compose -f compose.yaml -f compose.real.yaml --profile gpu up -d --build
docker compose -f compose.yaml -f compose.real.yaml --profile gpu logs -f ai-api gpu-worker
```

Các biến `HOST_OOTD_DIR`, `HOST_WEIGHTS_DIR`, `HOST_RUNTIME_DIR`, `HOST_HF_CACHE_DIR`
chọn thư mục trên máy chủ. Nếu tải OOTDiffusion sang ổ khác, đổi `HOST_OOTD_DIR`
cho Compose; `OOTD_ROOT` dùng khi chạy Python trực tiếp. Compose tự đặt đường dẫn
bên trong container.

Dừng mock bằng `docker compose down`. Dừng bộ chạy thật bằng:

```bash
docker compose -f compose.yaml -f compose.real.yaml --profile gpu down
```

Giữ thư mục dữ liệu trên máy chủ để dùng lại. Template Vast dạng container không
mặc định hỗ trợ chạy thêm Docker bên trong; dùng Python trực tiếp ở mục 5.
VM có Docker/GPU runtime có thể dùng Compose. Hiện mới kiểm tra cấu hình Compose,
chưa xác nhận build image hoặc inference GPU thành công.

## 7. Kiểm thử API

| Đường dẫn                     | Công dụng                                                            |
| --------------------------------- | ---------------------------------------------------------------------- |
| `GET /health/live`              | Kiểm tra tiến trình API đang sống                                 |
| `GET /health/ready`             | Kiểm tra điều kiện cơ bản, heartbeat worker và file cần thiết |
| `GET /internal/v1/config-check` | Kiểm tra cấu hình ứng dụng                                        |
| `GET /internal/v1/models`       | Kiểm tra sự hiện diện của các file model được khai báo       |
| `/docs`                         | Thử endpoint qua Swagger                                              |

**Health ready hoặc file hiện diện chưa chứng minh inference chạy thành công.**
Để nghiệm thu model thật, upload ảnh, tạo job, theo dõi đến trạng thái kết thúc và
mở ảnh kết quả. Các tài liệu để tự kiểm tra:

- [Hướng dẫn Postman từng endpoint](../YoFaSoShop_Docs/extra-info/apiTestingGuide.md)
- [Postman collection](postman/YoFaSoShop_AI.postman_collection.json)
- [Environment mẫu](postman/YoFaSoShop_local.postman_environment.json)
- [Hợp đồng OpenAPI](contracts/openapi.json)
- [Báo cáo kiểm thử và giới hạn xác minh](reports/verification.md)

Sau khi cài `requirements-dev.txt`, kích hoạt `.venv-api` rồi chạy:

```bash
python -m pytest -q
python scripts/http_smoke.py
python -m pip check
```

Trên Windows kích hoạt bằng `.\.venv-api\Scripts\Activate.ps1`; trên Linux/macOS
là `source .venv-api/bin/activate`. Nếu PowerShell chặn kích hoạt, dùng đường dẫn
`.\.venv-api\Scripts\python.exe` thay `python` trong từng lệnh.

`http_smoke.py` tự chạy server mock riêng, không phải benchmark GPU.
Khi chủ động cập nhật hợp đồng, dùng `python scripts/export_contract.py` rồi kiểm
tra diff OpenAPI/Postman. Các hàm xử lý ảnh nằm trong `api/vision/`;
`scripts/extract_notebook_primitives.py` chỉ dùng khi chủ động đồng bộ từ notebook
và chạy regression, không execute notebook theo từng request.

## 8. Cấu trúc thư mục và quản lý tài nguyên

```text
YoFaSoShop_AIService/
├── api/
│   ├── main.py                 # Điểm khởi động FastAPI
│   ├── application.py          # Các endpoint
│   ├── core/                   # Cấu hình, lỗi và queue
│   ├── services/               # Xử lý ảnh, routing, điều phối model
│   ├── vision/                 # Hàm xử lý ảnh tách từ notebook
│   └── models/ootdiffusion/    # Code/checkpoint tải riêng khi cần
├── workers/                    # Tiến trình xử lý job
├── weights/                    # DeepFashion2, YOLO pose và tài nguyên khác
├── scripts/                    # Thiết lập, tải model, kiểm thử, tiện ích
├── tests/                      # Kiểm thử tự động
├── postman/                    # Collection và environment mẫu
├── contracts/                  # Thiết kế/hợp đồng API
├── data/runtime/               # Queue database, ảnh upload và kết quả phát sinh
├── .env.example                # Cấu hình mẫu có thể chia sẻ
├── .env                        # Cấu hình thật từng máy, không commit
└── compose.yaml                # API và profile worker GPU
```

`.gitignore` ngăn dữ liệu lớn, runtime, secrets và checkpoint tải lại được đi vào
Git. **Bị ignore không có nghĩa là không cần khi chạy.** Sao lưu riêng weight tự
huấn luyện hoặc chỉnh sửa không tải lại được.

`.gitattributes` cấu hình LFS cho các định dạng model; `.gitignore` vẫn quyết định
file nào được phép add. LFS không tự thêm file. OOTDiffusion checkpoint được tải
riêng thay vì mặc định upload lại lên GitHub. Xem
[giới hạn Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

README này không tự thay đổi `.gitignore`: muốn bỏ **toàn bộ** OOTDiffusion khỏi
repo thì cần quy tắc tương ứng; chỉ bỏ `checkpoints/` vẫn cho phép theo dõi code.
Thêm ignore cũng không tự bỏ theo dõi file đã commit trước đó.

## 9. Xử lý lỗi thường gặp

| Hiện tượng                                   | Kiểm tra / xử lý                                                                                                   |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Thiếu `fastapi` hoặc `huggingface_hub`    | Dùng cùng Python của venv để cài và chạy                                                                      |
| Không tạo được venv                        | Kiểm tra Python 3.10 và thành phần `venv` của hệ điều hành                                                 |
| Đã init nhưng ứng dụng vẫn chạy real     | `.env` cũ được giữ nguyên; sửa `AI_MODE`/`EMBEDDED_WORKER` rồi khởi động lại                        |
| API trả `401`                                | Nhập đúng `AI_INTERNAL_API_KEY` của server đang chạy                                                          |
| Cổng `8000` đang được dùng              | Dừng server cũ hoặc đổi `--port` và URL test tương ứng                                                     |
| Downloader báo dưới 20 GiB trống            | Giải phóng ổ hoặc dùng `--local-dir`; `--force-low-disk` chỉ bỏ kiểm tra, không tạo dung lượng        |
| Tải gián đoạn                               | Chạy lại cùng nguồn/revision/thư mục, kiểm tra mạng và dung lượng                                          |
| `download_assets.py` báo thiếu code         | Tải bằng `download_ootdiffusion.py` trước; kiểm tra `OOTD_ROOT`                                              |
| `MODEL_NOT_READY`                             | Kiểm tra `.env`, mount Docker và `checkpoints/ootd/model_index.json`                                            |
| `CUDA available: False` / `GPU_UNAVAILABLE` | Kiểm tra `nvidia-smi`, PyTorch, driver và quyền cấp GPU cho container                                           |
| Không nạp được YOLO26                      | Kiểm tra file thật thay vì con trỏ LFS và độ tương thích phiên bản Ultralytics; cần test model thực tế |
| Job mãi `QUEUED`                             | Kiểm tra worker, log và việc dùng chung `.env`/`RUNTIME_DIR`                                                  |
| Hết VRAM / inference thất bại                | Đọc mã lỗi/log, kiểm tra GPU và tham số; hệ thống không tự fallback mock để báo thành công            |
| Docker không kết nối được daemon          | Khởi động Docker Engine/Desktop hoặc dùng Python trực tiếp trên template container                            |
