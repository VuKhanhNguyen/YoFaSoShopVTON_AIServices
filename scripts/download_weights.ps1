# ==============================================================================
# Script tự động tải các trọng số mô hình cho SmartClosetAI (Windows PowerShell)
# Đặt trong thư mục: ml-service/scripts/
# ==============================================================================

Write-Host "=== [1/4] Tạo cấu trúc thư mục chứa trọng số mô hình trong YoFaSoShop_AIService ===" -ForegroundColor Cyan
$baseDir = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path "$baseDir\api\models\idm-vton\ckpt\densepose" | Out-Null
New-Item -ItemType Directory -Force -Path "$baseDir\api\models\idm-vton\ckpt\humanparsing" | Out-Null
New-Item -ItemType Directory -Force -Path "$baseDir\api\models\idm-vton\ckpt\openpose\ckpts" | Out-Null
New-Item -ItemType Directory -Force -Path "$baseDir\weights" | Out-Null

Write-Host "=== [2/4] Tải trọng số DensePose (Meta Detectron2 - 255MB) ===" -ForegroundColor Cyan
$densePoseUrl = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
$densePosePath = "$baseDir\api\models\idm-vton\ckpt\densepose\model_final_162be9.pkl"
if (-not (Test-Path $densePosePath)) {
    Write-Host "Đang tải $densePoseUrl..."
    Invoke-WebRequest -Uri $densePoseUrl -OutFile $densePosePath
} else {
    Write-Host "DensePose checkpoint đã tồn tại, bỏ qua." -ForegroundColor Yellow
}

Write-Host "=== [3/4] Tải trọng số Human Parsing & OpenPose (IDM-VTON Preprocessing) ===" -ForegroundColor Cyan
$parsingUrl = "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx"
$parsingPath = "$baseDir\api\models\idm-vton\ckpt\humanparsing\parsing_atr.onnx"
if (-not (Test-Path $parsingPath)) {
    Write-Host "Đang tải $parsingUrl..."
    Invoke-WebRequest -Uri $parsingUrl -OutFile $parsingPath
} else {
    Write-Host "parsing_atr.onnx đã tồn tại, bỏ qua." -ForegroundColor Yellow
}

$openposeUrl = "https://huggingface.co/yisol/IDM-VTON/resolve/main/openpose/ckpts/body_pose_model.pth"
$openposePath = "$baseDir\api\models\idm-vton\ckpt\openpose\ckpts\body_pose_model.pth"
if (-not (Test-Path $openposePath)) {
    Write-Host "Đang tải $openposeUrl..."
    Invoke-WebRequest -Uri $openposeUrl -OutFile $openposePath
} else {
    Write-Host "body_pose_model.pth đã tồn tại, bỏ qua." -ForegroundColor Yellow
}

Write-Host "=== [4/4] Kiểm tra DeepFashion2 & IDM-VTON ===" -ForegroundColor Cyan
$df2Path = "$baseDir\weights\deepfashion2_yolov8s-seg.pt"
if (Test-Path $df2Path) {
    Write-Host "Đã phát hiện DeepFashion2 weights: $df2Path (Sẵn sàng sử dụng!)" -ForegroundColor Green
} else {
    Write-Host "Chưa có deepfashion2_yolov8s-seg.pt trong ml-service\weights\" -ForegroundColor Yellow
}

Write-Host "IDM-VTON chính nạp qua Hugging Face Diffusers (yisol/IDM-VTON)"
Write-Host "Hoặc git clone vào: $baseDir\models\idm-vton\ckpt"

Write-Host "=== Hoàn tất chuẩn bị và tải trọng số trong ml-service! ===" -ForegroundColor Green
