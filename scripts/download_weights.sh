#!/usr/bin/env bash
set -e

# ==============================================================================
# Script tự động tải các trọng số mô hình cho SmartClosetAI (Linux / Colab / Bash)
# Đặt trong thư mục: ml-service/scripts/
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== [1/4] Tạo cấu trúc thư mục chứa trọng số mô hình trong ml-service ==="
mkdir -p "$BASE_DIR/models/idm-vton/ckpt/densepose"
mkdir -p "$BASE_DIR/models/idm-vton/ckpt/humanparsing"
mkdir -p "$BASE_DIR/models/idm-vton/ckpt/openpose/ckpts"
mkdir -p "$BASE_DIR/weights"

echo "=== [2/4] Tải trọng số DensePose (Meta Detectron2 - 255MB) ==="
DENSEPOSE_PATH="$BASE_DIR/models/idm-vton/ckpt/densepose/model_final_162be9.pkl"
if [ ! -f "$DENSEPOSE_PATH" ]; then
    curl -L -o "$DENSEPOSE_PATH" \
      https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
else
    echo "DensePose checkpoint đã tồn tại, bỏ qua."
fi

echo "=== [3/4] Tải trọng số Human Parsing & OpenPose (IDM-VTON Preprocessing) ==="
PARSING_PATH="$BASE_DIR/models/idm-vton/ckpt/humanparsing/parsing_atr.onnx"
if [ ! -f "$PARSING_PATH" ]; then
    curl -L -o "$PARSING_PATH" \
      https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx
else
    echo "parsing_atr.onnx đã tồn tại, bỏ qua."
fi

OPENPOSE_PATH="$BASE_DIR/models/idm-vton/ckpt/openpose/ckpts/body_pose_model.pth"
if [ ! -f "$OPENPOSE_PATH" ]; then
    curl -L -o "$OPENPOSE_PATH" \
      https://huggingface.co/yisol/IDM-VTON/resolve/main/openpose/ckpts/body_pose_model.pth
else
    echo "body_pose_model.pth đã tồn tại, bỏ qua."
fi

echo "=== [4/4] Kiểm tra DeepFashion2 & IDM-VTON ==="
DF2_PATH="$BASE_DIR/weights/deepfashion2_yolov8s-seg.pt"
if [ -f "$DF2_PATH" ]; then
    echo "Đã phát hiện DeepFashion2 weights: $DF2_PATH (Sẵn sàng sử dụng!)"
else
    echo "Chưa có deepfashion2_yolov8s-seg.pt trong ml-service/weights/"
fi

echo "IDM-VTON chính nạp qua Hugging Face Diffusers (yisol/IDM-VTON)."
echo "=== Hoàn thành chuẩn bị thư mục và tải trọng số trong ml-service thành công! ==="
