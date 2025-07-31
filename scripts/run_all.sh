#!/bin/bash
set -e

echo "Step 1: 生成合成数据..."
python src/synthetic_data.py

echo "Step 2: 离线训练模型..."
CUDA_VISIBLE_DEVICES=0 python src/train.py

echo "Step 3: 在线推理..."
CUDA_VISIBLE_DEVICES=0 python src/inference.py

echo "All steps finished successfully!"
