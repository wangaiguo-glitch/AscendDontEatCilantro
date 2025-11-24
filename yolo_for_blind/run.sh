#!/bin/bash

# 激活 conda 环境
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate Mindspore

# 设置 PYTHONPATH
export PYTHONPATH=/home/HwHiAiUser/work/hejunhao:$PYTHONPATH

# 运行 YOLO 推理服务（5000 端口）
python /home/HwHiAiUser/work/hejunhao/yolo_for_blind/predict.py \
    --mindir_path /home/HwHiAiUser/work/hejunhao/yolo_for_blind/yolov8s_wotr.mindir \
    --image_path /dev/video0 \
    --flask_port 5000 \
    --save_result False \
    --config /home/HwHiAiUser/work/hejunhao/yolo_for_blind/configs/yolov8/yolov8s_wotr.yaml
