# 💡 YOLO For Blind

## 盲途智导——基于华为昇腾AI的视障实时感知与语音导航系统
👨‍💻 作者：何俊豪 魏震东 张嘉蓉

📅 版本：v1.0
> 基于 MindSpore & 昇腾 CANN 的轻量级盲人辅助视觉系统  
> —— 实现实时目标检测与语音播报，助力无障碍出行  

---

## 📖 项目简介

**YOLO For Blind** 是一个基于 **MindSpore 2.5.0** 与 **MindSpore Lite 2.5.0** 构建的视觉辅助系统，专为视障人士设计，提供 **实时目标检测与语音播报功能**。  
系统采用 **YOLOv8s** 模型，并在特定场景数据集上进行了微调，可高效运行于 **昇腾 CANN 8.0 RC3 架构** 的硬件平台（如 **香橙派 Ascend** 等），实现边缘侧智能识别与实时语音提示。

> 🔗 **项目参考**: 本项目基于 [MindYOLO](https://github.com/mindspore-lab/mindyolo) 框架开发

---

## ⚙️ 环境依赖

| 组件 | 版本 |
|------|------|
| MindSpore | 2.5.0 |
| MindSpore Lite | 2.5.0 |
| CANN | 8.0 RC3 |
| Python | 3.10 |
| 硬件平台 | 华为昇腾 901B / 香橙派 Ascend 等 |

### 环境配置

1. **安装核心框架**（建议在华为官方网站安装）：
   ```bash
   # 请从华为官方渠道安装 MindSpore 和 MindSpore Lite
   # mindspore==2.5.0
   # mindspore-lite==2.5.0
   ```

2. **安装项目依赖**：
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 模型说明

- **模型结构**：YOLOv8s  
- **预训练权重**：`yolov8-s_500e_mAP446-3086f0c9.ckpt`
- **训练数据集**：项目专用盲人场景数据集（包含日常物体、交通标志、行人、障碍物等类别）
  
  **数据集链接**：  
  [A_dataset_for_the_visually_impaired_walk_on_the_road](https://d1wqtxts1xzle7.cloudfront.net/111632905/A_dataset_for_the_visually_impaired_walk_on_the_road-libre.pdf?1708385152=&response-content-disposition=inline%3B+filename%3DA_dataset_for_the_visually_impaired_walk.pdf&Expires=1762003912&Signature=X6ci~Z0YmDZUBxvAy8hvudXDOhnBEGVziC2rN1F4Mw3udvk6dJSuhsu2RObo0A3AI6EZyP6ch-QPmZoI3va5jLS5WvDy9GkEDLPMGZ13kdOWfr5LbJ9bVWruXY3DzAWNTeiBmlK~Xzp0fwcfjpZGJE~veGGubO13UtGdxbxpxkymaNcJYEOhtBzQne1VS8-FFnb-8bXxSxfzHY~EtmlKaFlo3ojo5SyfJA2mRbqKasYV29YrdRA3LzgLLlEdANeW8swYjd9xgZHNU3XSVBfpdLIJrkZs3cPpoX8exKdS3eCH65y9M-PVUVevNTrCjaO2~0AAAcErD0oO-cqjPMqRZQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)

- **优化方向**：轻量化、高实时性、语音播报兼容  

---

## 🚀 使用指南

### 1️⃣ 环境配置

```bash
# 安装项目依赖
pip install -r requirements.txt
```

### 2️⃣ 模型训练

1. **下载预训练权重**：
   ```bash
   # 将权重文件 yolov8-s_500e_mAP446-3086f0c9.ckpt 置于 weights/ 目录下
   ```

2. **准备数据集**：配置数据集路径于相应的 `.yaml` 配置文件中。

3. **启动训练**（昇腾 901B 环境）：
   ```bash
   python train.py \
       --config /home/hejunhao/yolo_for_blind/configs/yolov8/yolov8s_wotr.yaml \
       --device_target Ascend \
       --weight /home/hejunhao/yolo_for_blind/weights/yolov8-s_500e_mAP446-3086f0c9.ckpt \
       --log_level "ERROR" \
       --strict_load False > train.log 2>&1 &
   ```

   训练日志将输出至 `train.log`，可使用以下命令实时查看：
   ```bash
   tail -f train.log
   ```

---

### 3️⃣ 模型转换：ckpt → mindir

训练完成后，将生成的权重文件转换为 **MINDIR** 格式以支持推理部署：

```bash
python export.py \
    --config /home/hejunhao/yolo_for_blind/configs/yolov8/yolov8s.yaml \
    --weight /home/hejunhao/yolo_for_blind/yolov8s_wotr-500_564.ckpt \
    --file_format MINDIR \
    --device_target Ascend
```

转换完成后，将生成类似以下格式的文件：
```bash
yolov8s_wotr.mindir
```

---

### 4️⃣ 模型推理与语音播报

支持实时摄像头输入与语音提示功能：

```bash
python /home/hejunhao/yolo_for_blind/blind_assist.py \
    --mindir_path /home/hejunhao/yolo_for_blind/yolov8s_wotr.mindir \
    --image_path /dev/video0 \
    --flask_port 5000 \
    --save_result False \
    --config /home/hejunhao/yolo_for_blind/configs/yolov8/yolov8s_wotr.yaml
```


## 🔗 相关链接

- [MindYOLO 原项目仓库](https://github.com/mindspore-lab/mindyolo)

---

**YOLO For Blind** —— 用技术点亮黑暗，让出行更安心 🌟
