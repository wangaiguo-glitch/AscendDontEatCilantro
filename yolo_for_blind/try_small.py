import os
import time
import cv2
import numpy as np
import mindspore_lite as mslite

# ---------------- 配置 ----------------
MODEL_PATH = "/home/HwHiAiUser/work/hejunhao/yolo_for_blind/midas_small.mindir"
IMG_PATH = "/home/HwHiAiUser/work/hejunhao/yolo_for_blind/1.jpg"  # 输入图像路径
OUTPUT_PATH = "/home/HwHiAiUser/work/hejunhao/yolo_for_blind/output.jpg"  # 推理结果保存路径
IMG_SIZE = 256  # 模型输入尺寸

# ---------------- 检查文件 ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"MindIR 模型不存在: {MODEL_PATH}")
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"输入图像不存在: {IMG_PATH}")

# ---------------- 创建上下文 ----------------
context = mslite.Context()  # ***
context.target = ["CPU"]  # ***
# context.ascend.device_id = 0
# context.ascend.rank_id = 0
context.ascend.provider = "ge"

# ---------------- 加载模型 ----------------
model = mslite.Model()  # ***
model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR, context)
print(f"模型加载成功: {MODEL_PATH}\n")

# ---------------- 查看输入输出 ----------------
inputs = model.get_inputs()  # ***
outputs = model.get_outputs()  # ***

print("输入信息:")
for i, t in enumerate(inputs):
    print(f"  输入[{i}] name={t.name} shape={t.shape}")

print("输出信息:")
for i, t in enumerate(outputs):
    print(f"  输出[{i}] name={t.name} shape={t.shape}")

# ---------------- 图像预处理 ----------------
img = cv2.imread(IMG_PATH)
h_ori, w_ori = img.shape[:2]

r = IMG_SIZE / max(h_ori, w_ori)
if r != 1:
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    img_resized = cv2.resize(img, (int(w_ori*r), int(h_ori*r)), interpolation=interp)
else:
    img_resized = img.copy()

h, w = img_resized.shape[:2]
if h < IMG_SIZE or w < IMG_SIZE:
    dh, dw = (IMG_SIZE - h) / 2, (IMG_SIZE - w) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
else:
    img_padded = img_resized

# HWC -> CHW, BGR->RGB, 归一化
img_input = img_padded[:, :, ::-1].transpose(2, 0, 1) / 255.0
img_input = np.ascontiguousarray(img_input[None].astype(np.float32))  # shape [1,3,256,256]

# ---------------- 推理 ----------------
_in = model.get_inputs()
_in[0].set_data_from_numpy(img_input)

t0 = time.time()
_out = model.predict(_in)
infer_time = (time.time() - t0) * 1000  # ms

out_numpy = _out[0].get_data_to_numpy().copy()  # shape [1,256,256]

print(f"推理完成, 输出 shape: {out_numpy.shape}, 耗时: {infer_time:.2f} ms")


# ---------------- 恢复到原图尺寸 ----------------
out_img = np.squeeze(out_numpy)  # [256,256]
out_resized = cv2.resize(out_img, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)

# 归一化到 0~255
out_resized = ((out_resized - out_resized.min()) / (out_resized.max() - out_resized.min()) * 255.0)# .astype(np.uint8)

cv2.imwrite(OUTPUT_PATH, out_resized)
print(f"推理完成, 输出 shape: {out_resized.shape}, 耗时: {infer_time:.2f} ms, 保存到: {OUTPUT_PATH}")


# ---------------- 保存为 CSV ----------------
np.savetxt('/home/HwHiAiUser/work/hejunhao/dat.csv', out_resized, delimiter=",", fmt="%.6f")
print(f"推理完成, 耗时: {infer_time:.2f} ms, CSV保存到: {'/home/HwHiAiUser/work/hejunhao/dat.csv'}")