import cv2
import time
import numpy as np
import mindspore_lite as mslite
# import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend
# import matplotlib.pyplot as plt
# -----------------------------------------------------
# 单帧深度图推理（模仿 YOLO detect_once）
# -----------------------------------------------------
def depth_once(model, img: np.ndarray, img_size: int = 256):
    """
    MiDaS depth estimation for single frame.
    Return:
        depth_map: np.ndarray (H,W)  - normalized depth map, 0~255 float32
        infer_time: float (milliseconds)
    """
    h_ori, w_ori = img.shape[:2]

    # ----------- resize & padding (和 detect_once 同结构) -----------
    r = img_size / max(h_ori, w_ori)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img_resized = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    else:
        img_resized = img.copy()

    h, w = img_resized.shape[:2]

    if h < img_size or w < img_size:
        dh, dw = (img_size - h) / 2, (img_size - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    else:
        img_padded = img_resized

    # ----------- preprocess -----------
    img_input = img_padded[:, :, ::-1].transpose(2, 0, 1) / 255.0
    img_input = np.ascontiguousarray(img_input[None].astype(np.float32))  # shape = [1,3,256,256]

    # ----------- inference -----------
    # t0 = time.time()

    inputs = model.get_inputs()
    inputs[0].set_data_from_numpy(img_input)
    outputs = model.predict(inputs)

    # infer_time = (time.time() - t0) * 1000  # ms

    out_numpy = outputs[0].get_data_to_numpy().copy()  # [1,256,256]
    depth_small = np.squeeze(out_numpy)

    if h < img_size or w < img_size:
        depth_no_pad = depth_small[top:h+top, left:w+left]
    else:
        depth_no_pad = depth_small
    # ----------- resize back to original size -----------
    depth_map = cv2.resize(depth_no_pad, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)

    # normalize to 0~255 float
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    depth_map = depth_map * 255.0  # keep float32, NO uint8

    return depth_map.astype(np.float32)


# -----------------------------------------------------
# 深度图可视化（彩色热力图）
# -----------------------------------------------------
# def draw_depth(depth_map: np.ndarray):
#     """
#     Visualize depth as heatmap (jet colormap).
#     Input:
#         img: 原图
#         depth_map: [H,W] float32, 0~255
#     Return:
#         depth_color: 彩色热力图
#     """
#     depth_uint8 = depth_map.astype(np.uint8)
#     depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

#     # (可选) 融合原图
#     # overlay = cv2.addWeighted(img, 0.4, depth_color, 0.6, 0)

#     return depth_color

def draw_depth(depth_map: np.ndarray):
    """
    Visualize depth as heatmap (jet colormap) + colorbar with numeric ticks.
    depth_map: [H,W] float32, range 0~255
    return: depth image with colorbar
    """

    depth_uint8 = depth_map.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    H, W = depth_color.shape[:2]
    bar_width = max(40, W // 12)    # colorbar 宽度

    # -----------------------------
    #  创建竖向 colorbar（0~255）
    # -----------------------------
    gradient = np.linspace(0, 255, H, dtype=np.uint8).reshape(H, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

    # -----------------------------
    #  在 colorbar 右侧添加刻度数字
    # -----------------------------
    bar_with_ticks = np.zeros((H, bar_width + 80, 3), dtype=np.uint8)
    bar_with_ticks[:, :bar_width] = colorbar

    # 选择几个刻度
    ticks = [0, 50, 100, 150, 200, 255]
    for t in ticks:
        y = int(H - (t / 255) * H)
        cv2.line(bar_with_ticks, (bar_width, y), (bar_width + 10, y), (255, 255, 255), 1)
        cv2.putText(
            bar_with_ticks, f"{t}",
            (bar_width + 15, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (255, 255, 255), 1, cv2.LINE_AA
        )

    # -----------------------------
    #  拼接热力图 + colorbar
    # -----------------------------
    depth_with_bar = np.hstack((depth_color, bar_with_ticks))

    return depth_with_bar