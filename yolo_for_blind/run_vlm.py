#!/usr/bin/env python3
import cv2
import time
import os
import signal
import sys
import gc

from vlm import vlm_infer
from voice import broadcast_warning_vlm

print("[VLM Independent] Starting...")

# ---------------- 自动重启机制 ----------------
def auto_restart(signum, frame):
    print("[VLM] 10分钟自动重启，防止极端卡死...")
    os.execl(sys.executable, sys.executable, *sys.argv)

signal.signal(signal.SIGALRM, auto_restart)
signal.alarm(600)  # 10 分钟后自动重启

# ---------------- 摄像头初始化 ----------------
cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    raise RuntimeError("VLM cannot open camera!")

last_time = time.time()
consecutive_failures = 0  # 连续失败计数

# ---------------- 主循环 ----------------
while True:
    try:
        # 只在调用 VLM 前读取一帧，减少无用 buffer
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # 每 20 秒调用一次 VLM
        if time.time() - last_time >= 20:
            print(f"[VLM] {time.strftime('%H:%M:%S')} 正在理解场景...")

            result = vlm_infer(frame)

            if result and isinstance(result, dict):
                resp = result.get('response', '')
                if resp:
                    resp = resp.replace('<|im_end|>', '').strip()
                    try:
                        vlm_list = eval(resp)
                        if isinstance(vlm_list, list) and len(vlm_list) == 5:
                            print(f"[VLM] 成功！播报: {vlm_list}")
                            broadcast_warning_vlm(vlm_list)
                            consecutive_failures = 0
                        else:
                            print(f"[VLM] 返回格式不对，跳过: {resp}")
                    except Exception as e:
                        print(f"[VLM] eval 失败，跳过: {e}")
                else:
                    print("[VLM] 返回空，跳过")
            else:
                consecutive_failures += 1
                print(f"[VLM] 调用失败（第{consecutive_failures}次），跳过...")

            # ---------------- 手动释放对象，避免内存累积 ----------------
            del frame, result
            try:
                del vlm_list
            except NameError:
                pass
            gc.collect()

            last_time = time.time()

        else:
            # 不推理时轻量等待
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[VLM] 手动停止")
        break
    except Exception as e:
        print(f"[VLM] 未知异常: {e}，继续运行...")
        time.sleep(1)

print("[VLM] 程序结束")
cap.release()
