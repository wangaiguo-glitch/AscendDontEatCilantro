# voice.py
import os
import subprocess
import time
import random
import threading
from collections import defaultdict

AUDIO_DIR = "/home/HwHiAiUser/work/hejunhao/yolo_for_blind/audios"

# ==================== 2. 全局控制 ====================
last_play_time = 0.0          # 上次播放时间戳
PLAY_COOLDOWN = 5.0           # 相邻两次播报间隔 3 秒
audio_lock = threading.Lock() # 线程安全

# ==================== 3. 检测参数 ====================
CONF_THRESHOLD = 0.5          # 置信度阈值
AREA_THRESHOLD = 10000         # 最小框面积（像素），可调：3000 ≈ 55x55 像素
# 640x640 输入下，3000 ≈ 较明显的物体

# ==================== 4. 类别映射（与模型一致） ====================
CLASS_NAMES = [
    'car','truck','pole','tree','crosswalk','warning_column','bicycle','person','dog','sign',
    'red_light','fire_hydrant','bus','motorcycle','reflective_cone','green_light',
    'ashcan','blind_road','tricycle','roadblock'
]



# ==================== 5. 播放函数 ====================
def _play_audio(class_name: str) -> bool:
    """播放 {class_name}.wav"""
    audio_path = os.path.join(AUDIO_DIR, f"{class_name}.wav")
    if not os.path.exists(audio_path):
        print(f"[VOICE] 音频缺失: {audio_path}")
        return False
    try:
        subprocess.Popen(
            ['aplay', '-q', audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        print(f"[VOICE] 播放失败 {audio_path}: {e}")
        return False

# ==================== 6. 主播报函数 ====================
def broadcast_warning(result_dict, midas_depth=None, vlm_results=None):
    """
    输入：detect_once 返回的 result_dict
    逻辑：
      1. 遍历所有检测框
      2. 满足：conf > 0.5 且 面积 > 3000
      3. 收集候选类（可重复）
      4. 若有多个 → 按数量加权随机选一个播报
      5. 冷却 3 秒
    """
    global last_play_time

    if not result_dict or not result_dict.get("category_id"):
        return

    # 收集候选类别（允许重复，用于加权）
    candidates = []

    for cat_id, score, bbox in zip(
        result_dict["category_id"],
        result_dict["score"],
        result_dict["bbox"]
    ):
        if score < CONF_THRESHOLD:
            continue
        class_name = CLASS_NAMES[cat_id]

        # 计算面积：w * h
        x, y, w, h = bbox
        area = w * h
        if area < AREA_THRESHOLD:
            continue

        # 满足条件 → 加入候选（可重复）
        candidates.append(class_name)

    if not candidates:
        return

    # 冷却判断
    now = time.time()
    with audio_lock:
        if now - last_play_time < PLAY_COOLDOWN:
            return

        # 加权随机选择一个类别
        chosen_class = random.choice(candidates)

        if _play_audio(chosen_class):
            last_play_time = now
            count = len(candidates)
            others = len(set(candidates)) - 1
            extra = f"（共 {count} 个目标，{others} 种其他）" if count > 1 else ""
            print(f"[VOICE] 播放: {chosen_class}.wav {extra}")