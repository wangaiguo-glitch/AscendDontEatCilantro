# voice.py
import os
import subprocess
import time
import random
import threading
from collections import defaultdict

import json

PARAM_FILE = "/home/HwHiAiUser/work/hejunhao/yolo_for_blind/params.json"
# AUDIO_DIR = "/mnt/d/Labs/AscendDontEatCilantro/yolo_for_blind/audios"
AUDIO_DIR = "/home/HwHiAiUser/work/hejunhao/yolo_for_blind/audios"

CLASS_NAMES = [
    'car',            'truck',        'pole',       'tree',       'crosswalk',
    'warning_column', 'bicycle',      'person',     'dog',        'sign',
    'red_light',      'fire_hydrant', 'bus',        'motorcycle', 'reflective_cone',
    'green_light',    'ashcan',       'blind_road', 'tricycle',   'roadblock'
]

# CONF_THRESHOLD  = [0.8, 0.8, 0.8, 0.8, 0.8,
#                    0.8, 0.8, 0.8, 0.8, 0.8,
#                    0.8, 0.8, 0.8, 0.8, 0.8,
#                    0.8, 0.8, 0.8, 0.8, 0.8]

# AREA_THRESHOLD = [50000, 50000, 20000, 20000, 30000,
#                   10000, 20000, 20000, 10000, 10000,
#                   50000, 20000, 50000, 20000, 10000,
#                   10000, 10000, 10000, 20000, 10000]

# DEPTH_THRESHOLD = [133, 133, 100, 100, 100,
#                    100, 100, 100, 100, 100,
#                    133, 100, 133, 100, 100,
#                    100, 100, 100, 100, 100]

last_play_time = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]

# PLAY_TIME_COOLDOWN = [5, 5, 5, 5, 5,
#                       5, 5, 5, 5, 5,
#                       5, 5, 5, 5, 5,
#                       5, 5, 5, 5, 5]

def save_params(new_cfg):
    with open(PARAM_FILE, "w") as f:
        json.dump(new_cfg, f, indent=2)
    print("[PARAM] 参数已保存")
    load_params()  # 立即让 voice.py 使用新参数


def get_params():
    return {
        "CONF_THRESHOLD": CONF_THRESHOLD,
        "AREA_THRESHOLD": AREA_THRESHOLD,
        "DEPTH_THRESHOLD": DEPTH_THRESHOLD,
        "PLAY_TIME_COOLDOWN": PLAY_TIME_COOLDOWN
    }


def load_params():
    global CONF_THRESHOLD, AREA_THRESHOLD, DEPTH_THRESHOLD, PLAY_TIME_COOLDOWN
    try:
        with open(PARAM_FILE, "r") as f:
            cfg = json.load(f)

        CONF_THRESHOLD = cfg["CONF_THRESHOLD"]
        AREA_THRESHOLD = cfg["AREA_THRESHOLD"]
        DEPTH_THRESHOLD = cfg["DEPTH_THRESHOLD"]
        PLAY_TIME_COOLDOWN = cfg["PLAY_TIME_COOLDOWN"]

        print("[PARAM] 参数已加载并应用")
    except Exception as e:
        print("[PARAM] 加载失败，使用默认参数:", e)


load_params()
vlm_last_result = None
audio_lock = threading.Lock()

play_queue = []
queue_last_play_time = 0
QUEUE_PLAY_TIME_COOLDOWN = 0.3


def _play_audio(name: str) -> bool:
    """播放 {class_name}.wav"""
    audio_path = os.path.join(AUDIO_DIR, f"{name}.wav")
    if not os.path.exists(audio_path):
        print(f"[VOICE] 音频缺失: {audio_path}")
        return False
    try:
        subprocess.run(
            ['aplay', '-q', audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        print(f"[VOICE] 播放失败 {audio_path}: {e}")
        return False
    

def _play_audio_sequence(audio_list):
    """
    按顺序播放一组音频文件。
    Args:
        audio_list (list): 音频文件名列表（不含路径和扩展名）。
    """
    for name in audio_list:
        _play_audio(name)


def broadcast_warning(result_dict, midas_depth):

    for cat_id, score, bbox in zip(
        result_dict["category_id"],
        result_dict["score"],
        result_dict["bbox"]
    ):
        x, y, w, h = bbox
        area = w * h # 面积
        depth = midas_depth[int(round(y)), int(round(x))]  # 中心深度
        if area < AREA_THRESHOLD[cat_id] or score < CONF_THRESHOLD[cat_id] or depth > DEPTH_THRESHOLD[cat_id]:
            continue

        if time.time() - last_play_time[cat_id] < PLAY_TIME_COOLDOWN[cat_id]:
            continue
        
        # 满足条件 → 加入候选（可重复）
        play_queue.append([cat_id, x, depth])

    

def broadcast_warning_vlm(vlm_result):
    """
    单独为 VLM 播报开启一个 subprocess，按顺序播放所有内容。
    """
    print(f"[VOICE] VLM 播报内容: {vlm_result}")
    global vlm_last_result
    if vlm_last_result == vlm_result:
        return
    vlm_last_result = vlm_result

    # 构建音频播放序列
    audio_sequence = ['you_are_at']  # 您正处于

    # 根据 x1 播报地点
    x1, x2, x3, x4, x5 = vlm_result
    if x1 == 1:
        audio_sequence.append('park')
    elif x1 == 2:
        audio_sequence.append('road')
    elif x1 == 3:
        audio_sequence.append('neighbourhood')
    elif x1 == 4:
        audio_sequence.append('marketplace')
    elif x1 == 5:
        audio_sequence.append('pedestrian_street')
    elif x1 == 6:
        audio_sequence.append('public_running_track')
    elif x1 == 7:
        audio_sequence.append('bridge')
    elif x1 == 8:
        audio_sequence.append('parking_lot')
    elif x1 == 9:
        audio_sequence.append('footbridge')
    elif x1 == 10:
        audio_sequence.append('stairway')

    audio_sequence.append("ahead2")

    # 根据 x2 播报道路情况
    if x2 == 1:
        audio_sequence.append('clear_and_wide_path')
    elif x2 == 2:
        audio_sequence.append('narrow_path')
    elif x2 == 3:
        audio_sequence.append('crowded_path')
    elif x2 == 4:
        audio_sequence.append('complex_intersection_or_crossing')

    audio_sequence.append("here2")

    # 根据 x3 播报步行安全等级
    if x3 == 1:
        audio_sequence.append('very_safe_smooth_walking')
    elif x3 == 2:
        audio_sequence.append('generally_safe_minor_obstacles')
    elif x3 == 3:
        audio_sequence.append('need_caution_noticeable_obstacles')
    elif x3 == 4:
        audio_sequence.append('dangerous_immediate_obstacles_ahead')

    audio_sequence.append('in_your_activity_area')

    # 根据 x4 播报移动物体情况
    if x4 == 1:
        audio_sequence.append('no_moving_objects_detected')
    elif x4 == 2:
        audio_sequence.append('slow_moving_objects_in_distance')
    elif x4 == 3:
        audio_sequence.append('fast_moving_objects_approaching')
    elif x4 == 4:
        audio_sequence.append('immediate_collision_risk_detected')

    audio_sequence.append('ground2')

    # 根据 x5 播报地面情况
    if x5 == 1:
        audio_sequence.append('flat_and_smooth_surface_easy_to_walk')
    elif x5 == 2:
        audio_sequence.append('slightly_uneven_but_generally_safe')
    elif x5 == 3:
        audio_sequence.append('rough_terrain_need_to_be_cautious')
    elif x5 == 4:
        audio_sequence.append('stairs_or_significant_elevation_changes')

    audio_sequence.append('please_stay_safe')

    # 开启一个 subprocess 播放整个序列
    threading.Thread(target=_play_audio_sequence, args=(audio_sequence,), daemon=True).start()


def _play_warn(cat_id, x, depth):
    """
    播放单条警告的音频，按顺序播放。
    """
    audio_sequence = []
    class_name = CLASS_NAMES[cat_id]
    if class_name not in {'sign', 'red_light', 'green_light', 'blind_road', 'crosswalk'}:
        audio_sequence.extend(['detect', class_name, 'on_your'])  # 检测到 + 类别 + 在您的
        if x < 213:
            audio_sequence.append('left')  # 左前方
        elif x > 426:
            audio_sequence.append('right')  # 右前方
        else:
            audio_sequence.append("ahead")  # 正前方
        if depth < 0.2:
            audio_sequence.append('close')  # 离您较近
        audio_sequence.append("attention")  # 请注意
    else:
        audio_sequence.append(class_name)

    # 开启一个线程播放该条警告的音频序列
    threading.Thread(target=_play_audio_sequence, args=(audio_sequence,), daemon=True).start()


def play_queue_worker():
    """
    持续检查 play_queue，如果有内容则调用 _play_warn 播放警告。
    """
    global queue_last_play_time
    while True:
        if not play_queue:
            time.sleep(0.1)  # 如果队列为空，稍作等待
            continue

        now = time.time()
        if now - queue_last_play_time < QUEUE_PLAY_TIME_COOLDOWN:
            time.sleep(0.1)  # 控制播放间隔
            continue

        with audio_lock:  # 确保线程安全
            if play_queue:
                cat_id, x, depth = play_queue.pop(0)  # 从队列中取出第一个警告
                _play_warn(cat_id, x, depth)
                last_play_time[cat_id] = now  # 更新最后播放时间
                queue_last_play_time = now  # 更新最后播放时间
                print(f"[VOICE] 播放警告音频: {CLASS_NAMES[cat_id]}")


if __name__ == "__main__":
    a = [1, 2, 2, 3, 3]
    broadcast_warning_vlm(a)