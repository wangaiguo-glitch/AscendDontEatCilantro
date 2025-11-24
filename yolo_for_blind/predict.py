#!/usr/bin/env python3
import threading
import argparse
import ast
import os
import time
import cv2
import numpy as np
from flask import Flask, Response
import mindspore_lite as mslite
import multiprocessing as mp
import io
from utils import logger
from utils.config import parse_args
from utils.utils import set_seed 
from voice import broadcast_warning, play_queue_worker, broadcast_warning_vlm   
from run_yolo import detect_once, draw_in_memory, set_default_infer
from run_midas import depth_once, draw_depth
from vlm import vlm_infer
import app as webapp
from app import run_flask

MIDAS_MODEL_PATH = "/home/HwHiAiUser/work/hejunhao/yolo_for_blind/midas_small.mindir"

# ---------- CLI parser ----------
def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="Infer", parents=[parents] if parents else [])
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--mindir_path", type=str, help="YOLO mindir path")
    parser.add_argument("--mindir_path_midas", type=str, help="MIDAS mindir path")
    parser.add_argument("--result_folder", type=str, default="./log_result")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--conf_thres", type=float, default=0.25)
    parser.add_argument("--iou_thres", type=float, default=0.65)
    parser.add_argument("--conf_free", type=ast.literal_eval, default=False)
    parser.add_argument("--nms_time_limit", type=float, default=60.0)
    parser.add_argument("--image_path", type=str, default="/dev/video0")
    parser.add_argument("--save_result", type=ast.literal_eval, default=True)
    parser.add_argument("--single_cls", type=ast.literal_eval, default=False)
    parser.add_argument("--flask_port", type=int, default=5000)
    return parser

# ---------- Global inference lock ----------
infer_lock = threading.Lock()  # 保证同一时间只有一个推理在执行

# ---------- 顶部新增 imports ----------
import multiprocessing as mp
import io

# ---------- 全局：vlm 交互队列 & 进程  ----------
# 使用 spawn 上下文来更安全地创建子进程（避免 fork 导致的驱动/线程问题）
MP_CTX = mp.get_context('spawn')
vlm_queue = MP_CTX.Queue(maxsize=1)
vlm_proc = None

def vlm_worker_process(queue: mp.Queue):
    """
    运行在子进程中：读取 JPEG bytes，解码，调用 vlm_infer，然后播放/广播结果。
    任何在此子进程发生的 C 层崩溃不会摧毁主进程。
    """
    print("[vlm_proc] started")
    from vlm import vlm_infer  # 在子进程内导入，避免在主进程 preload 时影响
    from voice import broadcast_warning_vlm
    import numpy as np
    import cv2
    import time
    import traceback
    while True:
        try:
            # 阻塞等待（超时可设置为 None，表示一直等）
            jpg_bytes = queue.get()
            if jpg_bytes is None:
                # 约定用 None 来通知退出
                break

            # decode bytes -> image
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                # 无效帧，略过
                continue

            # 调用网络 API (vlm_infer)：这个操作可能阻塞，但在子进程
            try:
                vlm_result = vlm_infer(frame)
                if not vlm_result:
                    continue
                vlm_response = vlm_result.get('response', '')
                vlm_response_cleaned = vlm_response.replace('<|im_end|>', '').strip()
                # 小心 eval（你原代码用 eval），这里保持原有行为但加 try
                try:
                    vlm_list = eval(vlm_response_cleaned)
                    if isinstance(vlm_list, list) and len(vlm_list) == 5:
                        # 直接在子进程调用播报函数（如果 broadcast_warning_vlm 使用 subprocess/aplay 等在子进程也能工作）
                        broadcast_warning_vlm(vlm_list)
                    else:
                        # 结果格式不对，记录
                        print(f"[vlm_proc] Invalid VLM response: {vlm_response_cleaned}")
                except Exception as e:
                    print(f"[vlm_proc] parse eval failed: {e}; resp={vlm_response_cleaned}")
            except Exception as e:
                print(f"[vlm_proc] vlm_infer failed: {e}")
        except Exception as e:
            print("[vlm_proc] unexpected error:", e)
            traceback.print_exc()
            time.sleep(1)  # 防止空转
    print("[vlm_proc] exit")

# ---------- Camera + Inference Loop ----------
def camera_infer_loop(model_yolo, args, model_midas=None):
    # global current_frame  # removed
    cap = cv2.VideoCapture(args.image_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera: {args.image_path}")

    frame_idx = 0
    start_time = time.time()
    
    while True:
        for _ in range(3):  # 通常读 3~5 次就够清空
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            time.sleep(0.01)
            continue
        frame_idx += 1

        try:
            # ---------------- YOLO 推理 ----------------
            with infer_lock:
                result_dict = detect_once(
                    model=model_yolo,
                    img=frame,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    conf_free=args.conf_free,
                    nms_time_limit=args.nms_time_limit,
                    img_size=args.img_size,
                )

            # ---------------- MIDAS 推理 ----------------
            with infer_lock:
                midas_depth = depth_once(
                    model=model_midas,
                    img=frame,
                )

            # ---------------- 语音播报 ----------------
            try:
                broadcast_warning(result_dict, midas_depth)
            except Exception as e:
                logger.warning(f"broadcast_warning failed: {e}")

            # ---------------- 每 20 秒调用一次 VLM ----------------
            elapsed_time = time.time() - start_time
            if elapsed_time >= 20:
                # 先尝试把当前帧压缩成 jpeg bytes
                try:
                    # 缩小传输的尺寸以减少带宽和编码时间（可选）
                    # small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
                    # 这里直接用原尺寸 jpeg，若希望更快可改为上面 small
                    _ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if _ok:
                        jpg_bytes = jpg.tobytes()
                        # 非阻塞 put：如果队列已满就丢弃（保持最新帧）
                        try:
                            vlm_queue.put_nowait(jpg_bytes)
                        except mp.queues.Full:
                            # 如果队列已满，主动丢弃旧的并替换为最新的一帧（保持最新）
                            try:
                                _ = vlm_queue.get_nowait()
                                vlm_queue.put_nowait(jpg_bytes)
                            except Exception:
                                pass
                    else:
                        logger.warning("Failed to encode frame for VLM; skip this VLM cycle.")
                except Exception as e:
                    logger.warning(f"Error encoding frame for VLM queue: {e}")
                start_time = time.time()

            # ---------------- 可视化 ----------------
            # drawn_frame = draw_in_memory(
            #     img=frame,
            #     result_dict=result_dict,
            #     data_names=args.data.names,
            #     is_coco_dataset=False
            # )
            # draw_midas_resized = cv2.resize(draw_depth(midas_depth), (frame.shape[1], frame.shape[0]))
            # combined = np.vstack((drawn_frame, draw_midas_resized))

            # # 更新到 Flask 输出（短临界区）
            # with webapp.frame_lock:
            #     webapp.current_frame = combined.copy()
            # ---------------- 可视化 ----------------
            drawn_frame = draw_in_memory(
                img=frame,
                result_dict=result_dict,
                data_names=args.data.names,
                is_coco_dataset=False
            )
            midas_visual = draw_depth(midas_depth)

            # 更新到 Flask 输出（短临界区）
            with webapp.frame_lock:
                webapp.current_frame = drawn_frame.copy()
                webapp.midas_frame = midas_visual.copy()

            # ---------------- 保存 YOLO 可视化 ----------------
            # if args.save_result:
            #     save_dir = os.path.join(args.result_folder, "detect_results")
            #     os.makedirs(save_dir, exist_ok=True)
            #     save_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
            #     try:
            #         cv2.imwrite(save_path, drawn_frame)
            #     except Exception as e:
            #         logger.warning(f"Failed to save frame {frame_idx}: {e}")

        except Exception as e:
            logger.error(f"Exception on frame {frame_idx}: {e}", exc_info=True)

# ---------- Main inference function ----------
def infer(args):
    set_seed(args.seed)
    set_default_infer(args)

    # ---------------- MIDAS (CPU) ----------------
    context_midas = mslite.Context()
    context_midas.target = ["CPU"]
    context_midas.ascend.provider = "ge"
    model_midas = mslite.Model()
    logger.info('mslite MIDAS model init...')
    model_midas.build_from_file(MIDAS_MODEL_PATH, mslite.ModelType.MINDIR, context_midas)

    # ---------------- YOLO (Ascend) ----------------
    context_yolo = mslite.Context()
    context_yolo.target = ["Ascend"]
    model_yolo = mslite.Model()
    logger.info('mslite YOLO model init...')
    model_yolo.build_from_file(args.mindir_path, mslite.ModelType.MINDIR, context_yolo)

    # ---------------- Flask 后台线程 ----------------
    flask_thread = threading.Thread(target=run_flask, kwargs={'host':'0.0.0.0', 'port': args.flask_port}, daemon=True)
    flask_thread.start()
    logger.info(f"Flask server thread started on port {args.flask_port}")

    # ---------- 顶部新增：启动 VLM 工作进程 ----------
    global vlm_proc
    vlm_proc = MP_CTX.Process(target=vlm_worker_process, args=(vlm_queue,), daemon=True)
    vlm_proc.start()
    logger.info(f"VLM worker process started (PID: {vlm_proc.pid})")

    # ---------------- Camera + 推理循环 ----------------
    camera_infer_loop(model_yolo, args, model_midas)

    # ---------- 程序退出时（可选）通知子进程退出 ----------
    # 在主进程退出流程里可以放：
    try:
        vlm_queue.put_nowait(None)
    except Exception:
        pass
    if vlm_proc is not None:
        vlm_proc.join(timeout=1)

# ---------- Entrypoint ----------
if __name__ == "__main__":
    parser = get_parser_infer()
    args = parse_args(parser)

    # 后台音频线程
    threading.Thread(target=play_queue_worker, daemon=True).start()

    # 启动主流程
    infer(args)
