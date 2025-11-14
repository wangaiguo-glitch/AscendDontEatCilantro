import threading
import argparse
import ast
import os
import time
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, Response
import threading
import random
import yaml
import mindspore_lite as mslite

from utils import logger
from utils.config import parse_args
from utils.metrics import non_max_suppression, scale_coords, xyxy2xywh
from utils.utils import set_seed 
from voice.voice import broadcast_warning

def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="Infer", parents=[parents] if parents else [])
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--mindir_path", type=str, help="mindir path")
    parser.add_argument("--result_folder", type=str, default="./log_result", help="predicted results folder")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="IOU threshold for NMS")
    parser.add_argument(
        "--conf_free", type=ast.literal_eval, default=False, help="Whether the prediction result include conf"
    )
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    parser.add_argument("--image_path", type=str, default="/dev/video0", help="path to image or camera device")
    parser.add_argument("--save_result", type=ast.literal_eval, default=True, help="whether save the inference result")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )
    parser.add_argument("--flask_port", type=int, default=5000, help="Flask port for web view")
    return parser


def set_default_infer(args):
    args.data.nc = 1 if args.single_cls else int(args.data.nc)
    args.data.names = ["item"] if args.single_cls and len(args.data.names) != 1 else args.data.names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )
    args.result_folder = os.path.join(args.result_folder, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    os.makedirs(args.result_folder, exist_ok=True)
    with open(os.path.join(args.result_folder, "cfg.yaml"), "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO")
    logger.setup_logging_file(log_dir=os.path.join(args.result_folder, "logs"))


def detect_once(
    model,
    img: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
    conf_free: bool = False,
    nms_time_limit: float = 20.0,
    img_size: int = 640,
):
    h_ori, w_ori = img.shape[:2]
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

    img_input = img_padded[:, :, ::-1].transpose(2, 0, 1) / 255.0
    img_input = np.ascontiguousarray(img_input[None].astype(np.float32))

    _t = time.time()
    inputs = model.get_inputs()
    assert img_input.shape == (1, 3, 640, 640), f"Input shape: {img_input.shape}"
    inputs[0].set_data_from_numpy(img_input)
    outputs = model.predict(inputs)
    out = [o.get_data_to_numpy().copy() for o in outputs][0]
    infer_times = time.time() - _t

    logger.info('perform nms...')
    t = time.time()
    out = non_max_suppression(
        out,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        conf_free=conf_free,
        multi_label=True,
        time_limit=nms_time_limit,
    )
    nms_times = time.time() - t

    result_dict = {"category_id": [], "bbox": [], "score": [], "segmentation": None}
    for pred in out:
        if len(pred) == 0:
            continue
        predn = pred.copy()
        scale_coords(img_padded.shape[:2], predn[:, :4], (h_ori, w_ori))
        boxes = xyxy2xywh(predn[:, :4])
        boxes[:, :2] -= boxes[:, 2:] / 2
        for p, b in zip(pred.tolist(), boxes.tolist()):
            result_dict["category_id"].append(int(p[5]))
            result_dict["bbox"].append([round(x, 3) for x in b])
            result_dict["score"].append(round(p[4], 5))

    t = tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)
    logger.info(f"Predict result is: {result_dict}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info("Detect a frame success.")
    return result_dict


def draw_in_memory(img: np.ndarray, result_dict, data_names, is_coco_dataset=False):
    im = img.copy()
    category_id = result_dict["category_id"]
    bbox = result_dict["bbox"]
    score = result_dict["score"]
    seg = result_dict.get("segmentation", None)
    mask = None if seg is None else np.zeros_like(im, dtype=np.float32)

    for i in range(len(bbox)):
        x_l, y_t, w, h = bbox[i]
        x_r, y_b = x_l + w, y_t + h
        x_l, y_t, x_r, y_b = int(x_l), int(y_t), int(x_r), int(y_b)
        _color = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(im, (x_l, y_t), (x_r, y_b), tuple(_color), 2)


        class_name_index = category_id[i]
        class_name = data_names[class_name_index]
        text = f"{class_name}: {score[i]}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(im, (x_l, y_t - text_h - baseline), (x_l + text_w, y_t), tuple(_color), -1)
        cv2.putText(im, text, (x_l, y_t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if seg:
        im = (0.7 * im + 0.3 * mask).astype(np.uint8)
    return im


app = Flask(__name__)
frame_lock = threading.Lock()
current_frame = None


def camera_loop(model, args):
    global current_frame
    cap = cv2.VideoCapture(args.image_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera: {args.image_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_idx += 1
        result_dict = detect_once(
            model=model,
            img=frame,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            nms_time_limit=args.nms_time_limit,
            img_size=args.img_size,
        )

        broadcast_warning(result_dict)

        drawn_frame = draw_in_memory(
            img=frame,
            result_dict=result_dict,
            data_names=args.data.names,
            is_coco_dataset=False
        )

        if args.save_result:
            save_dir = os.path.join(args.result_folder, "detect_results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(save_path, drawn_frame)

        with frame_lock:
            current_frame = drawn_frame.copy()

        time.sleep(0.03)  # ~30 FPS


@app.route('/video_feed')
def video_feed():
    def gen():
        global current_frame
        while True:
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.1)
                    continue
                frame = current_frame.copy()
            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return '''
    <h1>MindYOLO Real-time Detection</h1>
    <p><img src="/video_feed" width="800"/></p>
    <p>Custom classes from config | Device: /dev/video0</p>
    '''

def infer(args):
    set_seed(args.seed)
    set_default_infer(args)

    context = mslite.Context()
    context.target = ["Ascend"]
    model = mslite.Model()
    logger.info('mslite model init...')
    model.build_from_file(args.mindir_path, mslite.ModelType.MINDIR, context)

    thread = threading.Thread(target=camera_loop, args=(model, args), daemon=True)
    thread.start()

    logger.info(f"Flask server: http://0.0.0.0:{args.flask_port}")
    logger.info(f"Open: http://<your-ip>:{args.flask_port}")
    app.run(host='0.0.0.0', port=args.flask_port, debug=False, use_reloader=False)


if __name__ == "__main__":
    parser = get_parser_infer()
    args = parse_args(parser)
    infer(args)