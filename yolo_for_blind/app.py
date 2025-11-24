#!/usr/bin/env python3
import threading
import time
import cv2
from flask import Flask, Response

from flask import request, jsonify
import voice  # 引入 voice.py
import json


app = Flask(__name__)

# ----------------- 与主推理共享的全局帧 -----------------
frame_lock = threading.Lock()
current_frame = None  # YOLO frame
midas_frame = None  # MIDAS frame

CLASS_NAMES = [
    'car',            'truck',        'pole',       'tree',       'crosswalk',
    'warning_column', 'bicycle',      'person',     'dog',        'sign',
    'red_light',      'fire_hydrant', 'bus',        'motorcycle', 'reflective_cone',
    'green_light',    'ashcan',       'blind_road', 'tricycle',   'roadblock'
]
# ----------------- 视频流路由 -----------------
@app.route('/video_feed')
def video_feed():
    def gen():
        global current_frame
        while True:
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.05)
                    continue
                frame = current_frame.copy()
            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------- MIDAS 视频流路由 -----------------
@app.route('/midas_feed')
def midas_feed():
    def gen():
        global midas_frame
        while True:
            with frame_lock:
                if midas_frame is None:
                    time.sleep(0.05)
                    continue
                frame = midas_frame.copy()
            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------- 首页 -----------------
@app.route('/')
def index():
    return f'''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>YOLO 参数控制面板</title>
<style>
    body {{ font-family: Arial; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px; text-align: center; }}
    th {{ background: #eee; }}
    input {{ width: 80px; }}
    #status {{ margin-top: 10px; font-weight: bold; }}
    .container {{ display: flex; flex-direction: column; align-items: center; }}
    .images {{ display: flex; flex-direction: column; align-items: center; }}
</style>
</head>
<body>

<div class="container">
    <div class="images">
        <div>
            <h2>YOLO Detection</h2>
            <img src="/video_feed" width="800"/>
        </div>
        <div>
            <h2>MIDAS Depth</h2>
            <img src="/midas_feed" width="800"/>
        </div>
    </div>

    <h2>🛠 YOLO 参数实时调节面板</h2>
    <p>修改后点击 <b>SAVE</b>，参数将立即应用于运行中的系统。</p>

    <table id="paramTable">
        <thead>
            <tr>
                <th>Class ID</th>
                <th>Class Name</th>
                <th>CONF_THRESHOLD</th>
                <th>AREA_THRESHOLD</th>
                <th>DEPTH_THRESHOLD</th>
                <th>COOLDOWN</th>
            </tr>
        </thead>
        <tbody></tbody>
    </table>

    <br>
    <button onclick="saveParams()">SAVE</button>
    <div id="status"></div>
</div>

<script>
var CLASS_NAMES = {json.dumps(CLASS_NAMES)};

function loadParams() {{
    fetch('/get_params')
        .then(res => res.json())
        .then(cfg => {{
            let tbody = document.querySelector('#paramTable tbody');
            tbody.innerHTML = '';

            for (let i = 0; i < cfg.CONF_THRESHOLD.length; i++) {{
                let row = `
                    <tr>
                        <td>${{i}}</td>
                        <td>${{CLASS_NAMES[i]}}</td>
                        <td><input id="c_${{i}}" value="${{cfg.CONF_THRESHOLD[i]}}"></td>
                        <td><input id="a_${{i}}" value="${{cfg.AREA_THRESHOLD[i]}}"></td>
                        <td><input id="d_${{i}}" value="${{cfg.DEPTH_THRESHOLD[i]}}"></td>
                        <td><input id="p_${{i}}" value="${{cfg.PLAY_TIME_COOLDOWN[i]}}"></td>
                    </tr>
                `;
                tbody.innerHTML += row;
            }}
        }});
}}

function saveParams() {{
    let newCfg = {{
        CONF_THRESHOLD: [],
        AREA_THRESHOLD: [],
        DEPTH_THRESHOLD: [],
        PLAY_TIME_COOLDOWN: []
    }};

    for (let i = 0; i < 20; i++) {{
        newCfg.CONF_THRESHOLD.push(parseFloat(document.getElementById("c_" + i).value));
        newCfg.AREA_THRESHOLD.push(parseInt(document.getElementById("a_" + i).value));
        newCfg.DEPTH_THRESHOLD.push(parseFloat(document.getElementById("d_" + i).value));
        newCfg.PLAY_TIME_COOLDOWN.push(parseFloat(document.getElementById("p_" + i).value));
    }}

    fetch('/save_params', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(newCfg)
    }})
    .then(res => res.json())
    .then(res => {{
        document.getElementById("status").innerText =
            res.status === "ok" ? "已保存并立即应用！" : "保存失败";
    }});
}}

loadParams(); // 页面加载时自动调用
</script>

</body>
</html>
    '''




@app.route("/get_params", methods=["GET"])
def get_params_api():
    return jsonify(voice.get_params())


@app.route("/save_params", methods=["POST"])
def save_params_api():
    data = request.json
    if not data:
        return jsonify({"status": "error", "msg": "empty json"}), 400

    voice.save_params(data)
    return jsonify({"status": "ok"})





def run_flask(host='0.0.0.0', port=5000):
    app.run(host=host, port=port, debug=False, use_reloader=False)