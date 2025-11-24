from flask import Flask, render_template_string, redirect, url_for
import subprocess
import os
import signal

app = Flask(__name__)

PID_FILE = "yolo.pid"

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO 控制面板</title>
</head>
<body style="text-align:center; margin-top: 60px;">

    <h1>YOLO 服务控制界面</h1>

    <h2 style="color: {{ status_color }};">
        当前状态：{{ status_text }}
    </h2>

    <br><br>

    <form action="/start" method="post">
        <button type="submit" style="font-size:24px; padding:10px 30px;">开始服务</button>
    </form>

    <br>

    <form action="/stop" method="post">
        <button type="submit" style="font-size:24px; padding:10px 30px;">结束服务</button>
    </form>

</body>
</html>
"""

def is_running():
    """检查 YOLO 服务是否正在运行"""
    if not os.path.exists(PID_FILE):
        return False

    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read())

        # 检查进程组是否存在
        os.killpg(os.getpgid(pid), 0)
        return True
    except:
        return False


@app.route("/")
def index():
    if is_running():
        status_text = "🟢 运行中"
        status_color = "green"
    else:
        status_text = "🔴 已停止"
        status_color = "red"

    return render_template_string(
        HTML_PAGE,
        status_text=status_text,
        status_color=status_color
    )


@app.route("/start", methods=["POST"])
def start_service():
    if is_running():
        return redirect(url_for("index"))

    # 后台启动并创建独立进程组
    process = subprocess.Popen(
        ["/bin/bash", "run_yolo.sh"],
        preexec_fn=os.setsid
    )

    with open(PID_FILE, "w") as f:
        f.write(str(process.pid))

    return redirect(url_for("index"))


@app.route("/stop", methods=["POST"])
def stop_service():
    if not os.path.exists(PID_FILE):
        return redirect(url_for("index"))

    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read())

        # 杀掉整个进程组
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except:
        pass

    # 删除 pid 文件
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
