# Vision-guide 项目说明

# 文件结构（需要关注的文件）
```
.
├── README.md 
├── app.py # flask网页
├── audios # wav音频文件夹
├── audios_list.txt # 音频需求列表
├── gen_wav.py # 生成音频文件 
├── gen_wav_from_list.py # 生成列表中所有音频的爬虫文件
├── midas_small.mindir # 模型文件
├── predict.py # 主函数
├── params.json # 语音播报参数
├── run.sh # 运行脚本
├── run_midas.py # midas模型推理函数
├── run_yolo.py # yolo模型推理函数
├── serve8000.py 开启手机端服务
├── vlm.py # vlm模型推理函数
├── voice.py # 语音播报逻辑
└── yolov8s_wotr.mindir # 模型文件
```

# 运行代码
1. 确定摄像头端口
先拔掉摄像头USB，执行
```bash
ls /dev/video*
```
再插上摄像头的USB执行这个，确定摄像头的端口 (video0或video2)
在run.sh 相应位置修改摄像头端口
```bash
# run.sh
--image_path /dev/video0 \
```
**每次重连香橙派都要看一遍**

2.确保你在/home/HwHiAiUser/work/hejunhao/yolo_for_blind路径下，且当前conda环境为(Mindspore)
```bash
cd /home/HwHiAiUser/work/hejunhao/yolo_for_blind
conda activate Mindspore
```
一般它会自动conda activate Mindspore

3.vlm 端开启服务，（在你的电脑执行qwen_serve.py),在vlm.py中改成你的域名
```python
# vlm.py
API_URL = "http://test.hejunhao2024.me/infer"
```

4.终端执行
```bash
./run.sh
```

5.手机端打开
buchixiangcai.hejunhao2024.me (这个网址和香橙派5000端口绑定了，不用改成你的)

注：调参阶段可以把vlm模式关掉，这样能避免跑了一会儿就断连，方法：
```python
# predict.py lin 158
            if elapsed_time >= 20: # 20改成10000
```