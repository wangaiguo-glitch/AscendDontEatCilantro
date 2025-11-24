项目在香橙派的路径:/home/HwHiAiUser/work/hejunhao/yolo_for_blind

1.根据yolo_for_blind/README.md运行代码,调midas, yolo参数

2.进入Qwen_2.5_vl_2B_4bit里的vlm评测脚本vlm_tes文件夹，调提示词和播报逻辑，改voice.py

如果播放时缺少语音文件，在audio_list.txt底部加上以后运行gen_wav_from_list.py自动在audios里面生成音频文件

3.美化app.py 前端页面