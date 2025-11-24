## 环境配置

``` bash
conda create -n Qwen python=3.10
conda activate Qwen

nvidia-smi 看cuda版本，12.7可以兼容12.x
# 在这个网址下载对应cuda版本的Pytorch。https://pytorch.org/get-started/previous-versions/（以12.1为例）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 验证pytorch已成功安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

# 安装transformers库和量化工具
pip install transformers accelerate bitsandbytes 

# 在 https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main 下载项目文件夹到本文件夹路径内,如下
(Qwen) hejunhao2024@Administrator:/mnt/d/Labs/Qwen_2.5_vl_2B_4bit$ tree .
.
├── Qwen3-VL-2B-Instruct
│   ├── chat_template.json
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── video_preprocessor_config.json
│   └── vocab.json
├── README.md
├── image1.png
├── image2.png
├── qwen.py
├── qwen_request.py
└── qwen_serve.py

```

## 运行
```bash
# 单张图片推理
python qwen.py

# 开启推理服务
python qwen_serve.py

# 发送请求调用服务
python evaluate.py
```
