import os
from gen_wav import text_to_speech

# 输入文件路径
AUDIO_LIST_FILE = "audios_list.txt"
# 输出音频文件目录
OUTPUT_DIR = "audios"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_audio_list(file_path):
    """
    解析 audio_list.txt 文件，提取标识符和文本内容。
    """
    audio_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # 跳过空行和注释
                continue
            if "," in line:
                key, text = line.split(",", 1)
                key = key.strip().strip("'")
                text = text.strip()
                audio_data.append((key, text))
    return audio_data

def generate_audio_files(audio_data, output_dir):
    """
    根据解析的音频数据生成 .wav 文件。
    """
    for key, text in audio_data:
        output_path = os.path.join(output_dir, f"{key}.wav")
        if os.path.exists(output_path):
            print(f"[SKIP] {output_path} 已存在，跳过生成。")
            continue
        try:
            print(f"[GENERATE] 生成音频: {key} -> {text}")
            text_to_speech(text, output_path)
        except Exception as e:
            print(f"[ERROR] 无法生成音频 {key}: {e}")

if __name__ == "__main__":
    # 解析文件
    audio_list = parse_audio_list(AUDIO_LIST_FILE)
    # 生成音频文件
    generate_audio_files(audio_list, OUTPUT_DIR)
