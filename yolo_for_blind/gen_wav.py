import requests

def text_to_speech(text, save_path="output.wav"):
    # 第一步：生成语音
    generate_url = "https://hbapi.qikekeji.com/freetts/tts/generate/"

    payload = {
        "text": text,
        "format": "wav",      
        "voice": "zhiqi",
        "speed": "1",
        "volume": "80",
        "pitch": "1",
        "quality": "high"
    }

    headers = {
    "authority": "hbapi.qikekeji.com",
    "method": "POST",
    "path": "/freetts/tts/generate/",
    "scheme": "https",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "priority": "u=0, i",
    "referer": "https://freetts.cn/",
    "sec-ch-ua": '"Chromium";v="142", "Microsoft Edge";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "cross-site",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
    "cookie": "__51uvsct__3LRY3CcwX96Ykhks=1; __51vcke__3LRY3CcwX96Ykhks=a0469a34-da94-5a3b-bebc-d80d42c7fb39; __51vuft__3LRY3CcwX96Ykhks=1763814195514; __vtins__3LRY3CcwX96Ykhks=%7B%22sid%22%3A%20%22ab844b59-4ebd-5ad6-84c2-836e83cb4f82%22%2C%20%22vd%22%3A%201%2C%20%22stt%22%3A%200%2C%20%22dr%22%3A%200%2C%20%22expires%22%3A%201763815995510%2C%20%22ct%22%3A%201763814195510%7D;"
}


    print("正在生成语音...")
    res = requests.post(generate_url, data=payload, headers=headers)

    if res.status_code != 200:
        print("生成失败:", res.text)
        return

    data = res.json()

    if data.get("status") != 1:
        print("API返回错误:", data)
        return

    file_url = data["data"]["file_url"]
    print("生成成功，文件下载地址:", file_url)

    # 第二步：下载音频
    print("正在下载音频文件...")
    audio_res = requests.get(file_url, headers=headers)

    if audio_res.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(audio_res.content)
        print("下载成功！文件已保存为:", save_path)
    else:
        print("下载失败:", audio_res.status_code)


if __name__ == "__main__":
    text = "检测到绿灯，请通行"
    text_to_speech(text, "result.wav")
