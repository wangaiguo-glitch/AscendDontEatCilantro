import base64
import requests
import os
from tqdm import tqdm  # 添加 tqdm 进度条库

# API_URL = "http://localhost:5000/infer"
API_URL = "http://test.hejunhao2024.me/infer"
DATA_PATH = "/mnt/d/Labs/AscendDontEatCilantro/Qwen_2.5_vl_2B_4bit/vlm_test"

# 读取 label.txt
labels = {}
with open(os.path.join(DATA_PATH, "label.txt"), "r") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            idx = int(parts[0])
            label_list = eval(parts[1])
            labels[idx] = label_list

# 初始化得分统计
total_score = 0
total_questions = 0
position_scores = [0] * 5  # t1 到 t5 的得分

# 遍历每张图片，添加 tqdm 进度条
for idx, true_list in tqdm(labels.items(), desc="Processing images", unit="image"):
    image_path = os.path.join(DATA_PATH, "data", f"{idx}.jpg")
    if not os.path.exists(image_path):
        print(f"Image {idx}.jpg not found, skipping.")
        continue

    # 编码图片
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 定义 prompt
    prompt = (
        "Look at the scene in the image and answer the following five questions. "
        "For each question, select the number corresponding to the correct option. "
        "Return your answer only as a list of five numbers in this format: [x1, x2, x3, x4, x5].\n\n"
        "1. What kind of place is this?\n"
        "   Options: 1. park, 2. road, 3. neighbourhood, 4. marketplace, "
        "5. pedestrian street, 6. public running track, 7. bridge, 8. parking lot, "
        "9. footbridge, 10. stairway\n\n"
        "2. What is the main road condition?\n"
        "   Options: 1. Clear and wide path, 2. Narrow path, 3. Crowded path, "
        "4. Complex intersection or crossing\n\n"
        "3. What is the immediate walking safety level?\n"
        "   Options: 1. Very safe, smooth walking, 2. Generally safe, minor obstacles, "
        "3. Need caution, noticeable obstacles, 4. Dangerous, immediate obstacles ahead\n\n"
        "4. Are there moving objects nearby?\n"
        "   Options: 1. No moving objects detected, 2. Slow moving objects in distance, "
        "3. Fast moving objects approaching, 4. Immediate collision risk detected\n\n"
        "5. Is the ground condition safe for walking?\n"
        "   Options: 1. Flat and smooth surface, easy to walk, 2. Slightly uneven but generally safe, "
        "3. Rough terrain, need to be cautious, 4. Stairs or significant elevation changes\n\n"
        "Return only the list [x1, x2, x3, x4, x5] with the corresponding numbers."
    )

    # 调用 VLM 接口
    payload = {"image": image_b64, "prompt": prompt}
    response = requests.post(API_URL, json=payload)
    vlm_result = response.json()

    # 处理响应
    vlm_response = vlm_result.get('response', '')
    vlm_response_cleaned = vlm_response.replace('<|im_end|>', '').strip()
    try:
        vlm_list = eval(vlm_response_cleaned)
        if isinstance(vlm_list, list) and len(vlm_list) == 5:
            score = 0
            for i in range(5):
                if vlm_list[i] == true_list[i]:
                    score += 1
                    position_scores[i] += 1
            total_score += score
            total_questions += 5
            print(f"Question {idx}:")
            print(f"  True: {true_list}")
            print(f"  Model: {vlm_list}")
        else:
            print(f"Question {idx}: Invalid response, score 0")
            print(f"  True: {true_list}")
            print(f"  Model: {vlm_response_cleaned}")
    except:
        print(f"Question {idx}: Failed to parse, score 0")
        print(f"  True: {true_list}")
        print(f"  Model: {vlm_response_cleaned}")

# 打印总分和位置得分
if total_questions > 0:
    overall_score = total_score / total_questions
    print(f"\nOverall Score: {overall_score:.2f}")
    print("Position Scores (t1 to t5):", position_scores)
else:
    print("No valid questions processed.")

