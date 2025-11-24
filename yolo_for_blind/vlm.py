# import base64
# import requests
# import cv2
# import numpy as np

# from utils import logger
# # API_URL = "http://localhost:5000/infer"
# API_URL = "http://test.hejunhao2024.me/infer"
# prompt = (
#     "Look at the scene in the image and answer the following five questions. "
#     "For each question, select the number corresponding to the correct option. "
#     "Return your answer only as a list of five numbers in this format: [x1, x2, x3, x4, x5].\n\n"
    
#     "1. What kind of place is this?\n"
#     "   Options: 1. park, 2. road, 3. neighbourhood, 4. marketplace, "
#     "5. pedestrian street, 6. public running track, 7. bridge, 8. parking lot, "
#     "9. footbridge, 10. stairway 11.indoors\n\n"
    
#     "2. What is the main road condition?\n"
#     "   Options: 1. Clear and wide path, 2. Narrow path, 3. Crowded path, "
#     "4. Complex intersection or crossing\n\n"
    
#     "3. What is the immediate walking safety level?\n"
#     "   Options: 1. Very safe, smooth walking, 2. Generally safe, minor obstacles, "
#     "3. Need caution, noticeable obstacles, 4. Dangerous, immediate obstacles ahead\n\n"
    
#     "4. Are there moving objects nearby?\n"
#     "   Options: 1. No moving objects detected, 2. Slow moving objects in distance, "
#     "3. Fast moving objects approaching, 4. Immediate collision risk detected\n\n"
    
#     "5. Is the ground condition safe for walking?\n"
#     "   Options: 1. Flat and smooth surface, easy to walk, 2. Slightly uneven but generally safe, "
#     "3. Rough terrain, need to be cautious, 4. Stairs or significant elevation changes\n\n"
    
#     "Return only the list [x1, x2, x3, x4, x5] with the corresponding numbers."
# )



# # def vlm_infer(frame, api_url="http://test.hejunhao2024.me/infer"):
# #     """
# #     Analyze a single frame using the VLM API.

# #     Args:
# #         frame (np.ndarray): The input image frame in BGR format.
# #         api_url (str): The URL of the VLM API.

# #     Returns:
# #         dict: The JSON response from the API.
# #     """
# #     # Encode the frame as a base64 string
# #     _, buffer = cv2.imencode('.png', frame)
# #     image_b64 = base64.b64encode(buffer).decode("utf-8")

# #     # Prepare the payload
# #     payload = {
# #         "image": image_b64,
# #         "prompt": prompt
# #     }

# #     # Send the request to the API
# #     response = requests.post(api_url, json=payload)

# #     # Return the JSON response
# #     return response.json()


# def vlm_infer(frame, api_url="http://test.hejunhao2024.me/infer"):
#     if frame is None or frame.size == 0:
#         logger.error("Empty frame, skipping VLM inference")
#         return None
#     print(333)
#     # 确保三通道
#     if len(frame.shape) != 3 or frame.shape[2] != 3:
#         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#     try:
#         _, buffer = cv2.imencode('.png', frame)
#         if not _:
#             logger.error("Failed to encode frame to PNG")
#             return None
#         image_b64 = base64.b64encode(buffer).decode("utf-8")
#     except Exception as e:
#         logger.error(f"OpenCV encode failed: {e}")
#         return None

#     payload = {"image": image_b64, "prompt": prompt}

#     try:
#         response = requests.post(api_url, json=payload, timeout=15)
#         return response.json()
#     except Exception as e:
#         logger.error(f"VLM API request failed: {e}")
#         return None

# if __name__ == "__main__":
#     test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
#     vlm_result = vlm_infer(test_frame)
#     print(vlm_result)
#     vlm_response = vlm_result.get('response', '')
#     vlm_response_cleaned = vlm_response.replace('<|im_end|>', '').strip()
#     vlm_list = eval(vlm_response_cleaned)
#     print(vlm_list)



# vlm.py
import base64
import requests
import cv2
import numpy as np
import time

API_URL = "http://test.hejunhao2024.me/infer"
prompt = (
    "Look at the scene in the image and answer the following five questions. "
    "For each question, select the number corresponding to the correct option. "
    "Return your answer only as a list of five numbers in this format: [x1, x2, x3, x4, x5].\n\n"
    
    "1. What kind of place is this?\n"
    "   Options: 1. park, 2. road, 3. neighbourhood, 4. marketplace, "
    "5. pedestrian street, 6. public running track, 7. bridge, 8. parking lot, "
    "9. footbridge, 10. stairway 11.indoors\n\n"
    
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


def vlm_infer(frame):
    if frame is None or frame.size == 0:
        return None

    # 确保三通道
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # 1. 编码为 JPG（比 PNG 小 70%，快 3 倍）
    success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not success:
        return None

    image_b64 = base64.b64encode(buffer).decode("utf-8")
    payload = {"image": image_b64, "prompt": prompt}

    # 2. 最多重试 2 次，超时 45 秒（3050 够用）
    for attempt in range(2):
        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=45,                    # 关键！给够时间
                headers={"Connection": "close"} # 强制关闭连接，防卡
            )

            # 200 才继续，其他直接跳过
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[VLM] 服务端错误 {response.status_code}，跳过本次")
                return None

        except requests.exceptions.Timeout:
            print(f"[VLM] 第{attempt+1}次超时，{'最后' if attempt==1 else '再试一次'}...")
            time.sleep(3)
        except requests.exceptions.ConnectionError:
            print(f"[VLM] 连接断开，跳过本次")
            return None
        except Exception as e:
            print(f"[VLM] 请求异常: {e}")
            return None

    return None  # 两次都失败，直接放弃本次