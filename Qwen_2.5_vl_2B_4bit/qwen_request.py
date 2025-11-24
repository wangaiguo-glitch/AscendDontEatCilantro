import base64
import requests
# API_URL = "http://localhost:5000/infer"
API_URL = "http://test.hejunhao2024.me/infer"
# API_URL = "http://test.jasminezzbot.dev/infer"
IMAGE_PATH = "3.jpg"

# ---------------------
# Encode image as base64
# ---------------------
with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")


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


payload = {
    "image": image_b64,
    "prompt": prompt,
}
response = requests.post(API_URL, json=payload)

print("Response:")
print(response.json())

