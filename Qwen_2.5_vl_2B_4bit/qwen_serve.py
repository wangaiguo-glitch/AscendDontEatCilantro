from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import base64
import io

app = Flask(__name__)

# ---------------------
# Load Model on Startup
# ---------------------
local_model_path = "./Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(local_model_path)

model = AutoModelForImageTextToText.from_pretrained(
    local_model_path,
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

# ---------------------
# Helper: Decode Base64
# ---------------------
def decode_base64_image(b64_string):
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


# ---------------------
# Inference Route
# ---------------------
@app.route("/infer", methods=["POST"])
def infer():
    try:
        data = request.json
        
        image_b64 = data.get("image")
        prompt = data.get("prompt", "")

        if image_b64 is None:
            return jsonify({"error": "Missing 'image'"}), 400

        # Decode image
        image = decode_base64_image(image_b64)

        # Build chat message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process inputs
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        # Model output
        outputs = model.generate(**inputs, max_new_tokens=200)
        result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])

        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return "Qwen3-VL API is running."


if __name__ == "__main__":
    # Accessible for Cloudflare Tunnel
    app.run(host="0.0.0.0", port=5000)
