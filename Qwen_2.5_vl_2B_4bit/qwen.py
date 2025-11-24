from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
local_model_path = "./Qwen3-VL-2B-Instruct" # Path to the local model directory
processor = AutoProcessor.from_pretrained(local_model_path)

model = AutoModelForImageTextToText.from_pretrained(
    local_model_path,
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True
)


image = Image.open("image1.png").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is the animal in the picture? "}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))