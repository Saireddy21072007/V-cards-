import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
import time

# 1. SETUP - Uses your local GPU
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_PATH = r"C:\Users\bsain\Desktop\Camera"
OUTPUT_FILE = r"C:\Users\bsain\Desktop\visiting_cards_extracted_local.json"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU!")

device = torch.device("cuda")
print(f"Loading local model on {device}...")
model_kwargs = {"trust_remote_code": True, "device_map": "cuda:0", "torch_dtype": torch.float16}

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **model_kwargs)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def extract_locally(image_path):
    # Prompt optimized for high-accuracy extraction
    prompt = """Extract all info from this business card into JSON.
Respond ONLY with a valid JSON object. Do not use markdown blocks (```json) or any other text.
{
  "first_name": "string", "last_name": "string", "organisation": "string",
  "designation": "string", "email": "string", "mobile": "string",
  "website": "string", "full_address": "string", "city": "string",
  "zip_code": "string", "gst_or_id": "string"
}"""

    messages = [
        {
            "role": "user",
            "content": [
            {"type": "image", "image": f"file://{image_path}", "max_pixels": 1003520},
            {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preprocessing
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Clean up markdown if the model includes it
    output_text = output_text.strip()
    if output_text.startswith("```json"):
        output_text = output_text[7:]
    if output_text.endswith("```"):
        output_text = output_text[:-3]
    output_text = output_text.strip()
    
    try:
        return json.loads(output_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON. Raw output:\n{output_text}")
        return {}

# 2. PROCESS ENTIRE DATASET
results = []
files = sorted(
    [f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
)[:3]

print(f"Processing 3 cards locally to measure performance...")
total_inference_time = 0
for filename in files:
    path = os.path.join(DATASET_PATH, filename)
    try:
        start_time = time.time()
        data = extract_locally(path)
        end_time = time.time()
        inference_time = end_time - start_time
        total_inference_time += inference_time
        
        data['filename'] = filename
        results.append(data)
        print(f"   [SUCCESS] Processed: {filename}")
        print(f"   [TIME] Time taken: {inference_time:.2f} seconds")
        print(f"   [DATA] Extracted Data: {json.dumps(data)}")
    except Exception as e:
        print(f"   [FAILED] Failed: {filename} - {e}")

# 3. SAVE
with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
    json.dump(results, f, indent=4)
print(f"DONE! Data saved to {OUTPUT_FILE}")
print(f"TOTAL INFERENCE TIME FOR 3 CARDS: {total_inference_time:.2f} seconds")
