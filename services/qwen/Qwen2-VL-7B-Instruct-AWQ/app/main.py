from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from typing import List
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from PIL import Image
import io
import requests
import base64
from qwen_vl_utils import process_vision_info

app = FastAPI()

model_id = "Qwen/Qwen2-VL-7B-Instruct-AWQ"

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="cuda", 
    trust_remote_code=True, 
    torch_dtype="auto", 
    attn_implementation='flash_attention_2',
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels
) 

@app.post("/generate")
async def generate(prompt: str = Form(...), images: List[UploadFile] = File(...)):
    # Process images
    pil_images = []
    for img in images:
        image = Image.open(io.BytesIO(await img.read()))
        pil_images.append(image)
    
    # Prepare the prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ] + [{"type": "image", "image": img} for img in pil_images]
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")
    
    # Stream the response
    async def generate_stream():
        generation_args = { 
            "max_new_tokens": 1000, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 
        
        generated_ids = model.generate(**inputs, **generation_args)
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        yield response

    return StreamingResponse(generate_stream(), media_type="text/plain")


@app.get("/")
async def root():
    return {"message": "Welcome to the Qwen2-VL-7B-Instruct-AWQ API"}
