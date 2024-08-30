from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from typing import List
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import io
import requests

app = FastAPI()

model_id = "yifeihu/TB-OCR-preview-0.1"

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cuda", 
    trust_remote_code=True, 
    torch_dtype="auto", 
    _attn_implementation='flash_attention_2',
    quantization_config=BitsAndBytesConfig(load_in_4bit=True) # Optional: Load model in 4-bit mode to save memory
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
    trust_remote_code=True, 
    num_crops=16
) 

@app.post("/generate")
async def generate(prompt: str = Form(...), images: List[UploadFile] = File(...)):
    # Process images
    pil_images = []
    for img in images:
        image = Image.open(io.BytesIO(await img.read()))
        pil_images.append(image)
    
    # Prepare the prompt
    image_placeholders = "\n".join([f"<|image_{i+1}|>" for i in range(len(pil_images))])
    messages = [
        {"role": "user", "content": f"{image_placeholders}\n{prompt}"},
    ]
    
    full_prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process input
    inputs = processor(full_prompt, pil_images, return_tensors="pt").to("cuda:0")
    
    # Stream the response
    async def generate_stream():
        generation_args = { 
            "max_new_tokens": 1000, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 
        
        generate_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            **generation_args
        )
        
        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0]
        
        yield response

    return StreamingResponse(generate_stream(), media_type="text/plain")

@app.get("/")
async def root():
    return {"message": "Welcome to the Phi-3.5 Vision API"}
