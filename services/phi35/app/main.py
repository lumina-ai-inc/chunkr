from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image
import io

app = FastAPI()

# Load the model and processor
model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=16)

class ChatMessage(BaseModel):
    role: str
    content: str

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
        for token in model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            streamer=None  # We'll implement our own streaming logic
        ):
            yield processor.tokenizer.decode([token], skip_special_tokens=True)

    return StreamingResponse(generate_stream(), media_type="text/plain")

@app.get("/")
async def root():
    return {"message": "Welcome to the Phi-3.5 Vision API"}
