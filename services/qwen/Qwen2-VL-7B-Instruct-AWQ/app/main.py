from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import requests
import base64
from typing import Dict, Any

app = FastAPI()

MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct-AWQ"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10},
    dtype="float16", 
    quantization="awq"  # Specify AWQ quantization
    
)

sampling_params = SamplingParams(
    temperature=0,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1256,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

@app.post("/generate")
async def generate(prompt: str = Form(...), images: List[UploadFile] = File(...)):
    # Process images
    pil_images = []
    for img in images:
        image = Image.open(io.BytesIO(await img.read()))
        pil_images.append(image)
    
    # Prepare the messages
    messages = [
        {"role": "system", "content": "You are great at reading charts, tables and images. You are being given multiple images and asked to answer a question about them. Respond for all."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ] + [{"type": "image", "image": img} for img in pil_images]
        }
    ]
    
    # Prepare the prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)
    
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    # Generate the response
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    return JSONResponse(content={"generated_text": generated_text})

@app.post("/generate/batch")
async def generate_batch(prompt: str = Form(...), images: List[UploadFile] = File(...)):
    # Process images
    pil_images = []
    for img in images:
        image = Image.open(io.BytesIO(await img.read()))
        pil_images.append(image)
    
    # Prepare individual messages for each image
    messages = []
    for img in pil_images:
        message = [
            {"role": "system", "content": "You are great at reading charts, tables and images. You are being given an image and asked to answer a question about it. You must be prepared to respond in JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": img}
                ]
            }
        ]
        messages.append(message)
    
    # Prepare the prompts
    prompts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    
    image_inputs, _ = process_vision_info(messages)
    
    llm_inputs = []
    for prompt, image_input in zip(prompts, image_inputs):
        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": [image_input]}
        })

    # Generate the responses using llm.generate
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    
    generated_texts = [output.outputs[0].text for output in outputs]

    return JSONResponse(content={"generated_texts": generated_texts})

@app.get("/")
async def root():
    return {"message": "Welcome to the Qwen2-VL-2B-Instruct-AWQ API"}