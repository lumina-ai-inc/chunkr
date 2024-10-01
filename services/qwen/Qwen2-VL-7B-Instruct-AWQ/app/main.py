from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import json
import os

from typing import Dict, Any

app = FastAPI()

MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10},
    dtype="float16",
    # quantization="awq"  # Specify AWQ quantization
)

sampling_params = SamplingParams(
    temperature=0,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1256,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

@app.post("/v1/chat/completions")
async def generate(messages: str = Form(...), prompt: str = Form(...), images: List[UploadFile] = File(...)):
    # Parse messages from JSON string
    messages = json.loads(messages)

    # Process images
    pil_images = []
    for img in images:
        image = Image.open(io.BytesIO(await img.read()))
        pil_images.append(image)

    # Add images to the messages
    for i, image in enumerate(pil_images):
        messages[i]["content"][0]["image"] = image

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
    
    # Prepare the list of response dictionaries
    responses = []
    for output in outputs:
        for generated_output in output.outputs:
            response = {
                "generated_text": generated_output.text,
                "finish_reason": generated_output.finish_reason,
                "token_count": len(generated_output.token_ids)
            }
            responses.append(response)

    return JSONResponse(content={"responses": responses})

@app.get("/")
async def root():
    return {"message": "Welcome to the Qwen2-VL-2B-Instruct-AWQ API"}