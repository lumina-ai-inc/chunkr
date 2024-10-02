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
    limit_mm_per_prompt={"image": 10, "video": 10},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=2024,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

@app.post("/v1/chat/completions")
async def generate(messages: str = Form(...), images: List[UploadFile] = File(...)):
    # Parse messages from JSON string
    messages_batch = json.loads(messages)

    # Process images
    pil_images = []
    for img in images:
        image = Image.open(io.BytesIO(await img.read()))
        pil_images.append(image)

    # Add images to the messages
    for i, batch in enumerate(messages_batch):
        for j, message in enumerate(batch):
            if message["role"] == "user":
                for k, content in enumerate(message["content"]):
                    if content["type"] == "image":
                        messages_batch[i][j]["content"][k]["image"] = pil_images[i]

    # Prepare the prompts
    prompts = [
        processor.apply_chat_template(
            batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        for batch in messages_batch
    ]

    image_inputs, _ = process_vision_info(messages_batch)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    llm_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        for prompt in prompts
    ]

    # Generate the responses
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    
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
    return {"message": "Welcome to the Qwen2-VL-7B-Instruct API"}