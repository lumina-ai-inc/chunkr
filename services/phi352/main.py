from fastapi import FastAPI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

app = FastAPI()

model_dir = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512,  # Note: Update this as per your use-case
    do_fuse=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=quantization_config,
)


@app.get("/")
async def root():
    is_cuda_available = torch.cuda.is_available()
    return {
        "message": "Hello World",
        "cuda_available": is_cuda_available,
    }


@app.get("/readiness")
async def readiness():
    return {"status": "ready"}


# an inference endpoint for text generation
@app.post("/generate")
async def generate_text(data: dict):
    text = data.get("text")
    if not text:
        return {"error": "text field is required"}
    prompt = text

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=256)
    response = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]
    return {"generated_text": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
