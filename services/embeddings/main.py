from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os

app = FastAPI()

# Initialize model and tokenizer globally
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Get environment variables with defaults
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "8192"))
TRUNCATE_DIM = int(os.getenv("TRUNCATE_DIM", "0"))  # 0 means no truncation

class EmbeddingRequest(BaseModel):
    inputs: str | List[str]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

@app.post("/embed")
async def create_embeddings(request: EmbeddingRequest):
    try:
        # Convert input to list if it's a single string
        texts = [request.inputs] if isinstance(request.inputs, str) else request.inputs

        # Tokenize the input texts
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        # Generate embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Apply mean pooling
        embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Truncate dimensions if specified
        if TRUNCATE_DIM > 0:
            embeddings = embeddings[:, :TRUNCATE_DIM]

        # Convert to list for JSON serialization
        embeddings_list = embeddings.cpu().numpy().tolist()

        return {"embeddings": embeddings_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
