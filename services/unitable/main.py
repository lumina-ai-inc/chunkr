import argparse
from PIL import Image
from robyn import Robyn, Request

from src.get_models import init_structure_model, init_bbox_model, init_content_model
from src.inference import run_structure_inference, run_bbox_inference, run_content_inference

app = Robyn(__file__)

structure_model = init_structure_model()
bbox_model = init_bbox_model()
content_model = init_content_model()


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/structure")
async def extract_structure(request: Request):
    if structure_model is None:
        return {"error": "Structure model not initialized"}

    image = request.files.get("image")
    if image is None:
        return {"error": "Image not found in request"}

    image = Image.open(image).convert("RGB")
    result = run_structure_inference(structure_model, image)
    return result


@app.post("/bbox")
async def extract_bbox(request: Request):
    if bbox_model is None:
        return {"error": "BBox model not initialized"}

    image = request.files.get("image")
    if image is None:
        return {"error": "Image not found in request"}

    image = Image.open(image).convert("RGB")
    result = run_bbox_inference(bbox_model, image)
    return result


@app.post("/content")
async def extract_content(request: Request):
    if content_model is None:
        return {"error": "Content model not initialized"}

    image = request.files.get("image")
    if image is None:
        return {"error": "Image not found in request"}

    image = Image.open(image).convert("RGB")
    result = run_content_inference(content_model, image)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the OCR service")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")

    args = parser.parse_args()

    app.start(host=args.host, port=args.port)
