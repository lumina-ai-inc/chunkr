import argparse
from PIL import Image
from robyn import Robyn, Request

from src.get_models import init_structure_model, init_bbox_model, init_content_model
from src.inference import run_structure_inference, run_bbox_inference, run_content_inference
from src.utils import build_table_from_html_and_cell

app = Robyn(__file__)

structure_model = init_structure_model()
bbox_model = init_bbox_model()
content_model = init_content_model()


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/structure/html")
async def extract_structure(request: Request):
    if structure_model is None:
        return {"error": "Structure model not initialized"}

    image = next(iter(request.files.values()), None)
    if image is None:
        return {"error": "Image not found in request"}

    image = Image.open(image).convert("RGB")
    result = run_structure_inference(structure_model, image)
    return result


@app.post("/bbox")
async def extract_bbox(request: Request):
    if bbox_model is None:
        return {"error": "BBox model not initialized"}

    image = next(iter(request.files.values()), None)
    if image is None:
        return {"error": "Image not found in request"}

    image = next(iter(request.files.values()), None)
    result = run_bbox_inference(bbox_model, image)

    if content_model is not None:
        result = run_content_inference(content_model, image, result)

    return result


@app.post("/table/html")
async def extract_table(request: Request):
    if structure_model is None:
        return {"error": "Structure model not initialized"}
    if bbox_model is None:
        return {"error": "BBox model not initialized"}


    image = next(iter(request.files.values()), None)
    if image is None:
        return {"error": "Image not found in request"}

    image = Image.open(image).convert("RGB")
    structure = run_structure_inference(structure_model, image)
    bbox = run_bbox_inference(bbox_model, image, structure)
    if content_model is not None:
        content = run_content_inference(content_model, image, bbox)
    else:
        content = [""] * len(bbox)

    result = build_table_from_html_and_cell(structure, content)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the OCR service")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")

    args = parser.parse_args()

    app.start(host=args.host, port=args.port)
