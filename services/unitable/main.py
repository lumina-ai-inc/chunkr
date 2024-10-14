import argparse
from io import BytesIO
from PIL import Image
from robyn import Robyn, Request

from src.get_models import init_structure_model, init_bbox_model, init_content_model
from src.inference import run_structure_inference, run_bbox_inference, run_content_inference
from src.utils import build_table_from_html_and_cell, html_table_template

app = Robyn(__file__)

structure_model = init_structure_model()
bbox_model = init_bbox_model()
content_model = init_content_model()


def get_image_from_request(request: Request) -> Image.Image:
    image_name = "test.jpg"
    image_path = f"{image_name}"
    image = Image.open(image_path).convert("RGB")
    return image
    image_file = next(iter(request.files.values()), None)
    if image_file is None:
        raise ValueError("Image not found in request")
    return Image.open(BytesIO(image_file)).convert("RGB")


def check_models(*models):
    for model in models:
        if model is None:
            raise ValueError(f"{model.__name__} not initialized")


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/structure")
async def extract_structure(request: Request):
    try:
        check_models(structure_model)
        image = get_image_from_request(request)
        result = run_structure_inference(structure_model, image)
        return result
    except ValueError as e:
        return {"error": str(e)}

@app.post("/bbox")
async def extract_bbox(request: Request):
    try:
        check_models(bbox_model)
        image = get_image_from_request(request)
        result = run_bbox_inference(bbox_model, image)

        if content_model is not None:
            result = run_content_inference(content_model, image, result)

        return result
    except ValueError as e:
        return {"error": str(e)}


@app.post("/table/html")
async def extract_table(request: Request):
    try:
        check_models(structure_model, bbox_model)
        image = get_image_from_request(request)
        structure = run_structure_inference(structure_model, image)
        bbox = run_bbox_inference(bbox_model, image)

        if content_model is not None:
            content = run_content_inference(content_model, image, bbox)
        else:
            content = [""] * len(bbox)

        result = build_table_from_html_and_cell(structure, content)
        return result
    except ValueError as e:
        return {"error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the OCR service")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")

    args = parser.parse_args()

    app.start(host=args.host, port=args.port)
