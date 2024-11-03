import argparse
from io import BytesIO
from paddlex import create_model
from PIL import Image
from robyn import Robyn
import tempfile

app = Robyn(__file__)

model = create_model("SLANet_plus")

@app.get("/")
def health(request):
    return "OK"


@app.post("/ocr")
def extract(request):
    files = request.files
    if not files:
        print("No files provided in the request.")
        return {"error": "No files provided."}
    file_key, file_content = next(iter(files.items()))
    image = Image.open(BytesIO(file_content))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        output = model.predict(temp_file.name, batch_size=1)
    for res in output:
        res.print(json_format=False)
        res.save_to_img("./output/")
        res.save_to_json("./output/res.json")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app.start(host=args.host, port=args.port)
