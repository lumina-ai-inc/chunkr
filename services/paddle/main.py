import asyncio
import base64
import json
import os
from paddlex import create_pipeline
from robyn import Robyn
import tempfile

app = Robyn(__file__)


@app.get("/")
async def h(request):
    return "Hello, world!"

@app.post("/ocr")
async def ocr(request):
    print("Processing image")
    image = request.json()["image"]
    image_bytes = base64.b64decode(image)
    results = []
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        temp_file.write(image_bytes)
        temp_file.close() 
        
        for result in pipeline.predict(temp_file.name):
            results.append(result)
            result.save_to_json(output_file.name)
        with open(output_file.name, 'r') as f:
            raw_result = json.load(f)
        
        texts = []
        for poly, score, text in zip(raw_result['dt_polys'], 
                                    raw_result['dt_scores'], 
                                    raw_result['rec_text']):
            texts.append({
                "poly": poly,
                "text": text,
                "score": float(score)
            })
            
        response = {
            "logId": "", 
            "errorCode": 0,
            "errorMsg": "",
            "result": {
                "texts": texts,
                "image": ""  
            }
        }
        return response
    finally:
        os.unlink(temp_file.name)
        os.unlink(output_file.name)
            
@app.post("/table-recognition")
async def table(request):
    print("Processing image")
    image = request.json()["image"]
    image_bytes = base64.b64decode(image)
    results = []
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    # output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    output_file = "./output/server-output/result.html"
    try:
        temp_file.write(image_bytes)
        temp_file.close() 
        
        for result in pipeline.predict(temp_file.name):
            results.append(result)
            result.save_to_html(output_file)
        with open(output_file, 'r') as f:
            raw_result = json.load(f)
        
        response = {
            "logId": "", 
            "errorCode": 0,
            "errorMsg": "",
            "result": {
                "tables": raw_result,
                "image": ""  
            }
        }
        return response
    finally:
        os.unlink(temp_file.name)
        # os.unlink(output_file.name)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--pipeline", type=str, default="config/OCR.yaml")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--use_hpip", type=bool, default=False)
    args = parser.parse_args()
    pipeline = create_pipeline(pipeline=args.pipeline, device=args.device, use_hpip=args.use_hpip)
    print("Server started...")
    app.start(port=args.port, host=args.host)
