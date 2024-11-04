import os
import time

def main():
    from paddlex import create_model
    
    model = create_model("SLANet_plus")
    start = time.time()
    output = model.predict("./input", batch_size=1)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    counter = 0
    for idx, res in enumerate(output):
        input_path = res["input_path"]
        base_name = os.path.basename(input_path)
        img_path = f"./output/table/{base_name}"
        json_path = f"./output/table/{base_name}.json"
        html_path = f"./output/table/{base_name}.html"
        
        res.save_to_img(img_path)
        res.save_to_json(json_path)
        
        structure = "\n".join(res["structure"])
        with open(html_path, "w") as f:
            f.write(structure)
        counter += 1
    print(f"Images per second: {counter / (end - start)}")
    
def ocr():
    from paddlex import create_pipeline
    
    pipeline = create_pipeline(pipeline="OCR")
    start = time.time()
    counter = 0
    
    output = pipeline.predict("./input")
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    
    for res in output:
        input_path = res["input_path"]
        base_name = os.path.basename(input_path)
        img_path = f"./output/ocr/{base_name}"
        json_path = f"./output/ocr/{base_name}.json"
        res.save_to_img(img_path)
        res.save_to_json(json_path)
        counter += 1
    
    print(f"Images per second: {counter / (end - start)}")
    
def ocr_service():
    import base64
    import requests

    API_URL = "http://localhost:8080/ocr"
    image_path = "./input/0ba94b7d-30fb-46d6-9c35-7aa0cd4b20f4.jpg"
    output_image_path = "./output/server/ocr/0ba94b7d-30fb-46d6-9c35-7aa0cd4b20f4.jpg"
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    with open(image_path, "rb") as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode("ascii")

    payload = {"image": image_data}

    response = requests.post(API_URL, json=payload)

    assert response.status_code == 200
    result = response.json()["result"]
    with open(output_image_path, "wb") as file:
        file.write(base64.b64decode(result["image"]))
    print(f"Output image saved at {output_image_path}")
    print("\nDetected texts:")
    print(result["texts"])
    
if __name__ == "__main__":
    ocr_service()