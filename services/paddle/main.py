import os
import time

def raw_table():
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
    
def run_ocr_pipeline():
    from paddlex import create_pipeline
    pipeline_path = "./config/OCR.yaml"
    output_dir = "./output/pipeline/ocr"
    os.makedirs(output_dir, exist_ok=True)
    pipeline = create_pipeline(pipeline=pipeline_path)
    output = pipeline.predict("./input")
    
    counter = 0
    start = time.time()
    for res in output:
        input_path = res["input_path"]
        base_name = os.path.basename(input_path)
        img_path = os.path.join(output_dir, base_name)
        json_path = os.path.join(output_dir, f"{base_name}.json")
        res.save_to_img(img_path)
        res.save_to_json(json_path)
        counter += 1
        
    end = time.time()
    print(f"Time taken for inference: {end - start} seconds")
    print(f"Images per second: {counter / (end - start)}")
    
def run_table_pipeline():
    from paddlex import create_pipeline
    
    pipeline_path = "./config/table_recognition.yaml"
    output_dir = "./output/pipeline/table"
    os.makedirs(output_dir, exist_ok=True)
    pipeline = create_pipeline(pipeline=pipeline_path)

    start = time.time()
    output = pipeline.predict("./input")
    end = time.time()
    
    counter = 0
    
    for res in output:
        input_path = res["input_path"]
        base_name = os.path.basename(input_path)
        img_path = os.path.join(output_dir, base_name)
        json_path = os.path.join(output_dir, f"{base_name}.json")
        html_path = os.path.join(output_dir, f"{base_name}.html")
        res.save_to_img(img_path)
        res.save_to_json(json_path)
        res.save_to_html(html_path)
        counter += 1
        
    print(f"Time taken for inference: {end - start} seconds")
    print(f"Images per second: {counter / (end - start)}")
    
def ocr_service():
    import base64
    import glob
    import asyncio
    import aiohttp
    import json
    
    async def process_single_image(session, image_data):
        print(f"Processing image")
        async with session.post(API_URL, json={"image": image_data}) as response:
            print(await response.json())
            return await response.json()

    async def process_all_images(image_data_list):
        async with aiohttp.ClientSession() as session:
            tasks = [
                process_single_image(session, image_data)
                for image_data in image_data_list
            ]
            return await asyncio.gather(*tasks)

    API_URL = "http://localhost:8003/ocr"
    input_dir = "./input"
    output_dir = "./output/server/ocr"
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
                 glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))
    
    image_data_list = []
    file_paths = []
    for image_path in image_files:
        with open(image_path, "rb") as file:
            image_bytes = file.read()
            image_data = base64.b64encode(image_bytes).decode("ascii")
            image_data_list.append(image_data)
            file_paths.append(image_path)

    start = time.time()
    responses = asyncio.run(process_all_images(image_data_list))
    end = time.time()
    
    for image_path, response in zip(file_paths, responses):
        base_name = os.path.basename(image_path)
        output_json_path = os.path.join(output_dir, f"{base_name}.json")
        output_image_path = os.path.join(output_dir, base_name)
        result = response["result"]
        with open(output_json_path, "w") as file:
            json.dump(result, file, indent=2)
        with open(output_image_path, "wb") as file:
            file.write(base64.b64decode(result["image"]))

    total_time = end - start
    print(f"Average time taken: {total_time / len(image_files)} seconds")
    print(f"Images per second: {len(image_files) / total_time}")
        
def table_service():
    import base64
    import glob
    import asyncio
    import aiohttp
    import json

    async def process_single_image(session, image_data):
        print(f"Processing image")
        async with session.post(API_URL, json={"image": image_data}) as response:
            return await response.json()

    async def process_all_images(image_data_list):
        async with aiohttp.ClientSession() as session:
            tasks = [
                process_single_image(session, image_data)
                for image_data in image_data_list
            ]
            return await asyncio.gather(*tasks)

    API_URL = "http://localhost:8080/table-recognition"
    input_dir = "./input"
    output_dir = "./output/server/table"
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
                 glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))
    
    image_data_list = []
    file_paths = []
    for image_path in image_files:
        with open(image_path, "rb") as file:
            image_bytes = file.read()
            image_data = base64.b64encode(image_bytes).decode("ascii")
            image_data_list.append(image_data)
            file_paths.append(image_path)

    start = time.time()
    responses = asyncio.run(process_all_images(image_data_list))
    end = time.time()
    
    CSS_STYLES = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-family: Arial, sans-serif;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
    """
    
    for image_path, response in zip(file_paths, responses):
        base_name = os.path.basename(image_path)
        output_ocr_path = os.path.join(output_dir, f"ocr_{base_name}")
        output_layout_path = os.path.join(output_dir, f"layout_{base_name}")
        output_json_path = os.path.join(output_dir, f"{base_name}.json")
        output_html_path = os.path.join(output_dir, f"{base_name}.html")
        
        result = response["result"]
        with open(output_ocr_path, "wb") as file:
            file.write(base64.b64decode(result["ocrImage"]))
        with open(output_layout_path, "wb") as file:
            file.write(base64.b64decode(result["layoutImage"]))
        with open(output_json_path, "w") as file:
            json.dump(result["tables"], file, indent=2)
            
        tables = result["tables"]
        for i, table in enumerate(tables):
            table_html = table["html"]
            styled_html = table_html.replace("<html>", f"<html>\n<head>{CSS_STYLES}</head>")
            
            if len(tables) == 1:
                html_path = output_html_path
            else:
                html_path = os.path.join(output_dir, f"{base_name}_table{i+1}.html")
                
            with open(html_path, "w", encoding="utf-8") as file:
                file.write(styled_html)

    total_time = end - start
    print(f"Average time taken: {total_time / len(image_files)} seconds")
    print(f"Images per second: {len(image_files) / total_time}")
    
if __name__ == "__main__":
    # Uncomment one of the following lines to run the corresponding service
    # You must have images in ./input directory
    # Outputs will be in ./output directory
    
    ocr_service()
    # run_ocr_pipeline()
    # ocr_service()
    # table_service()
    # run_table_pipeline()
    # raw_table()
    pass