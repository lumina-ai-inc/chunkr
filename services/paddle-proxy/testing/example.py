import os
import time
import base64
import glob
import asyncio
import aiohttp
import json

API_URLS = {
    "ocr": "http://localhost:8000/ocr",
    "table": "http://localhost:8000/table-recognition"
}

async def process_single_image(session, url, image_data):
    print(f"Processing image")
    try:
        async with session.post(url, json={"image": image_data}) as response:
            response.raise_for_status()
            content_types = ['application/json']
            if response.content_type in content_types:
                return await response.json()
            else:
                text = await response.text()
                print(f"Unexpected response type: {response.content_type}")
                print(f"Response text: {text}")
                return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

async def process_all_images(url, image_data_list):
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_single_image(session, url, image_data)
            for image_data in image_data_list
        ]
        return await asyncio.gather(*tasks)

def process_images(service_type):
    input_dir = "./input"
    output_dir = f"./output/server/{service_type}"
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
    responses = asyncio.run(process_all_images(API_URLS[service_type], image_data_list))
    end = time.time()
    
    if service_type == "ocr":
        process_ocr_responses(responses, file_paths, output_dir)
    else:
        process_table_responses(responses, file_paths, output_dir)

    total_time = end - start
    print(f"Average time taken: {total_time / len(image_files)} seconds")
    print(f"Images per second: {len(image_files) / total_time}")

def process_ocr_responses(responses, file_paths, output_dir):
    for image_path, response in zip(file_paths, responses):
        base_name = os.path.basename(image_path).split(".")[0]
        output_json_path = os.path.join(output_dir, f"{base_name}.json")
        output_image_path = os.path.join(output_dir, f"{base_name}.jpg")
        if response is not None:
            result = response["result"]
            with open(output_json_path, "w") as file:
                json.dump(result, file, indent=2)
            with open(output_image_path, "wb") as file:
                file.write(base64.b64decode(result["image"]))
        else:
            print(f"Error processing image: {image_path}")

def process_table_responses(responses, file_paths, output_dir):
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
    </style>
    """
    
    for image_path, response in zip(file_paths, responses):
        base_name = os.path.basename(image_path).split(".")[0]
        output_raw_path = os.path.join(output_dir, f"raw_{base_name}.json")
        output_ocr_path = os.path.join(output_dir, f"ocr_{base_name}.jpg")
        output_layout_path = os.path.join(output_dir, f"layout_{base_name}.jpg")
        output_json_path = os.path.join(output_dir, f"{base_name}.json")
        output_html_path = os.path.join(output_dir, f"{base_name}.html")
        
        if response is not None:
            with open(output_raw_path, "wb") as file:
                file.write(json.dumps(response).encode("utf-8"))
            if "result" not in response:
                print(f"Error processing image: {image_path}")
                continue
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

if __name__ == "__main__":
    print("Choose a service to run:")
    print("1. OCR Service")
    print("2. Table Service")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        print("Running OCR Service...")
        process_images("ocr")
    elif choice == "2":
        print("Running Table Service...")
        process_images("table")
    else:
        print("Invalid choice. Please enter 1 or 2.")