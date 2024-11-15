import asyncio
import aiohttp
import json
from pathlib import Path
import time

async def process_image(session: aiohttp.ClientSession, image_path: Path, url: str) -> dict:
    mpwriter = aiohttp.MultipartWriter('form-data')
    with open(image_path, 'rb') as f:
        part = mpwriter.append(f)
        part.set_content_disposition('form-data', name='file', filename=image_path.name)
        
        start_time = time.time()
        async with session.post(url, data=mpwriter) as response:
            result = await response.json()
            processing_time = time.time() - start_time
            
            # Save result to JSON file
            output_dir = Path('output/server')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{image_path.stem}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Processed {image_path.name} in {processing_time:.2f}s")
            return result

async def main():
    input_dir = Path('input/ocr')
    url = 'http://localhost:8000/ocr'
    
    image_files = [
        f for f in input_dir.glob('*') 
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    ]
    
    if not image_files:
        print("No images found in input directory!")
        return
    # Process all images concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_image(session, image_path, url)
            for image_path in image_files
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"\nProcessed {len(image_files)} images in {total_time:.2f}s")
        print(f"Average time per image: {total_time/len(image_files):.2f}s")
        print(f"Average images per second: {len(image_files) / total_time:.2f} images/s")

async def run_multiple_mains(n: int, sleep_time: float | None = None):
    tasks = []
    for _ in range(n):
        tasks.append(asyncio.create_task(main()))
        if sleep_time:
            await asyncio.sleep(sleep_time)  
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    n_instances = 1
    sleep_time = None
    asyncio.run(run_multiple_mains(n_instances, sleep_time))
