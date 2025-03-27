import asyncio
import aiohttp
import json
from pathlib import Path
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

async def process_image(session: aiohttp.ClientSession, image_path: Path, url: str) -> dict:
    mpwriter = aiohttp.MultipartWriter('form-data')
    with open(image_path, 'rb') as f:
        part = mpwriter.append(f)
        part.set_content_disposition('form-data', name='files', filename=image_path.name)
        
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
            
            # Plot the results
            plot_results(image_path, result[0], output_dir)
            
            print(f"Processed {image_path.name} in {processing_time:.2f}s")
            return result

def plot_results(image_path, ocr_result, output_dir):
    img = Image.open(image_path)
    img_np = np.array(img)
    
    fig, ax = plt.subplots(1, figsize=(15, 15))
    ax.imshow(img_np)
    
    page_content = ocr_result["page_content"]
    img_h, img_w = img_np.shape[:2]
    
    for block in page_content["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                geometry = word["geometry"]
                
                # Convert normalized points to image coordinates
                points = np.array(geometry) * np.array([img_w, img_h])
                
                # Draw polygon instead of rectangle for better accuracy
                polygon = patches.Polygon(
                    points, closed=True, 
                    linewidth=1, edgecolor='r', facecolor='none'
                )
                ax.add_patch(polygon)
                
                # Calculate top-left for text placement
                x_min, y_min = points.min(axis=0)
                
                # Add text above the word
                plt.text(
                    x_min, y_min - 5, word["value"], 
                    color='blue', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7)
                )
    
    plt.axis('off')
    output_plot = output_dir / f"{image_path.stem}_plot.png"
    plt.savefig(output_plot, bbox_inches='tight', dpi=200)
    plt.close(fig)

async def main():
    input_dir = Path('input/ocr')
    url = 'http://localhost:8000/batch'
    
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
