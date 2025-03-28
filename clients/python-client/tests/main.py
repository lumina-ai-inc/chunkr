from chunkr_ai import Chunkr
from chunkr_ai.models import Tokenizer, ChunkProcessing
import asyncio
import multiprocessing
import time
import base64
import os

chunkr = Chunkr()

async def upload_file_async(file_path):
    task = await chunkr.upload(file_path)
    print(f"Uploaded {file_path}: {task.task_id}")
    return task.task_id

async def test_async_concurrent():
    print("\nTesting async concurrency...")
    start_time = time.time()
    
    # Create multiple upload tasks
    tasks = [
        upload_file_async("./files/test.pdf") for _ in range(5)
    ]
    # Run them concurrently
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"Async concurrent time: {end_time - start_time:.2f} seconds")
    return results

async def test_async_concurrent_with_tasks(tasks):
    print("\nTesting async concurrency...")
    start_time = time.time()
    
    # Run them concurrently
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"Async concurrent time: {end_time - start_time:.2f} seconds")
    return results

def upload_file_sync(file_path):
    task = chunkr.upload(file_path)
    print(f"Uploaded {file_path}: {task.task_id}")
    task.poll()
    return task.task_id

def test_multiprocessing():
    print("\nTesting multiprocessing...")
    start_time = time.time()
    
    with multiprocessing.Pool(processes=5) as pool:
        files = ["./files/test.pdf"] * 5
        results = pool.map(upload_file_sync, files)
    
    end_time = time.time()
    print(f"Multiprocessing time: {end_time - start_time:.2f} seconds")
    return results

def process_worker(file_path):
    # Each process will run its own async event loop
    async def run_uploads():
        tasks = [upload_file_async(file_path) for _ in range(5)]
        return await asyncio.gather(*tasks)
    
    return asyncio.run(run_uploads())

def test_multiprocess_with_async():
    print("\nTesting multiprocessing with async concurrency...")
    start_time = time.time()
    
    with multiprocessing.Pool(processes=3) as pool:
        files = ["./files/test.pdf"] * 3  # One file path per process
        results = pool.map(process_worker, files)
    
    end_time = time.time()
    print(f"Multiprocessing with async time: {end_time - start_time:.2f} seconds")
    return results

def save_base64_to_file():
    task = chunkr.upload("./files/test.pdf")
    task = chunkr.get_task(task.task_id, base64_urls=True)
    print(task.configuration)
    with open("./output/task.json", "w") as f:
        f.write(task.model_dump_json())
    pdf_url = task.output.pdf_url
    pdf_data = base64.b64decode(pdf_url)
    output_path = "./output/converted.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(pdf_data)
   
    segment_image = task.output.chunks[0].segments[0].image
    segment_image_data = base64.b64decode(segment_image)
    output_path = "./output/segment_image.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(segment_image_data)

if __name__ == "__main__":
    
    # task = chunkr.upload("./files/test.pdf")
    # task.markdown("./output/markdown.md")
    # print(task.output.chunks[1].embed)
    # print(task.output.chunks[0].segments[0].confidence)
    tokenizers = [Tokenizer.WORD, Tokenizer.CL100K_BASE, Tokenizer.XLM_ROBERTA_BASE, "Qwen/Qwen-tokenizer", "Word"]
    for tokenizer in tokenizers:
        print(tokenizer)
        chunk_processing = ChunkProcessing(
            target_length=500,
            tokenizer=tokenizer
        )
        print(chunk_processing)
        print(chunk_processing.model_dump_json())
        print("\n")

    tokenizer = tokenizers[1]
    print(tokenizer)
    chunk_processing = ChunkProcessing(
        tokenizer=tokenizer
    )
    task = chunkr.upload("./files/test.pdf", chunk_processing)
    task.markdown("./output/markdown.md")
    print(task.output.chunks[1].embed)
    print(task.configuration.chunk_processing)
    print(task.configuration.chunk_processing.tokenizer)
    assert task.configuration.chunk_processing.tokenizer == tokenizer