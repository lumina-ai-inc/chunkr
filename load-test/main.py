import asyncio
import json
from datetime import datetime
from pathlib import Path
import logging
from chunkr_ai import Chunkr
from chunkr_ai.models import Status
from asyncio import Queue
from dataclasses import dataclass, field
import multiprocessing
from itertools import islice

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

chunkr = Chunkr()
file_lock = asyncio.Lock()

@dataclass
class ProcessingStats:
    start_time: datetime = field(default_factory=datetime.now)
    total_pages: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_tasks: int = 0
    elapsed_seconds: float = 0
    
    def calculate_pages_per_second(self) -> float:
        self.elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
        return self.total_pages / self.elapsed_seconds if self.elapsed_seconds > 0 else 0

    def format_elapsed_time(self) -> str:
        hours, remainder = divmod(int(self.elapsed_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

stats = ProcessingStats()

async def write_and_log_worker(output_jsonl: Path, run_dir: Path, completed_queue: Queue):
    stats_file = run_dir / "stats.log"
    while True:
        task_data = await completed_queue.get()
        async with file_lock:
            with open(output_jsonl, "a") as f:
                json.dump(task_data, f)
                f.write("\n")
            if task_data.get("status") == Status.SUCCEEDED:
                stats.successful_tasks += 1
                stats.total_pages += task_data.get("output", {}).get("page_count", 0)
            else:
                stats.failed_tasks += 1
            stats.total_tasks += 1
            pages_per_second = stats.calculate_pages_per_second()
            log_entry = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Pages/sec: {pages_per_second:.2f}, "
                f"Total pages: {stats.total_pages}, "
                f"Time elapsed: {stats.format_elapsed_time()}, "
                f"Succeeded: {stats.successful_tasks}, "
                f"Failed: {stats.failed_tasks}, "
                f"Total tasks: {stats.total_tasks}\n"
            )
            print(log_entry)
            with open(stats_file, "a") as f:
                f.write(log_entry)
        completed_queue.task_done()

async def process_single_file(file_path: Path, completed_queue: Queue):
    try:
        task = await chunkr.upload(file_path)
        task_data = task.model_dump(mode='json')
        await completed_queue.put(task_data)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return

async def process_files_chunk(files: list[Path], output_dir: str, concurrent_limit: int, chunk_id: int):
    """Process a chunk of files in a separate process"""
    # Initialize a new Chunkr instance for this process
    chunkr = Chunkr()
    completed_queue = Queue()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = run_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    tasks_jsonl = output_path / f"tasks_chunk_{chunk_id}.jsonl"
    
    writer_task = asyncio.create_task(write_and_log_worker(tasks_jsonl, run_dir, completed_queue))
    
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    async def bounded_process(file_path: Path):
        async with semaphore:
            await process_single_file(file_path, completed_queue)
    
    tasks = [bounded_process(file_path) for file_path in files]
    await asyncio.gather(*tasks)
    
    await chunkr.close()
    await completed_queue.join()
    
    writer_task.cancel()
    try:
        await writer_task
    except asyncio.CancelledError:
        pass

def process_chunk_wrapper(chunk_data):
    """Wrapper function to run asyncio event loop in a separate process"""
    files, output_dir, concurrent_limit, chunk_id = chunk_data
    asyncio.run(process_files_chunk(files, output_dir, concurrent_limit, chunk_id))

async def process_all_files(input_folder: str, output_dir: str, num_threads: int = 1, max_concurrent: int = None, max_files: int = None):
    print(f"Processing {input_folder} to {output_dir} with {num_threads} threads, {max_concurrent} concurrent tasks per thread")
    
    input_path = Path(input_folder)
    files = list(input_path.glob("*.*"))
    if max_files is not None:
        files = files[:max_files]
    
    # Calculate files per thread and concurrent tasks per thread
    files_per_thread = len(files) // num_threads + (1 if len(files) % num_threads else 0)
    concurrent_per_thread = max_concurrent if max_concurrent is not None else files_per_thread
    
    print(f"Processing {len(files)} files, {files_per_thread} files per thread")
    
    # Split files into chunks for each thread
    file_chunks = [list(islice(files, i, i + files_per_thread)) 
                  for i in range(0, len(files), files_per_thread)]
    
    # Create process pool and distribute work
    with multiprocessing.Pool(processes=num_threads) as pool:
        chunk_data = [(chunk, output_dir, concurrent_per_thread, i) 
                     for i, chunk in enumerate(file_chunks)]
        pool.map(process_chunk_wrapper, chunk_data)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process files using Chunkr')
    parser.add_argument('--input', default="./input",
                      help='Input folder path (default: input)')
    parser.add_argument('--output', default="./output",
                      help='Output directory name (default: output)')
    parser.add_argument('--threads', type=int, 
                      default=multiprocessing.cpu_count(),
                      help=f'Number of processing threads (default: {multiprocessing.cpu_count()} - system CPU count)')
    parser.add_argument('--concurrent', type=int, default=None,
                      help='Maximum number of concurrent processes per thread (default: all files)')
    parser.add_argument('--max-files', type=int, default=None,
                      help='Maximum number of files to process (default: all files)')
    
    args = parser.parse_args()
    
    asyncio.run(process_all_files(
        args.input, 
        args.output, 
        args.threads, 
        args.concurrent, 
        args.max_files
    )) 