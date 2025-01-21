import asyncio
import json
from datetime import datetime
from pathlib import Path
import logging
from chunkr_ai import ChunkrAsync
from chunkr_ai.models import Status
from asyncio import Queue
from dataclasses import dataclass, field
import csv

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

chunkr = ChunkrAsync()
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
            with open(stats_file, "a") as f:
                f.write(log_entry)
        completed_queue.task_done()

async def process_single_url(file_name: str, presigned_url: str, completed_queue: Queue):
    try:
        task = await chunkr.create_task(presigned_url)
        task_id = task.task_id
        try:
            await task.poll()
            if task.status == Status.SUCCEEDED:
                if task.page_count is None:
                    stats.total_pages += 1
                stats.successful_tasks += 1
            else:
                logger.error(f"Task failed {task_id}: {task.status}")
                stats.failed_tasks += 1
        except Exception as e:
            stats.failed_tasks += 1
            logger.error(f"Task failed {task_id}: {str(e)}")
        task = await chunkr.get_task(task_id)
        task_data = task.model_dump(mode='json')
        task_data['file_name'] = file_name
        await completed_queue.put(task_data)
    except Exception as e:
        stats.failed_tasks += 1
        logger.error(f"Error processing {file_name}: {str(e)}")
        return

async def process_all_files(input_csv: str, output_dir: str, max_concurrent: int = None, max_files: int = None):
    completed_queue = Queue()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = run_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    tasks_jsonl = output_path / "tasks.jsonl"
    
    writer_task = asyncio.create_task(write_and_log_worker(tasks_jsonl, run_dir, completed_queue))
    
    files = []
    with open(input_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            files.append((row['file_name'], row['presigned_url']))
    
    if max_files is not None:
        files = files[:max_files]
    
    concurrent_limit = max_concurrent if max_concurrent is not None else len(files)
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    async def bounded_process(file_name: str, presigned_url: str):
        async with semaphore:
            await process_single_url(file_name, presigned_url, completed_queue)
    
    tasks = [bounded_process(file_name, presigned_url) for file_name, presigned_url in files]
    await asyncio.gather(*tasks)
    
    await completed_queue.join()
    
    writer_task.cancel()
    try:
        await writer_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process files using Chunkr')
    parser.add_argument('--input', default="input/presigned_urls.csv",
                      help='Input CSV file path (default: input/presigned_urls.csv)')
    parser.add_argument('--output', default="output",
                      help='Output directory name (default: output)')
    parser.add_argument('--concurrent', type=int, default=None,
                      help='Maximum number of concurrent processes (default: all files)')
    parser.add_argument('--max-files', type=int, default=None,
                      help='Maximum number of files to process (default: all files)')
    
    args = parser.parse_args()
    
    asyncio.run(process_all_files(args.input, args.output, args.concurrent, args.max_files)) 