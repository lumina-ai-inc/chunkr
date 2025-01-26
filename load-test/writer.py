from datetime import datetime
from typing import Dict, List
from pathlib import Path
import json

from redis_utils import RedisManager
from models import WritePayload

class WriterStats:
    def __init__(self):
        self.tasks: List[WritePayload] = []
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_pages = 0
        self.successful_pages = 0
        self.failed_pages = 0
        
    def add_task(self, payload: WritePayload):
        self.tasks.append(payload)
        try:
            with open(payload.task_path, 'r') as f:
                task_data = json.load(f)
            
            page_count = task_data.get('output', {}).get('page_count', 0)
            status = task_data.get('status')
            
            self.total_pages += page_count
            
            if status == 'Succeeded':
                self.successful_tasks += 1
                self.successful_pages += page_count
            else:
                self.failed_tasks += 1
                self.failed_pages += page_count
                
        except Exception as e:
            print(f"Error processing task file {payload.task_path}: {e}")
            self.failed_tasks += 1
    
    def calculate_rates(self, elapsed_seconds: float) -> Dict[str, float]:
        return {
            'successful_pages_per_second': self.successful_pages / elapsed_seconds if elapsed_seconds > 0 else 0,
            'failed_pages_per_second': self.failed_pages / elapsed_seconds if elapsed_seconds > 0 else 0,
            'pages_per_second': self.total_pages / elapsed_seconds if elapsed_seconds > 0 else 0
        }

def main():
    redis_manager = RedisManager()
    stats = WriterStats()
    for data in redis_manager.read_from_writer_queue():
        if data is None:
            continue
        try:
            data_dict = json.loads(data) if isinstance(data, str) else data
            payload = WritePayload.model_validate(data_dict)
            stats.add_task(payload)
            
            start_time = min(datetime.fromisoformat(task.start_time) for task in stats.tasks)
            end_time = max(datetime.fromisoformat(task.end_time) for task in stats.tasks)
            elapsed_seconds = (end_time - start_time).total_seconds()
            
            rates = stats.calculate_rates(elapsed_seconds)
            
            log_entry = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Successful pages/sec: {rates['successful_pages_per_second']:.2f}, "
                f"Failed pages/sec: {rates['failed_pages_per_second']:.2f}, "
                f"Total pages/sec: {rates['pages_per_second']:.2f}, "
                f"Total pages: {stats.total_pages}, "
                f"Failed pages: {stats.failed_pages}, "
                f"Successful pages: {stats.successful_pages}, "
                f"Time elapsed: {elapsed_seconds:.1f}s, "
                f"Succeeded: {stats.successful_tasks}, "
                f"Failed: {stats.failed_tasks}\n"
            )
            with open(payload.stats_path, 'a') as f:
                print(f"Stats file updated: {payload.stats_path}")
                print(log_entry)
                f.write(log_entry)
                
        except Exception as e:
            print(f"Error processing write payload: {e}")
            continue

if __name__ == "__main__":
    main()
