from datetime import datetime
from typing import Dict, List
from pathlib import Path
import json

from redis_utils import RedisManager
from models import WritePayload

class WriterStats:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.tasks: List[WritePayload] = []
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_pages = 0
        self.successful_pages = 0
        self.failed_pages = 0
        self.max_successful_pages_per_second = 0
        self.max_failed_pages_per_second = 0
        self.max_pages_per_second = 0
        
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
        self.elapsed_seconds = elapsed_seconds
        if elapsed_seconds > 0:
            current_successful_rate = self.successful_pages / elapsed_seconds
            current_failed_rate = self.failed_pages / elapsed_seconds
            current_total_rate = self.total_pages / elapsed_seconds
            
            self.max_successful_pages_per_second = max(self.max_successful_pages_per_second, current_successful_rate)
            self.max_failed_pages_per_second = max(self.max_failed_pages_per_second, current_failed_rate)
            self.max_pages_per_second = max(self.max_pages_per_second, current_total_rate)
            
            return {
                'successful_pages_per_second': current_successful_rate,
                'failed_pages_per_second': current_failed_rate,
                'pages_per_second': current_total_rate,
                'max_successful_pages_per_second': self.max_successful_pages_per_second,
                'max_failed_pages_per_second': self.max_failed_pages_per_second,
                'max_pages_per_second': self.max_pages_per_second
            }
        return {
            'successful_pages_per_second': 0,
            'failed_pages_per_second': 0,
            'pages_per_second': 0,
            'max_successful_pages_per_second': 0,
            'max_failed_pages_per_second': 0,
            'max_pages_per_second': 0
        }

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary representation of the stats."""
        return {
            'run_id': self.run_id,
            'successful': self.successful_tasks,
            'failed': self.failed_tasks,
            'total': len(self.tasks),
            'total_pages': self.total_pages,
            'successful_pages': self.successful_pages,
            'failed_pages': self.failed_pages,
            'successful_pages_per_second': self.successful_pages / self.elapsed_seconds,
            'failed_pages_per_second': self.failed_pages / self.elapsed_seconds,
            'total_pages_per_second': self.total_pages / self.elapsed_seconds,
            'max_successful_pages_per_second': self.max_successful_pages_per_second,
            'max_failed_pages_per_second': self.max_failed_pages_per_second,
            'max_pages_per_second': self.max_pages_per_second,
            'elapsed_seconds': self.elapsed_seconds
        }

def main():
    redis_manager = RedisManager()
    stats_by_run: Dict[str, WriterStats] = {}
    for data in redis_manager.read_from_writer_queue():
        if data is None:
            print("No data found in the queue")
            continue
        try:
            data_dict = json.loads(data) if isinstance(data, str) else data
            payload = WritePayload.model_validate(data_dict)
            
            if payload.run_id not in stats_by_run:
                stats_by_run[payload.run_id] = WriterStats(payload.run_id)
            
            stats = stats_by_run[payload.run_id]
            stats.add_task(payload)
            
            stats_path = Path(payload.log_dir) / "stats.log"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            
            logs_path = Path(payload.log_dir) / "stats.jsonl"
            logs_path.parent.mkdir(parents=True, exist_ok=True)
            
            start_time = min(datetime.fromisoformat(task.start_time) for task in stats.tasks)
            end_time = max(datetime.fromisoformat(task.end_time) for task in stats.tasks)
            elapsed_seconds = (end_time - start_time).total_seconds()
            
            rates = stats.calculate_rates(elapsed_seconds)
            
            log_entry = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                # Task counts
                f"max total pps: {rates['max_pages_per_second']:.2f} p/s, "
                f"total: {len(stats.tasks)} "
                f"succeeded: {stats.successful_tasks}, "
                f"failed: {stats.failed_tasks} "
                # Page counts
                f"total pages: {stats.total_pages}, "
                f"successful pages: {stats.successful_pages}, "
                f"failed pages: {stats.failed_pages} "
                # Current rates
                f"total pps: {rates['pages_per_second']:.2f} p/s, "
                f"successful pps: {rates['successful_pages_per_second']:.2f} p/s, "
                f"failed pps: {rates['failed_pages_per_second']:.2f} p/s "
                # Peak rates
                f"max successful pps: {rates['max_successful_pages_per_second']:.2f} p/s, "
                f"max failed pps: {rates['max_failed_pages_per_second']:.2f} p/s "
                f"time elapsed: {elapsed_seconds:.1f}s"
            )
            with open(stats_path, 'a') as f:
                f.write(log_entry + '\n')
                
            stats_entry = {
                'timestamp': datetime.now().isoformat(),
                **stats.to_dict() 
            }
            with open(logs_path, 'a') as f:
                f.write(json.dumps(stats_entry) + '\n')
                
        except Exception as e:
            print(f"Error processing write payload: {e}")
            continue

if __name__ == "__main__":
    main()
