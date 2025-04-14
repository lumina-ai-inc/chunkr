import os
import json
import time
import redis
from pathlib import Path
import logging
import argparse
from dotenv import load_dotenv
import subprocess
import shutil
import signal
import sys
import socket
import atexit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pdf_queue')

class RedisManager:
    
    def __init__(self, data_dir="./redis-data", port=6379):
        self.redis_process = None
        self.port = port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.pid_file = self.data_dir / "redis.pid"
        self.log_file = self.data_dir / "redis.log"
    
    def is_redis_running(self):
        try:
            client = redis.Redis(host='localhost', port=self.port, db=0, socket_timeout=1)
            return client.ping()
        except (redis.exceptions.ConnectionError, redis.exceptions.ResponseError):
            return False
    
    def start_redis(self):
        if self.is_redis_running():
            logger.info(f"Redis already running on port {self.port}")
            return True
        
        redis_server_path = shutil.which("redis-server")
        if not redis_server_path:
            logger.error("Redis server not found. Please install Redis.")
            return False
        
        try:
            logger.info(f"Starting Redis server on port {self.port}...")
            
            redis_cmd = [
                redis_server_path,
                "--port", str(self.port),
                "--dir", str(self.data_dir),
                "--pidfile", str(self.pid_file),
                "--daemonize", "no",
                "--loglevel", "notice",
                "--save", "",
                "--appendonly", "no"
            ]
            
            self.redis_process = subprocess.Popen(
                redis_cmd,
                stdout=open(self.log_file, 'a'),
                stderr=subprocess.STDOUT,
            )
            
            start_time = timew.time()
            while not self.is_redis_running() and time.time() - start_time < 5:
                time.sleep(0.1)
                if self.redis_process.poll() is not None:
                    logger.error(f"Redis process exited prematurely with code {self.redis_process.returncode}")
                    with open(self.log_file, 'r') as f:
                        logger.error(f"Redis log: {f.read()}")
                    return False
            
            if self.is_redis_running():
                logger.info(f"Redis server started successfully on port {self.port}")
                return True
            else:
                logger.error("Failed to start Redis server")
                return False
                
        except Exception as e:
            logger.error(f"Error starting Redis server: {str(e)}")
            return False
    
    def stop_redis(self):
        if self.redis_process:
            logger.info("Stopping Redis server...")
            try:
                try:
                    client = redis.Redis(host='localhost', port=self.port, db=0, socket_timeout=1)
                    client.flushall()
                    logger.info("Flushed all Redis data")
                except Exception as e:
                    logger.warning(f"Could not flush Redis data: {str(e)}")
                
                self.redis_process.terminate()
                self.redis_process.wait(timeout=5)
                logger.info("Redis server stopped")
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self.redis_process.kill()
                    logger.info("Redis server forcefully terminated")
                except OSError as e:
                    logger.error(f"Error killing Redis process: {str(e)}")
            
            self.redis_process = None
            
            try:
                for file in self.data_dir.glob("*.rdb"):
                    file.unlink()
                for file in self.data_dir.glob("*.aof"):
                    file.unlink()
                logger.info("Cleaned up Redis data files")
            except Exception as e:
                logger.warning(f"Error cleaning up Redis data files: {str(e)}")
            
            return True
        
        elif self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                
                try:
                    client = redis.Redis(host='localhost', port=self.port, db=0, socket_timeout=1)
                    client.flushall()
                    logger.info("Flushed all Redis data")
                except Exception as e:
                    logger.warning(f"Could not flush Redis data: {str(e)}")
                    
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to Redis process with PID {pid}")
                
                try:
                    for file in self.data_dir.glob("*.rdb"):
                        file.unlink()
                    for file in self.data_dir.glob("*.aof"):
                        file.unlink()
                    logger.info("Cleaned up Redis data files")
                except Exception as e:
                    logger.warning(f"Error cleaning up Redis data files: {str(e)}")
                    
                return True
            except (ValueError, ProcessLookupError, PermissionError) as e:
                logger.error(f"Failed to stop Redis using PID file: {str(e)}")
                return False
        
        return False

class PDFQueue:
    
    def __init__(self, redis_url=None):
        load_dotenv(override=True)
        
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.redis = redis.from_url(self.redis_url)
        
        self.task_queue = "pdf:processing:tasks"
        self.result_queue = "pdf:processing:results"
        self.processing_queue = "pdf:processing:in_progress"
        
        logger.info(f"Initialized PDF Queue with Redis at {self.redis_url}")
    
    def add_task(self, pdf_key, api_key, s3_bucket, dataset_name, task_id=None):
        if task_id is None:
            task_id = f"task_{int(time.time())}_{Path(pdf_key).stem}"
        
        task = {
            "task_id": task_id,
            "pdf_key": pdf_key,
            "api_key": api_key,
            "s3_bucket": s3_bucket,
            "dataset_name": dataset_name,
            "queued_at": time.time()
        }
        
        self.redis.lpush(self.task_queue, json.dumps(task))
        logger.info(f"Added task {task_id} to queue for PDF: {pdf_key}")
        
        return task_id
    
    def get_task(self, timeout=0):
        if not self.ensure_connection():
            logger.error("Cannot get task: Redis connection failed")
            return None
        
        try:
            result = self.redis.brpoplpush(self.task_queue, self.processing_queue, timeout)
            
            if result:
                try:
                    task = json.loads(result)
                    logger.info(f"Retrieved task {task['task_id']} for processing")
                    return task
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding task JSON: {str(e)}")
                    self.redis.lpush("pdf:processing:errors", result)
                    self.redis.lrem(self.processing_queue, 1, result)
                    return None
            
            return None
            
        except redis.exceptions.ConnectionError:
            logger.error("Lost connection to Redis while getting task")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving task: {str(e)}")
            return None
    
    def complete_task(self, task_id, success, skipped=False, error=None):
        for i, task_data in enumerate(self.redis.lrange(self.processing_queue, 0, -1)):
            task = json.loads(task_data)
            if task["task_id"] == task_id:
                result = {
                    "task_id": task_id,
                    "success": success,
                    "skipped": skipped,
                    "error": error,
                    "pdf_key": task["pdf_key"],
                    "completed_at": time.time()
                }
                
                self.redis.lpush(self.result_queue, json.dumps(result))
                
                self.redis.lrem(self.processing_queue, 1, task_data)
                
                logger.info(f"Completed task {task_id} with {'success' if success else 'failure'}")
                return True
        
        logger.warning(f"Task {task_id} not found in processing queue")
        return False
    
    def get_stats(self):
        return {
            "queued": self.redis.llen(self.task_queue),
            "processing": self.redis.llen(self.processing_queue),
            "completed": self.redis.llen(self.result_queue)
        }
    
    def clear_all_queues(self):
        self.redis.delete(self.task_queue)
        self.redis.delete(self.processing_queue)
        self.redis.delete(self.result_queue)
        logger.info("Cleared all PDF processing queues")
        return True
    
    def ensure_connection(self, max_retries=3):
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.redis.ping()
                if retry_count > 0:
                    logger.info("Successfully reconnected to Redis")
                return True
            except redis.exceptions.ConnectionError:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to connect to Redis after {max_retries} attempts")
                    return False
                
                wait_time = 2 ** retry_count
                logger.warning(f"Redis connection failed. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                time.sleep(wait_time)
                
                try:
                    self.redis = redis.from_url(self.redis_url)
                except Exception as e:
                    logger.error(f"Error creating Redis connection: {str(e)}")
        
        return False


def add_pdfs_to_queue(pdf_keys, api_key, s3_bucket, dataset_name, limit=None):
    queue = PDFQueue()
    
    if limit:
        pdf_keys = pdf_keys[:limit]
    
    count = 0
    for pdf_key in pdf_keys:
        queue.add_task(pdf_key, api_key, s3_bucket, dataset_name)
        count += 1
    
    logger.info(f"Added {count} PDFs to processing queue")
    return count


def display_stats_continuously(refresh_interval=5):
    queue = PDFQueue()
    
    try:
        while True:
            print("\033c", end="")
            
            stats = queue.get_stats()
            print("\n--- PDF Queue Statistics ---")
            print(f"Tasks queued: {stats['queued']}")
            print(f"Tasks processing: {stats['processing']}")
            print(f"Tasks completed: {stats['completed']}")
            print(f"\nTotal tasks: {stats['queued'] + stats['processing'] + stats['completed']}")
            print(f"\nPress Ctrl+C to exit. Refreshing every {refresh_interval} seconds...")
            
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\nExiting stats monitor.")

def main():
    parser = argparse.ArgumentParser(description='PDF Processing Queue Management')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    redis_parser = subparsers.add_parser('redis', help='Manage Redis server')
    redis_parser.add_argument('action', choices=['start', 'stop', 'status'], 
                        help='Action to perform on Redis server')
    redis_parser.add_argument('--port', type=int, default=6379,
                        help='Redis server port (default: 6379)')
    redis_parser.add_argument('--foreground', action='store_true',
                        help='Keep Redis running in the foreground')
    
    stats_parser = subparsers.add_parser('stats', help='Show queue statistics')
    stats_parser.add_argument('--refresh', type=int, default=5,
                      help='Refresh interval in seconds (default: 5)')
    stats_parser.add_argument('--once', action='store_true',
                      help='Show stats once and exit')
    
    clear_parser = subparsers.add_parser('clear', help='Clear all queues')
    
    queue_parser = subparsers.add_parser('queue', help='Queue PDFs for processing')
    queue_parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of PDFs to queue')
    queue_parser.add_argument('--bucket', type=str, default="chunkr-datasets",
                        help='S3 bucket name (defaults to environment variable S3_BUCKET)')
    queue_parser.add_argument('--dataset', type=str, default="tables-vlm-azure-distill-v1",
                        help='Dataset name/path in the bucket (defaults to environment variable DATASET_NAME)')
    queue_parser.add_argument('--count', type=int, required=True,
                        help='Number of PDFs to queue (required)')
    
    reset_parser = subparsers.add_parser('reset', help='Reset Redis: stop server, clear all data, and restart')
    
    args = parser.parse_args()
    
    if args.command == 'redis':
        redis_manager = RedisManager(port=args.port)
        
        if args.action == 'start':
            if redis_manager.start_redis():
                print(f"Redis server started on port {args.port}")
                
                if hasattr(args, 'foreground') and args.foreground:
                    try:
                        print("Redis server running in foreground. Press Ctrl+C to stop...")
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\nStopping Redis server...")
                        redis_manager.stop_redis()
                        print("Redis server stopped")
            else:
                print("Failed to start Redis server")
                sys.exit(1)
        
        elif args.action == 'stop':
            if redis_manager.stop_redis():
                print("Redis server stopped")
            else:
                print("No Redis server to stop or failed to stop")
        
        elif args.action == 'status':
            if redis_manager.is_redis_running():
                print(f"Redis server is running on port {args.port}")
            else:
                print(f"Redis server is not running on port {args.port}")
        
        return
    
    redis_manager = RedisManager()
    if not redis_manager.is_redis_running():
        start_choice = input("Redis is not running. Start it now? [y/N]: ")
        if start_choice.lower() == 'y':
            if not redis_manager.start_redis():
                print("Failed to start Redis. Please start it manually first.")
                sys.exit(1)
        else:
            print("Redis must be running to use this command. Exiting.")
            sys.exit(1)
    
    queue = PDFQueue()
    
    if args.command == 'stats':
        if hasattr(args, 'once') and args.once:
            stats = queue.get_stats()
            print("\n--- PDF Queue Statistics ---")
            print(f"Tasks queued: {stats['queued']}")
            print(f"Tasks processing: {stats['processing']}")
            print(f"Tasks completed: {stats['completed']}")
        else:
            refresh_interval = 5
            if hasattr(args, 'refresh'):
                refresh_interval = args.refresh
            display_stats_continuously(refresh_interval)
    
    elif args.command == 'clear':
        queue.clear_all_queues()
        print("All queues have been cleared")
    
    elif args.command == 'queue':
        print("--- PDF Queue: Queuing PDFs ---")
        print(f"Dataset: {args.dataset}")
        print(f"S3 bucket: {args.bucket}")
        print(f"Number of PDFs to queue: {args.count}")
        
        load_dotenv(override=True)
        api_key = os.environ.get("CHUNKR_API_KEY")
        if not api_key:
            print("ERROR: No Chunkr API key found. Set CHUNKR_API_KEY environment variable.")
            return
        
        try:
            from storage import TableS3Storage
            s3_storage = TableS3Storage(args.bucket, args.dataset)
            
            base_prefix = f"{s3_storage.base_prefix}{s3_storage.dataset_name}/"
            raw_pdfs_prefix = f"{base_prefix}raw-pdfs/"
            
            pdfs = []
            try:
                paginator = s3_storage.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=s3_storage.s3_bucket, Prefix=raw_pdfs_prefix):
                    for obj in page.get('Contents', []):
                        if obj['Key'].endswith('.pdf'):
                            pdfs.append(obj['Key'])
            except Exception as e:
                print(f"Error listing PDFs in S3: {str(e)}")
                return
            
            if args.count and args.count < len(pdfs):
                pdfs = pdfs[:args.count]
            else:
                print(f"Warning: Requested {args.count} PDFs, but only {len(pdfs)} available. Queueing all available PDFs.")
            
            count = add_pdfs_to_queue(pdfs, api_key, args.bucket, args.dataset, args.limit)
            
            print(f"Successfully queued {count} PDFs for processing")
            
        except ImportError:
            print("ERROR: Could not import TableS3Storage. Make sure storage.py is in the same directory.")
            return
    
    elif args.command == 'reset':
        redis_manager = RedisManager()
        print("Resetting Redis server and all data...")
        
        if redis_manager.is_redis_running():
            redis_manager.stop_redis()
        
        try:
            for file in redis_manager.data_dir.glob("*.rdb"):
                file.unlink()
            for file in redis_manager.data_dir.glob("*.aof"):
                file.unlink()
            print("Cleaned up Redis data files")
        except Exception as e:
            print(f"Error cleaning up Redis data files: {str(e)}")
        
        if redis_manager.start_redis():
            print("Redis server restarted with clean data")
        else:
            print("Failed to restart Redis server")
            sys.exit(1)
    
    else:
        parser.print_help()

def signal_handler(sig, frame):
    print("\nShutting down...")
    sys.exit(0)

if __name__ == "__main__":
    def cleanup_on_exit():
        logger.info("Cleaning up before exit...")
        redis_manager = RedisManager()
        if redis_manager.is_redis_running():
            redis_manager.stop_redis() 
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    atexit.register(cleanup_on_exit)
    
    main()

def start_redis_server():
    redis_manager = RedisManager()
    
    if redis_manager.is_redis_running():
        logger.info("Redis is already running")
        return True
    
    logger.info("Starting Redis server...")
    return redis_manager.start_redis()

def check_redis_running():
    redis_manager = RedisManager()
    return redis_manager.is_redis_running()

def cleanup_on_exit():
    logger.info("Cleaning up before exit...")
    redis_manager = RedisManager()
    if redis_manager.is_redis_running():
        redis_manager.stop_redis() 