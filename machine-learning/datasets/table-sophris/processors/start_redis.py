#!/usr/bin/env python
import os
import sys
import time
import signal
import logging
import subprocess
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('redis_starter')

def run_redis_server(port=6379, data_dir="./redis-data"):
    redis_server_path = shutil.which("redis-server")
    if not redis_server_path:
        logger.error("Redis server not found. Please install Redis.")
        sys.exit(1)
    
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    pid_file = data_dir / "redis.pid"
    log_file = data_dir / "redis.log"
    
    proc = None
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                if proc.poll() is None:
                    proc.kill()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    redis_cmd = [
        redis_server_path,
        "--port", str(port),
        "--dir", str(data_dir),
        "--pidfile", str(pid_file),
        "--daemonize", "no",
        "--loglevel", "notice"
    ]
    
    logger.info(f"Starting Redis server on port {port}...")
    
    proc = subprocess.Popen(
        redis_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    
    logger.info(f"Redis server running (PID: {proc.pid}). Press Ctrl+C to stop.")
    
    try:
        proc.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted. Shutting down Redis...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Redis didn't terminate gracefully, forcing...")
            proc.kill()
    finally:
        exit_code = proc.poll()
        logger.info(f"Redis server exited with code {exit_code}")

if __name__ == "__main__":
    port = 6379
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    run_redis_server(port)