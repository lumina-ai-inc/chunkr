import redis
import json
from typing import Dict, Any, Generator
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisManager:
    def __init__(
        self, 
        host: str = 'localhost', 
        port: int = 6379, 
        queue_name: str = None,
        writer_queue: str = None
    ):
        # Load environment variables from .env file
        load_dotenv()
        
        # Get queue names from env vars with fallbacks and parameter override
        self.queue_name = queue_name or os.getenv('PROCESSING_QUEUE', 'processing_queue')
        self.writer_queue = writer_queue or os.getenv('WRITER_QUEUE', 'writer_queue')
        
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True
        )

    def add_to_processing_queue(self, data: Dict[Any, Any]) -> None:
        """
        Add item to the processing queue
        """
        try:
            self.redis_client.lpush(self.queue_name, json.dumps(data))
            logger.info("Added message to processing queue")
        except Exception as e:
            logger.error(f"Error adding to queue: {e}")
            raise

    def read_from_processing_queue(self, timeout: int = 0) -> Generator[Dict[str, Any], None, None]:
        """
        Read from the processing queue (blocking operation)
        """
        while True:
            try:
                # Read new messages with blocking operation
                result = self.redis_client.brpop(self.queue_name, timeout)
                if result:
                    data = json.loads(result[1])
                    yield data
                else:
                    # If timeout occurred, yield None
                    yield None
            except Exception as e:
                logger.error(f"Error reading from queue: {e}")
                continue

    def add_to_writer_queue(self, data: Dict[Any, Any]) -> None:
        """
        Add processed data to the writer queue
        """
        try:
            self.redis_client.lpush(self.writer_queue, json.dumps(data))
            logger.info("Added data to writer queue")
        except Exception as e:
            logger.error(f"Error adding to writer queue: {e}")
            raise

    def read_from_writer_queue(self, timeout: int = 0) -> Generator[Dict[str, Any], None, None]:
        """
        Read from the writer queue (blocking operation)
        """
        while True:
            try:
                result = self.redis_client.brpop(self.writer_queue, timeout)
                if result:
                    data = json.loads(result[1])
                    yield data
                else:
                    yield None
            except Exception as e:
                logger.error(f"Error reading from writer queue: {e}")
                continue 