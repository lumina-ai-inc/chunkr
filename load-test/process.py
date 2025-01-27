from datetime import datetime
import asyncio
from typing import Optional
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.ERROR)

from chunkr_ai import Chunkr
from chunkr_ai.models import TaskResponse
from redis_utils import RedisManager
from models import ProcessPayload, WritePayload

async def process_file(chunkr: Chunkr, file_path: str) -> Optional[TaskResponse]:
    try:    
        task = await chunkr.upload(file_path)
        return task
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

async def main():
    chunkr = Chunkr()
    redis_manager = RedisManager()
    for data in redis_manager.read_from_processing_queue():
        if data is None:
            await asyncio.sleep(0.1)
            continue
        try:
            data_dict = json.loads(data) if isinstance(data, str) else data
            payload = ProcessPayload.model_validate(data_dict)
            result = await process_file(chunkr, payload.input_file)
            if result:
                output_file = Path(payload.output_dir) / f"output/{result.task_id}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(result.model_dump(mode='json'), f, indent=2)
                write_payload = WritePayload(
                    run_id=payload.run_id,
                    log_dir=payload.log_dir,
                    task_path=str(output_file),
                    start_time=payload.start_time,
                    end_time=str(datetime.now().isoformat())
                )
                redis_manager.add_to_writer_queue(write_payload.model_dump())
            else:
                print(f"Error processing file {payload.input_file}")
        except Exception as e:
            print(f"Error validating payload: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(main())