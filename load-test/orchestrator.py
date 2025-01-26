import argparse
from datetime import datetime
from pathlib import Path
from redis_utils import RedisManager
from models import ProcessPayload

def main(input_dir: str, max_files: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs").resolve() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / "stats.log"
    
    redis_manager = RedisManager()

    input_path = Path(input_dir)
    files = list(input_path.resolve().glob("*"))[:max_files] if max_files else list(input_path.resolve().glob("*"))

    for file_path in files:
        payload = ProcessPayload(
            input_file=str(file_path.resolve()),
            output_dir=str(run_dir.resolve()),
            start_time=datetime.now().isoformat(),
            stats_path=str(stats_path.resolve()),
        ).model_dump_json()
        
        redis_manager.add_to_processing_queue(payload)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files using Chunkr')
    parser.add_argument('--input', default="./input",
                      help='Input folder path (default: input)')
    parser.add_argument('--max-files', type=int, default=None,
                      help='Maximum number of files to process (default: all files)')
    args = parser.parse_args()
    main(args.input, args.max_files)
    
    