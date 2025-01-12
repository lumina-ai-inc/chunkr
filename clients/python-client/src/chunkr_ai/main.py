from chunkr_ai.api.chunkr import Chunkr
from chunkr_ai.models import Configuration
from chunkr_ai.api.config import SegmentationStrategy, ChunkProcessing

if __name__ == "__main__":
    chunkr = Chunkr()
    task = chunkr.update_task("556b4fe5-e3f7-48dc-9f56-0fb7fbacdb87", Configuration(
        chunk_processing=ChunkProcessing(
            target_length=1000
        )
    ))
    print(task)