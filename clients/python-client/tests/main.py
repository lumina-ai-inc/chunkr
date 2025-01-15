from chunkr_ai import Chunkr
from chunkr_ai.models import Configuration, SegmentationStrategy, OcrStrategy, SegmentProcessing, GenerationConfig, GenerationStrategy

def main():
    chunkr = Chunkr()
    task_id = "702e910e-c01a-4287-84b4-2b759d3c943e"
    new_config = Configuration(
        segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS,
        segment_processing=SegmentProcessing(
            picture=GenerationConfig(
                html=GenerationStrategy.LLM,
                markdown=GenerationStrategy.LLM,
                llm="Describe the picture in detail"
            )
        )
    )
    # new_config = Configuration(
    #     segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS,
    # )
    try:
        task = chunkr.update(task_id, new_config)
        print(task)
    except Exception as e:
        print(e)
if __name__ == "__main__":
    main()

