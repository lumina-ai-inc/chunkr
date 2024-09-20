from src.models.segment_model import Segment

def adjust_segments(segments: list[Segment], offset: float = 5.0):
    for segment in segments:
        segment.width = segment.width + offset * 2
        segment.height = segment.height + offset * 2
        segment.left = segment.left - offset
        segment.top = segment.top - offset