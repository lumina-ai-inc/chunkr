from src.models.segment_model import Segment

def adjust_segments(segments: list[Segment], offset: float = 5.0, density: int = 300, pdla_density: int = 72):
    scale_factor = density / pdla_density
    for segment in segments:
        # Scale dimensions and positions
        segment.width *= scale_factor
        segment.height *= scale_factor
        segment.left *= scale_factor
        segment.top *= scale_factor
        
        # Apply offset
        segment.width += offset * 2
        segment.height += offset * 2
        segment.left -= offset
        segment.top -= offset