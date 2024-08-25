from configuration import service_logger
from pdf_token_type_labels.TokenType import TokenType


def extract_text(segment_boxes: list[dict], types: list[TokenType]):
    service_logger.info(f"Extracted types: {[t.name for t in types]}")
    text = "\n".join(
        [
            segment_box["text"]
            for segment_box in segment_boxes
            if TokenType.from_text(segment_box["type"].replace(" ", "_")) in types
        ]
    )
    return text
