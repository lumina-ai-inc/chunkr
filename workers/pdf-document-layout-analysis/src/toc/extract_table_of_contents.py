import tempfile
import uuid
from os.path import join
from pathlib import Path
from typing import AnyStr
from fast_trainer.PdfSegment import PdfSegment
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from toc.TOCExtractor import TOCExtractor
from configuration import service_logger
from toc.PdfSegmentation import PdfSegmentation

TITLE_TYPES = {TokenType.TITLE, TokenType.SECTION_HEADER}
SKIP_TYPES = {TokenType.TITLE, TokenType.SECTION_HEADER, TokenType.PAGE_HEADER, TokenType.PICTURE}


def get_file_path(file_name, extension):
    return join(tempfile.gettempdir(), file_name + "." + extension)


def pdf_content_to_pdf_path(file_content):
    file_id = str(uuid.uuid1())

    pdf_path = Path(get_file_path(file_id, "pdf"))
    pdf_path.write_bytes(file_content)

    return pdf_path


def skip_name_of_the_document(pdf_segments: list[PdfSegment], title_segments: list[PdfSegment]):
    segments_to_remove = []
    last_segment = None
    for segment in pdf_segments:
        if segment.segment_type not in SKIP_TYPES:
            break
        if segment.segment_type == TokenType.PAGE_HEADER or segment.segment_type == TokenType.PICTURE:
            continue
        if not last_segment:
            last_segment = segment
        else:
            if segment.bounding_box.right < last_segment.bounding_box.left + last_segment.bounding_box.width * 0.66:
                break
            last_segment = segment
        if segment.segment_type in TITLE_TYPES:
            segments_to_remove.append(segment)
    for segment in segments_to_remove:
        title_segments.remove(segment)


def get_pdf_segments_from_segment_boxes(pdf_features: PdfFeatures, segment_boxes: list[dict]) -> list[PdfSegment]:
    pdf_segments: list[PdfSegment] = []
    for segment_box in segment_boxes:
        left, top, width, height = segment_box["left"], segment_box["top"], segment_box["width"], segment_box["height"]
        bounding_box = Rectangle.from_width_height(left, top, width, height)
        segment_type = TokenType.from_value(segment_box["type"])
        pdf_name = pdf_features.file_name
        segment = PdfSegment(segment_box["page_number"], bounding_box, segment_box["text"], segment_type, pdf_name)
        pdf_segments.append(segment)
    return pdf_segments


def extract_table_of_contents(file: AnyStr, segment_boxes: list[dict], skip_document_name=False):
    service_logger.info("Getting TOC")
    pdf_path = pdf_content_to_pdf_path(file)
    pdf_features: PdfFeatures = PdfFeatures.from_pdf_path(pdf_path)
    pdf_segments: list[PdfSegment] = get_pdf_segments_from_segment_boxes(pdf_features, segment_boxes)
    title_segments = [segment for segment in pdf_segments if segment.segment_type in TITLE_TYPES]
    if skip_document_name:
        skip_name_of_the_document(pdf_segments, title_segments)
    pdf_segmentation: PdfSegmentation = PdfSegmentation(pdf_features, title_segments)
    toc_instance: TOCExtractor = TOCExtractor(pdf_segmentation)
    return toc_instance.to_dict()
