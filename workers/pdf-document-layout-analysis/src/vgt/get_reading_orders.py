from fast_trainer.PdfSegment import PdfSegment
from pdf_features.PdfPage import PdfPage
from pdf_features.PdfToken import PdfToken
from pdf_token_type_labels.TokenType import TokenType

from data_model.PdfImages import PdfImages


def find_segment_for_token(token: PdfToken, segments: list[PdfSegment], tokens_by_segments):
    best_score: float = 0
    most_probable_segment: PdfSegment | None = None
    for segment in segments:
        intersection_percentage = token.bounding_box.get_intersection_percentage(segment.bounding_box)
        if intersection_percentage > best_score:
            best_score = intersection_percentage
            most_probable_segment = segment
            if best_score >= 99:
                break
    if most_probable_segment:
        tokens_by_segments.setdefault(most_probable_segment, list()).append(token)


def get_average_reading_order_for_segment(page: PdfPage, tokens_for_segment: list[PdfToken]):
    reading_order_sum: int = sum(page.tokens.index(token) for token in tokens_for_segment)
    return reading_order_sum / len(tokens_for_segment)


def get_distance_between_segments(segment1: PdfSegment, segment2: PdfSegment):
    center_1_x = (segment1.bounding_box.left + segment1.bounding_box.right) / 2
    center_1_y = (segment1.bounding_box.top + segment1.bounding_box.bottom) / 2
    center_2_x = (segment2.bounding_box.left + segment2.bounding_box.right) / 2
    center_2_y = (segment2.bounding_box.top + segment2.bounding_box.bottom) / 2
    return ((center_1_x - center_2_x) ** 2 + (center_1_y - center_2_y) ** 2) ** 0.5


def add_no_token_segments(segments, no_token_segments):
    if segments:
        for no_token_segment in no_token_segments:
            closest_segment = sorted(segments, key=lambda seg: get_distance_between_segments(no_token_segment, seg))[0]
            closest_index = segments.index(closest_segment)
            if closest_segment.bounding_box.top < no_token_segment.bounding_box.top:
                segments.insert(closest_index + 1, no_token_segment)
            else:
                segments.insert(closest_index, no_token_segment)
    else:
        for segment in sorted(no_token_segments, key=lambda r: (r.bounding_box.left, r.bounding_box.top)):
            segments.append(segment)


def filter_and_sort_segments(page, tokens_by_segments, types):
    filtered_segments = [seg for seg in tokens_by_segments.keys() if seg.segment_type in types]
    order = {seg: get_average_reading_order_for_segment(page, tokens_by_segments[seg]) for seg in filtered_segments}
    return sorted(filtered_segments, key=lambda seg: order[seg])


def get_ordered_segments_for_page(segments_for_page: list[PdfSegment], page: PdfPage):
    tokens_by_segments: dict[PdfSegment, list[PdfToken]] = {}
    for token in page.tokens:
        find_segment_for_token(token, segments_for_page, tokens_by_segments)

    page_number_segment: None | PdfSegment = None
    if tokens_by_segments:
        last_segment = max(tokens_by_segments.keys(), key=lambda seg: seg.bounding_box.top)
        if last_segment.text_content and len(last_segment.text_content) < 5:
            page_number_segment = last_segment
            del tokens_by_segments[last_segment]

    header_segments: list[PdfSegment] = filter_and_sort_segments(page, tokens_by_segments, {TokenType.PAGE_HEADER})
    paragraph_types = {t for t in TokenType if t.name not in {"PAGE_HEADER", "PAGE_FOOTER", "FOOTNOTE"}}
    paragraph_segments = filter_and_sort_segments(page, tokens_by_segments, paragraph_types)
    footer_segments = filter_and_sort_segments(page, tokens_by_segments, {TokenType.PAGE_FOOTER, TokenType.FOOTNOTE})
    if page_number_segment:
        footer_segments.append(page_number_segment)
    ordered_segments = header_segments + paragraph_segments + footer_segments
    no_token_segments = [segment for segment in segments_for_page if segment not in ordered_segments]
    add_no_token_segments(ordered_segments, no_token_segments)
    return ordered_segments


def get_reading_orders(pdf_images_list: list[PdfImages], predicted_segments: list[PdfSegment]):
    ordered_segments: list[PdfSegment] = []
    for pdf_images in pdf_images_list:
        pdf_name = pdf_images.pdf_features.file_name
        segments_for_file = [segment for segment in predicted_segments if segment.pdf_name == pdf_name]
        for page in pdf_images.pdf_features.pages:
            segments_for_page = [segment for segment in segments_for_file if segment.page_number == page.page_number]
            ordered_segments.extend(get_ordered_segments_for_page(segments_for_page, page))
    return ordered_segments
