from fast_trainer.PdfSegment import PdfSegment
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken


class PdfSegmentation:
    def __init__(self, pdf_features: PdfFeatures, pdf_segments: list[PdfSegment]):
        self.pdf_features: PdfFeatures = pdf_features
        self.pdf_segments: list[PdfSegment] = pdf_segments
        self.tokens_by_segments: dict[PdfSegment, list[PdfToken]] = self.find_tokens_by_segments()

    @staticmethod
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

    def find_tokens_by_segments(self):
        tokens_by_segments: dict[PdfSegment, list[PdfToken]] = {}
        for page in self.pdf_features.pages:
            page_segments = [segment for segment in self.pdf_segments if segment.page_number == page.page_number]
            for token in page.tokens:
                self.find_segment_for_token(token, page_segments, tokens_by_segments)
        return tokens_by_segments
