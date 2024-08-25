from pathlib import Path

from fast_trainer.Paragraph import Paragraph
from fast_trainer.PdfSegment import PdfSegment
from pdf_features.PdfToken import PdfToken
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer


class ParagraphExtractorTrainer(TokenTypeTrainer):
    def get_context_features(self, token_features: TokenFeatures, page_tokens: list[PdfToken], token_index: int):
        token_row_features = list()
        first_token_from_context = token_index - self.model_configuration.context_size
        for i in range(self.model_configuration.context_size * 2):
            first_token = page_tokens[first_token_from_context + i]
            second_token = page_tokens[first_token_from_context + i + 1]
            features = token_features.get_features(first_token, second_token, page_tokens)
            features += self.get_paragraph_extraction_features(first_token, second_token)
            token_row_features.extend(features)

        return token_row_features

    @staticmethod
    def get_paragraph_extraction_features(first_token: PdfToken, second_token: PdfToken) -> list[int]:
        one_hot_token_type_1 = [1 if token_type == first_token.token_type else 0 for token_type in TokenType]
        one_hot_token_type_2 = [1 if token_type == second_token.token_type else 0 for token_type in TokenType]
        return one_hot_token_type_1 + one_hot_token_type_2

    def loop_token_next_token(self):
        for pdf_features in self.pdfs_features:
            for page in pdf_features.pages:
                if not page.tokens:
                    continue
                if len(page.tokens) == 1:
                    yield page, page.tokens[0], page.tokens[0]
                for token, next_token in zip(page.tokens, page.tokens[1:]):
                    yield page, token, next_token

    def get_pdf_segments(self, paragraph_extractor_model_path: str | Path) -> list[PdfSegment]:
        paragraphs = self.get_paragraphs(paragraph_extractor_model_path)
        pdf_segments = [PdfSegment.from_pdf_tokens(paragraph.tokens, paragraph.pdf_name) for paragraph in paragraphs]

        return pdf_segments

    def get_paragraphs(self, paragraph_extractor_model_path) -> list[Paragraph]:
        self.predict(paragraph_extractor_model_path)
        paragraphs: list[Paragraph] = []
        last_page = None
        for page, token, next_token in self.loop_token_next_token():
            if last_page != page:
                last_page = page
                paragraphs.append(Paragraph([token], page.pdf_name))
            if token == next_token:
                continue
            if token.prediction:
                paragraphs[-1].add_token(next_token)
                continue
            paragraphs.append(Paragraph([next_token], page.pdf_name))

        return paragraphs
