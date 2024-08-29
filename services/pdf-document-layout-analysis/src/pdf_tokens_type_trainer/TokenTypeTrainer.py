from pathlib import Path

import numpy as np
from tqdm import tqdm

from pdf_features.PdfToken import PdfToken
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.PdfTrainer import PdfTrainer
from pdf_tokens_type_trainer.TokenFeatures import TokenFeatures


class TokenTypeTrainer(PdfTrainer):
    def get_model_input(self) -> np.ndarray:
        features_rows = []

        contex_size = self.model_configuration.context_size
        for token_features, page in self.loop_token_features():
            page_tokens = [
                self.get_padding_token(segment_number=i - 999999, page_number=page.page_number) for i in range(contex_size)
            ]
            page_tokens += page.tokens
            page_tokens += [
                self.get_padding_token(segment_number=999999 + i, page_number=page.page_number) for i in range(contex_size)
            ]

            tokens_indexes = range(contex_size, len(page_tokens) - contex_size)
            page_features = [self.get_context_features(token_features, page_tokens, i) for i in tokens_indexes]
            features_rows.extend(page_features)

        return self.features_rows_to_x(features_rows)

    def loop_token_features(self):
        for pdf_features in tqdm(self.pdfs_features):
            token_features = TokenFeatures(pdf_features)

            for page in pdf_features.pages:
                if not page.tokens:
                    continue

                yield token_features, page

    def get_context_features(self, token_features: TokenFeatures, page_tokens: list[PdfToken], token_index: int):
        token_row_features = []
        first_token_from_context = token_index - self.model_configuration.context_size
        for i in range(self.model_configuration.context_size * 2):
            first_token = page_tokens[first_token_from_context + i]
            second_token = page_tokens[first_token_from_context + i + 1]
            token_row_features.extend(token_features.get_features(first_token, second_token, page_tokens))

        return token_row_features

    def predict(self, model_path: str | Path = None):
        predictions = super().predict(model_path)
        predictions_assigned = 0
        for token_features, page in self.loop_token_features():
            for token, prediction in zip(
                page.tokens, predictions[predictions_assigned : predictions_assigned + len(page.tokens)]
            ):
                token.prediction = int(np.argmax(prediction))

            predictions_assigned += len(page.tokens)

    def set_token_types(self, model_path: str | Path = None):
        self.predict(model_path)
        for token in self.loop_tokens():
            token.token_type = TokenType.from_index(token.prediction)
