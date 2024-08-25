import os
from os.path import exists, join
from pathlib import Path

import lightgbm as lgb
import numpy as np

from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfFont import PdfFont
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.download_models import pdf_tokens_type_model


class PdfTrainer:
    def __init__(self, pdfs_features: list[PdfFeatures], model_configuration: ModelConfiguration = None):
        self.pdfs_features = pdfs_features
        self.model_configuration = model_configuration if model_configuration else ModelConfiguration()

    def get_model_input(self) -> np.ndarray:
        pass

    @staticmethod
    def features_rows_to_x(features_rows):
        if not features_rows:
            return np.zeros((0, 0))

        x = np.zeros(((len(features_rows)), len(features_rows[0])))
        for i, v in enumerate(features_rows):
            x[i] = v
        return x

    def train(self, model_path: str | Path, labels: list[int]):
        print(f"Getting model input")
        x_train = self.get_model_input()

        if not x_train.any():
            print("No data for training")
            return

        lgb_train = lgb.Dataset(x_train, labels)
        lgb_eval = lgb.Dataset(x_train, labels, reference=lgb_train)
        print(f"Training")

        if self.model_configuration.resume_training and exists(model_path):
            model = lgb.Booster(model_file=model_path)
            gbm = model.refit(x_train, labels)
        else:
            gbm = lgb.train(params=self.model_configuration.dict(), train_set=lgb_train, valid_sets=[lgb_eval])

        print(f"Saving")
        gbm.save_model(model_path, num_iteration=gbm.best_iteration)

    def loop_tokens(self):
        for pdf_features in self.pdfs_features:
            for page, token in pdf_features.loop_tokens():
                yield token

    @staticmethod
    def get_padding_token(segment_number: int, page_number: int):
        return PdfToken(
            page_number,
            "pad_token",
            "",
            PdfFont("pad_font_id", False, False, 0.0, "#000000"),
            segment_number,
            Rectangle(0, 0, 0, 0),
            TokenType.TEXT,
        )

    def predict(self, model_path: str | Path = None):
        model_path = model_path if model_path else pdf_tokens_type_model
        x = self.get_model_input()

        if not x.any():
            return self.pdfs_features

        lightgbm_model = lgb.Booster(model_file=model_path)
        return lightgbm_model.predict(x)

    def save_training_data(self, save_folder_path: str | Path, labels: list[int]):
        os.makedirs(save_folder_path, exist_ok=True)

        x = self.get_model_input()

        np.save(join(str(save_folder_path), "x.npy"), x)
        np.save(join(str(save_folder_path), "y.npy"), labels)
