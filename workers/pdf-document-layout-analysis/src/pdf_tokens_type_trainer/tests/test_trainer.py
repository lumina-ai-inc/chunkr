from os.path import join, exists
from unittest import TestCase

from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer

from pdf_features.PdfFeatures import PdfFeatures

from configuration import ROOT_PATH


class TestTrainer(TestCase):
    def test_train_blank_pdf(self):
        pdf_features = PdfFeatures.from_pdf_path(join(ROOT_PATH, "test_pdfs", "blank.pdf"))
        model_path = join(ROOT_PATH, "model", "blank.model")
        trainer = TokenTypeTrainer([pdf_features])
        trainer.train(model_path, [])
        self.assertFalse(exists(model_path))

    def test_predict_blank_pdf(self):
        pdf_features = PdfFeatures.from_pdf_path(join(ROOT_PATH, "test_pdfs", "blank.pdf"))
        trainer = TokenTypeTrainer([pdf_features])
        trainer.set_token_types()
        self.assertEqual([], pdf_features.pages[0].tokens)

    def test_predict(self):
        pdf_features = PdfFeatures.from_pdf_path(join(ROOT_PATH, "test_pdfs", "test.pdf"))
        trainer = TokenTypeTrainer([pdf_features])
        trainer.set_token_types()
        tokens = pdf_features.pages[0].tokens
        self.assertEqual(TokenType.TITLE, tokens[0].token_type)
        self.assertEqual("Document Big Centered Title", tokens[0].content)
        self.assertEqual(TokenType.TEXT, tokens[1].token_type)
        self.assertEqual("List Title", tokens[10].content)
