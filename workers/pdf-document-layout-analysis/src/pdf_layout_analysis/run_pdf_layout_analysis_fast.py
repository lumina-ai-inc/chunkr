import os
from os.path import join
from pathlib import Path
from typing import AnyStr

from fast_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from fast_trainer.model_configuration import MODEL_CONFIGURATION as PARAGRAPH_EXTRACTION_CONFIGURATION
from pdf_features.PdfFeatures import PdfFeatures
from pdf_layout_analysis.run_pdf_layout_analysis import pdf_content_to_pdf_path
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration

from configuration import ROOT_PATH, service_logger, XMLS_PATH
from data_model.SegmentBox import SegmentBox


def analyze_pdf_fast(file: AnyStr, xml_file_name: str = "") -> list[dict]:
    pdf_path = pdf_content_to_pdf_path(file)
    service_logger.info("Creating Paragraph Tokens [fast]")

    xml_path = Path(join(XMLS_PATH, xml_file_name)) if xml_file_name else None

    if xml_path and not xml_path.parent.exists():
        os.makedirs(xml_path.parent, exist_ok=True)

    pdf_features = PdfFeatures.from_pdf_path(pdf_path, str(xml_path) if xml_path else None)
    token_type_trainer = TokenTypeTrainer([pdf_features], ModelConfiguration())
    token_type_trainer.set_token_types(join(ROOT_PATH, "models", "token_type_lightgbm.model"))
    trainer = ParagraphExtractorTrainer(pdfs_features=[pdf_features], model_configuration=PARAGRAPH_EXTRACTION_CONFIGURATION)
    segments = trainer.get_pdf_segments(join(ROOT_PATH, "models", "paragraph_extraction_lightgbm.model"))
    return [SegmentBox.from_pdf_segment(pdf_segment, pdf_features.pages).to_dict() for pdf_segment in segments]
