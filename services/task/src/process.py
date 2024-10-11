import base64
import concurrent.futures
import os
import tempfile
from pathlib import Path
from psycopg2 import connect
from threading import Lock
from textractor.entities.document import Document

import pandas as pd
from src.configs.llm_config import LLM__BASE_URL
from src.configs.pgsql_config import PG__URL
from src.configs.task_config import TASK__OCR_MODEL, TASK__TABLE_OCR_MODEL
from src.converters import crop_image
from src.llm import process_llm, extract_html_from_response, table_to_html
from src.models.ocr_model import ProcessInfo, ProcessType
from src.models.segment_model import Segment, SegmentType
from src.ocr import ppocr, ppstructure_table
from src.s3 import upload_file_to_s3
from PIL import Image
from textractor import Textractor
from textractor.visualizers.entitylist import EntityList
from textractor.data.constants import TextractFeatures
from types import SimpleNamespace
from src.models.ocr_model import OCRResult, OCRResponse, BoundingBox
from src.configs.aws_config import TASK__AWS_ACCESS_KEY, TASK__AWS_SECRET_KEY, TASK__AWS_REGION
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def adjust_segments(segments: list[Segment], offset: float = 5.0, density: int = 300, pdla_density: int = 72):
    scale_factor = density / pdla_density
    for segment in segments:
        segment.bbox.width *= scale_factor
        segment.bbox.height *= scale_factor
        segment.bbox.left *= scale_factor
        segment.bbox.top *= scale_factor

        segment.page_height *= scale_factor
        segment.page_width *= scale_factor

        segment.bbox.width += offset * 2
        segment.bbox.height += offset * 2
        segment.bbox.left -= offset
        segment.bbox.top -= offset

def process_segment_ocr(
    segment: Segment,
    segment_temp_file: Path
):
    process_info = ProcessInfo(
        segment_id=segment.segment_id, process_type=ProcessType.OCR)

    if segment.segment_type == SegmentType.Table:
        if TASK__TABLE_OCR_MODEL == "LLM" and LLM__BASE_URL:
            def llm_task():
                detail, response = process_llm(
                    segment_temp_file, table_to_html)
                return detail, response, extract_html_from_response(response)

            def ocr_task():
                return ppocr(segment_temp_file)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                llm_future = executor.submit(llm_task)
                ocr_future = executor.submit(ocr_task)

                detail, response, html = llm_future.result()
                ocr_results = ocr_future.result()

            segment.html = html
            process_info.detail = detail
            process_info.input_tokens = response.usage.prompt_tokens
            process_info.output_tokens = response.usage.completion_tokens
            segment.ocr = ocr_results
        elif TASK__TABLE_OCR_MODEL == "ppstructure_table":
            table_ocr_results = ppstructure_table(segment_temp_file)
            segment.ocr = table_ocr_results.results
            segment.html = table_ocr_results.html
        elif TASK__TABLE_OCR_MODEL == "textract":
            print("textract!")
            textract_results = process_table_textract(segment_temp_file, TextractFeatures.TABLES)
            segment.ocr = textract_results.results
            segment.html = textract_results.html
    else:
        if TASK__OCR_MODEL == "paddleocr":
            ocr_results = ppocr(segment_temp_file)
            segment.ocr = ocr_results
        elif TASK__OCR_MODEL == "textract":
            textract_results = process_textract(segment_temp_file)
            segment.ocr = textract_results

    process_info.avg_ocr_confidence = segment.calculate_avg_ocr_confidence()
    process_info.finalize()
    return process_info

def process_textract(image_path: Path):
    loaded_img = Image.open(image_path)
    import boto3

    # Set up default AWS profile with the session
  
    extractor=Textractor(profile_name="default")


    response = extractor.detect_document_text(
        file_source=loaded_img,
        save_image=True
    )

    ocr_result = OCRResult(
        text=response.text,
        bbox=None,
        confidence=None
    )
   
    return ocr_result


def process_table_textract(image_path: Path, feature: TextractFeatures):
    from src.configs.llm_config import LLM__MODEL, LLM__BASE_URL, LLM__INPUT_TOKEN_PRICE, LLM__OUTPUT_TOKEN_PRICE
    loaded_img = Image.open(image_path)
    import boto3
    extractor=Textractor(profile_name="default")


    response = extractor.analyze_document(
        file_source=loaded_img,
        features=[feature],
        save_image=True
    )
    ocr_results = []
    html = None

  
    if response.tables:
        table = response.tables[0]
        if table:
            df = table.to_pandas()
            html = table.to_html()
            
            for cell in table.table_cells:
                bbox = BoundingBox(
                    left=cell.bbox.x,
                    top=cell.bbox.y,
                    width=cell.bbox.width,
                    height=cell.bbox.height)
                ocr_result = OCRResult(
                    bbox=bbox,
                    text=cell.text,
                    confidence=cell.confidence
                )
                ocr_results.append(ocr_result)
            print(f"Processed {len(ocr_results)} cells in the table")
        else:
            print("No table found in the response")
            html = ""
    else:
        print("No tables found in the response")
        html = ""


    return OCRResponse(results=ocr_results, html=html)

def process_segment(
    user_id: str,
    task_id: str,
    segment: Segment,
    image_folder_location: str,
    page_image_file_paths: dict[int, Path],
    segment_image_extension: str,
    segment_image_quality: int,
    segment_image_resize: str,
    ocr_strategy: str
) -> Segment:
    try:
        ocr_needed = ocr_strategy == "All" or (
            ocr_strategy != "Off" and (
                segment.segment_type in [SegmentType.Table, SegmentType.Picture] or
                (ocr_strategy == "Auto" and not segment.content)
            )
        )

        base64_image = crop_image(
            page_image_file_paths[segment.page_number],
            segment.bbox,
            segment_image_extension,
            segment_image_quality,
            segment_image_resize
        )

        image_s3_path = f"{image_folder_location}/{segment.segment_id}.{segment_image_extension}"
        temp_image_file = tempfile.NamedTemporaryFile(
            suffix=f".{segment_image_extension}", delete=False)
        try:
            temp_image_file.write(base64.b64decode(base64_image))
            temp_image_file.close()
            upload_file_to_s3(
                temp_image_file.name,
                image_s3_path,
            )
            segment.image = image_s3_path


            if ocr_needed:
                process_info = process_segment_ocr(
                    segment,
                    Path(temp_image_file.name)
                )
                executor.submit(insert_segment_process, user_id, task_id, process_info)
        finally:
            os.remove(temp_image_file.name)
    except Exception as e:
        print(
            f"Error processing segment {segment.segment_type} on page {segment.page_number}: {e}")
    finally:
        segment.finalize()
    return segment


def insert_segment_process(
    user_id: str,
    task_id: str,
    process_info: ProcessInfo
):
    if not PG__URL:
        return
    try:
        with connect(PG__URL) as conn:
            with conn.cursor() as cur:
                query = """
                    INSERT INTO segment_process (
                        user_id, task_id, segment_id, process_type, model_name,
                        base_url, input_tokens, output_tokens, input_price,
                        output_price, total_cost, detail, latency, avg_ocr_confidence
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """
                values = (
                    user_id, task_id, process_info.segment_id, process_info.process_type, process_info.llm_model_name,
                    process_info.base_url, process_info.input_tokens, process_info.output_tokens, process_info.input_price,
                    process_info.output_price, process_info.total_cost, process_info.detail, process_info.latency, process_info.avg_ocr_confidence
                )
                cur.execute(query, values)
            conn.commit()
    except Exception as e:
        print(f"Error inserting segment process: {e}")
