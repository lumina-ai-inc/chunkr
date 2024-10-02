from openai import OpenAI
from openai.types.chat import ChatCompletion
from pathlib import Path
import re
from typing import Tuple

from src.configs.llm_config import LLM__BASE_URL, LLM__API_KEY, LLM__MODEL
from src.converters import to_base64
from src.resize_pipeline import resize_pipeline


client = OpenAI(base_url=LLM__BASE_URL, api_key=LLM__API_KEY)


def table_to_html(image: str, detail: str) -> str:
    response = client.chat.completions.create(
        model=LLM__MODEL,
        messages=[
            {"role": "system", "content": "You are an OCR system that converts table data to HTML."},
            {"role": "user",
             "content": [
                 {"type": "text", "text": "Convert the following table to HTML exactly"},
                 {
                     "type": "image_url",
                     "image_url": {
                         "url": image,
                         "detail": detail
                     },
                 },
             ],
             }
        ],
        temperature=0.0
    )

    return response

def extract_html_from_response(response: ChatCompletion) -> str:
    text = response.choices[0].message.content
    html_pattern = r'```html\n(.*?)```'
    match = re.search(html_pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return text.strip()


def process_table(image_path: Path) -> Tuple[ChatCompletion, float]:
    detail = resize_pipeline(image_path)
    image = f"data:image/jpeg;base64,{to_base64(image_path)}"
    response = table_to_html(image, detail)
    return (detail, response)
