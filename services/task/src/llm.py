import base64
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pathlib import Path
import time

from src.configs.llm_config import OPEN_AI__BASE_URL, OPEN_AI__API_KEY, OPEN_AI__MODEL, OPEN_AI_INPUT_TOKEN_PRICE, OPEN_AI_OUTPUT_TOKEN_PRICE
from src.converters import resize_image

client = OpenAI(base_url=OPEN_AI__BASE_URL, api_key=OPEN_AI__API_KEY)

def table_to_html(image: str) -> str:
    response = client.chat.completions.create(
        model=OPEN_AI__MODEL,
        messages=[
            {"role": "system", "content": "You are an OCR system that converts table data to HTML."},
            {"role": "user",
             "content": [
                 {"type": "text", "text": "Convert the following table to HTML"},
                 {
                     "type": "image_url",
                     "image_url": {
                         "url": image,
                         "detail": "low"
                     },
                 },
             ],
             }
        ],
    )

    return response


def calculate_cost(response: ChatCompletion):
    cost = OPEN_AI_INPUT_TOKEN_PRICE * response.usage.prompt_tokens + \
        OPEN_AI_OUTPUT_TOKEN_PRICE * response.usage.completion_tokens
    return cost

def to_base64(image: str) -> str:
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_table(image_path: Path) -> str:
    image = resize_image(image_path, (512, 512), "jpg", 100)
    response = table_to_html(f"data:image/jpeg;base64,{image}")
    cost = calculate_cost(response)
    print(response)
    print(response.choices[0].message.content)
    print(f"Time taken: {time.time() - start_time} seconds")
    print(f"Cost: ${cost} | ¢{cost * 100}")

if __name__ == "__main__":
    input_token_price = 0.15 / 1_000_000
    output_token_price = 0.6 / 1_000_000
    start_time = time.time()
    image_path = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/llm/c86ce3d2-295a-4d1b-a8c5-6fe427b6da21.jpg"
    image = resize_image(image_path, (512, 512), "jpg", 100)
    response = table_to_html(f"data:image/jpeg;base64,{image}")
    cost = calculate_cost(input_token_price, output_token_price, response)
    print(response)
    print(response.choices[0].message.content)
    print(f"Time taken: {time.time() - start_time} seconds")
    print(f"Cost: ${cost} | ¢{cost * 100}")

