from pathlib import Path
from PIL import Image
from src.configs.llm_config import LLM__MAX_HEIGHT, LLM__MAX_WIDTH

def resize_pipeline(image_path: Path) -> str:
    with Image.open(image_path) as img:
        width, height = img.size

    print(height, width)

    if height <= LLM__MAX_HEIGHT or width <= LLM__MAX_WIDTH:
        return 'low'
    else:
        return 'auto'