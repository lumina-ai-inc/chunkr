from pathlib import Path
from PIL import Image
from src.configs.llm_config import LLM__MAX_HEIGHT, LLM__MAX_WIDTH

def resize_pipeline(image_path: Path) -> str:
    with Image.open(image_path) as img:
        width, height = img.size

    if width >= LLM__MAX_WIDTH or height >= LLM__MAX_HEIGHT :
        return 'auto'
    else:
        return 'low'
