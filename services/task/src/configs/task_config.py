import dotenv
import os

dotenv.load_dotenv(override=True)

TASK__OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("TASK__OCR_CONFIDENCE_THRESHOLD") if os.getenv("TASK__OCR_CONFIDENCE_THRESHOLD") else 0.85)
TASK__OCR_MAX_SIZE: int = int(os.getenv("TASK__OCR_MAX_SIZE") if os.getenv("TASK__OCR_MAX_SIZE") else 4000)
TASK__OCR_POOL_SIZE: int = int(os.getenv("TASK__OCR_POOL_SIZE") if os.getenv("TASK__OCR_POOL_SIZE") else 4)
TASK__TABLE_ENGINE_POOL_SIZE: int = int(os.getenv("TASK__TABLE_ENGINE_POOL_SIZE") if os.getenv("TASK__TABLE_ENGINE_POOL_SIZE") else 4)
