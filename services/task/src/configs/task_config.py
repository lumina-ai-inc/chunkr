import dotenv
import os

dotenv.load_dotenv(override=True)

TASK__OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("TASK__OCR_CONFIDENCE_THRESHOLD") if os.getenv("TASK__OCR_CONFIDENCE_THRESHOLD") else 0.85)
TASK__OCR_SERVICE_URL: str = os.getenv("TASK__OCR_SERVICE_URL") if os.getenv("TASK__OCR_SERVICE_URL") else "http://localhost:8000"