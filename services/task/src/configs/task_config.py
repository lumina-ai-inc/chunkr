import dotenv
import os

dotenv.load_dotenv(override=True)

TASK__OCR_CONFIDENCE_THRESHOLD = os.getenv("TASK__OCR_CONFIDENCE_THRESHOLD") if os.getenv("TASK__OCR_CONFIDENCE_THRESHOLD") else 0.85