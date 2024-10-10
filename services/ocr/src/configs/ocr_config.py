import dotenv
import os

dotenv.load_dotenv(override=True)

OCR__MAX_SIZE: int = int(os.getenv("OCR__MAX_SIZE") if os.getenv("OCR__MAX_SIZE") else 4000)
