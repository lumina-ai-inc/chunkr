import os
import dotenv

dotenv.load_dotenv(override=True)

OPEN_AI__BASE_URL = os.getenv("OPEN_AI__BASE_URL")
OPEN_AI__MODEL = os.getenv("OPEN_AI__MODEL")
OPEN_AI__API_KEY = os.getenv("OPEN_AI__API_KEY")
OPEN_AI_INPUT_TOKEN_PRICE = 0.15 / 1_000_000
OPEN_AI_OUTPUT_TOKEN_PRICE = 0.6 / 1_000_000
