import os
import dotenv

dotenv.load_dotenv(override=True)

LLM__BASE_URL = os.getenv("OPEN_AI__BASE_URL")
LLM__MODEL = os.getenv("OPEN_AI__MODEL")
LLM__API_KEY = os.getenv("OPEN_AI__API_KEY")
LLM_INPUT_TOKEN_PRICE = 0.15 / 1_000_000
LLM_OUTPUT_TOKEN_PRICE = 0.6 / 1_000_000
LLM__MAX_HEIGHT = 1024
LLM__MAX_WIDTH = 2046   