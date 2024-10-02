import os
import dotenv

dotenv.load_dotenv(override=True)

LLM__BASE_URL: str = os.getenv("LLM__BASE_URL")
LLM__MODEL: str = os.getenv("LLM__MODEL")
LLM__API_KEY: str = os.getenv("LLM__API_KEY")
LLM__MAX_WIDTH: int = int(os.getenv("LLM__MAX_WIDTH") or 2048)
LLM__MAX_HEIGHT: int = int(os.getenv("LLM__MAX_HEIGHT") or 1024)

if LLM__MODEL == "gpt-4o-mini":
    LLM__INPUT_TOKEN_PRICE: float = float(
        os.getenv("LLM__INPUT_TOKEN_PRICE") or 0.15 / 1_000_000)
    LLM__OUTPUT_TOKEN_PRICE: float = float(
        os.getenv("LLM__OUTPUT_TOKEN_PRICE") or 0.6 / 1_000_000)
elif LLM__MODEL == "gpt-4o":
    LLM__INPUT_TOKEN_PRICE: float = float(
        os.getenv("LLM__INPUT_TOKEN_PRICE") or 5 / 1_000_000)
    LLM__OUTPUT_TOKEN_PRICE: float = float(
        os.getenv("LLM_OUTPUT_TOKEN_PRICE") or 15 / 1_000_000)
else:
    LLM__INPUT_TOKEN_PRICE: float = float(
        os.getenv("LLM__INPUT_TOKEN_PRICE") or 0)
    LLM__OUTPUT_TOKEN_PRICE: float = float(
        os.getenv("LLM__OUTPUT_TOKEN_PRICE") or 0)
