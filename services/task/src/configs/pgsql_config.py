import os
from dotenv import load_dotenv

load_dotenv(override=True)

PG__URL: str = os.getenv("PG__URL")
if not PG__URL:
    raise ValueError("PG__URL is not set")