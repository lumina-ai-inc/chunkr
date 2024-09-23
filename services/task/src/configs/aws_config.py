import dotenv
import os

dotenv.load_dotenv(override=True)

AWS__ENDPOINT = os.getenv("AWS__ENDPOINT")
AWS__ACCESS_KEY = os.getenv("AWS__ACCESS_KEY")
AWS__SECRET_KEY = os.getenv("AWS__SECRET_KEY")
AWS__REGION = os.getenv("AWS__REGION")