import torch
import dotenv
import os

dotenv.load_dotenv(override=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_SIZE = 1000

def get_tesseract_url():
    return os.getenv("TESSERACT_URL")