import os

current_dir = os.path.dirname(os.path.abspath(__file__))

TABLE_STRUCTURE_LOCAL_MODEL_PATH = os.path.join(current_dir, "models/table-structure-recognition-v1.1-all")
TABLE_STRUCTURE_REMOTE_MODEL_NAME = "microsoft/table-structure-recognition-v1.1-all"

EASYOCR_LOCAL_MODEL_PATH = os.path.join(current_dir, "models/easyocr")