from os.path import join
from pathlib import Path

PDF_TOKENS_TYPE_ROOT_PATH = Path(__file__).parent.parent.parent.absolute()
PDF_LABELED_DATA_ROOT_PATH = Path(join(PDF_TOKENS_TYPE_ROOT_PATH.parent.absolute(), "pdf-labeled-data"))
TOKEN_TYPE_LABEL_PATH = Path(join(PDF_LABELED_DATA_ROOT_PATH, "labeled_data", "token_type"))

TRAINED_MODEL_PATH = join(PDF_TOKENS_TYPE_ROOT_PATH, "model", "pdf_tokens_type.model")
TOKEN_TYPE_RELATIVE_PATH = join("labeled_data", "token_type")
MISTAKES_RELATIVE_PATH = join("labeled_data", "task_mistakes")

XML_NAME = "etree.xml"
LABELS_FILE_NAME = "labels.json"
STATUS_FILE_NAME = "status.txt"

CHARACTER_TYPE = [
    "Lt",
    "Lo",
    "Sk",
    "Lm",
    "Sm",
    "Cf",
    "Nl",
    "Pe",
    "Po",
    "Pd",
    "Me",
    "Sc",
    "Ll",
    "Pf",
    "Mc",
    "Lu",
    "Zs",
    "Cn",
    "Cc",
    "No",
    "Co",
    "Ps",
    "Nd",
    "Mn",
    "Pi",
    "So",
    "Pc",
]

if __name__ == "__main__":
    print(PDF_TOKENS_TYPE_ROOT_PATH)
    print(PDF_LABELED_DATA_ROOT_PATH)
