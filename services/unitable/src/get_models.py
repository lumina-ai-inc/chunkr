from dotenv import load_dotenv
import os
from pathlib import Path

from src.tools import load_vocab_and_model

load_dotenv(override=True)


def get_model_name(env_var, default=None):
    return os.getenv(env_var) or default


MODEL_DIR = Path(
    f"{Path(__file__).parent.parent}/experiments/unitable_weights")
STRUCTURE_MODEL = get_model_name('STRUCTURE_MODEL')
BBOX_MODEL = get_model_name('BBOX_MODEL')
CONTENT_MODEL = get_model_name('CONTENT_MODEL')

MODEL_FILE_MAP = {
    model_type: f"{model_name}.pt"
    for model_type, model_name in [
        ('structure', STRUCTURE_MODEL),
        ('bbox', BBOX_MODEL),
        ('content', CONTENT_MODEL)
    ]
    if model_name is not None
}

if MODEL_FILE_MAP:
    assert all([(MODEL_DIR / name).is_file() for name in MODEL_FILE_MAP.values()]), \
        f"Please download model weights from HuggingFace: https://huggingface.co/poloclub/UniTable/tree/main"
else:
    raise ValueError("No models specified in environment variables.")


def init_structure_model():
    if STRUCTURE_MODEL is None:
        return None
    vocab, model = load_vocab_and_model(
        vocab_path="./vocab/vocab_html.json",
        max_seq_len=784,
        model_weights=MODEL_DIR / MODEL_FILE_MAP['structure'],
    )
    print("structure model loaded: ", MODEL_DIR / MODEL_FILE_MAP['structure'])
    return vocab, model


def init_bbox_model():
    if BBOX_MODEL is None:
        return None
    vocab, model = load_vocab_and_model(
        vocab_path="./vocab/vocab_bbox.json",
        max_seq_len=1024,
        model_weights=MODEL_DIR / MODEL_FILE_MAP['bbox'],
    )
    print("bbox model loaded: ", MODEL_DIR / MODEL_FILE_MAP['bbox'])
    return vocab, model


def init_content_model():
    if CONTENT_MODEL is None:
        return None
    vocab, model = load_vocab_and_model(
        vocab_path="./vocab/vocab_cell_6k.json",
        max_seq_len=200,
        model_weights=MODEL_DIR / MODEL_FILE_MAP['content'],
    )
    print("content model loaded: ", MODEL_DIR / MODEL_FILE_MAP['content'])
    return vocab, model
