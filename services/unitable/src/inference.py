from PIL import Image
import re
import torch
from typing import Sequence
from src.trainer.utils import VALID_HTML_TOKEN, VALID_BBOX_TOKEN, INVALID_CELL_TOKEN
from src.tools import autoregressive_decode, image_to_tensor, rescale_bbox
from src.utils import bbox_str_to_token_list, cell_str_to_token_list, html_str_to_token_list


def run_structure_inference(structure_model: tuple, image: Image):
    vocab, model = structure_model
    image_tensor = image_to_tensor(image, size=(448, 448))

    pred_html = autoregressive_decode(
        model=model,
        image=image_tensor,
        prefix=[vocab.token_to_id("[html]")],
        max_decode_len=512,
        eos_id=vocab.token_to_id("<eos>"),
        token_whitelist=[vocab.token_to_id(i) for i in VALID_HTML_TOKEN],
        token_blacklist=None
    )

    pred_html = pred_html.detach().cpu().numpy()[0]
    pred_html = vocab.decode(pred_html, skip_special_tokens=False)
    pred_html = html_str_to_token_list(pred_html)

    return pred_html


def run_bbox_inference(bbox_model: tuple, image: Image):
    vocab, model = bbox_model
    image_tensor = image_to_tensor(image, size=(448, 448))
    image_size = image.size

    pred_bbox = autoregressive_decode(
        model=model,
        image=image_tensor,
        prefix=[vocab.token_to_id("[bbox]")],
        max_decode_len=1024,
        eos_id=vocab.token_to_id("<eos>"),
        token_whitelist=[vocab.token_to_id(i)
                         for i in VALID_BBOX_TOKEN[: 449]],
        token_blacklist=None
    )

    pred_bbox = pred_bbox.detach().cpu().numpy()[0]
    pred_bbox = vocab.decode(pred_bbox, skip_special_tokens=False)
    pred_bbox = bbox_str_to_token_list(pred_bbox)
    pred_bbox = rescale_bbox(pred_bbox, src=(448, 448), tgt=image_size)

    return pred_bbox


def run_content_inference(content_model: tuple, image: Image, pred_bbox: Sequence[Sequence[float]]):
    vocab, model = content_model
    image_tensor = [image_to_tensor(image.crop(bbox), size=(112, 448)) for bbox in pred_bbox]
    image_tensor = torch.cat(image_tensor, dim=0)

    pred_cell = autoregressive_decode(
        model=model,
        image=image_tensor,
        prefix=[vocab.token_to_id("[cell]")],
        max_decode_len=200,
        eos_id=vocab.token_to_id("<eos>"),
        token_whitelist=None,
        token_blacklist = [vocab.token_to_id(i) for i in INVALID_CELL_TOKEN]
    )

    pred_cell = pred_cell.detach().cpu().numpy()
    pred_cell = vocab.decode_batch(pred_cell, skip_special_tokens=False)
    pred_cell = [cell_str_to_token_list(i) for i in pred_cell]
    pred_cell = [re.sub(r'(\d).\s+(\d)', r'\1.\2', i) for i in pred_cell]

    return pred_cell
