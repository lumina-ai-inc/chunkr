from typing import List, Tuple
import random
import tokenizers as tk
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.vocab import TASK_TOKENS, CELL_SPECIAL
from src.model.encoderdecoder import EncoderDecoder
from .misc import html_table_template

__all__ = [
    "subsequent_mask",
    "combine_cell_char_seq",
    "random_continuous_sequence",
    "prepare_html_seq",
    "prepare_cell_seq",
    "prepare_bbox_seq",
    "html_str_to_token_list",
    "cell_str_to_token_list",
    "bbox_str_to_token_list",
    "build_table_from_html_and_cell",
    "pred_token_within_range",
    "batch_autoregressive_decode",
    "greedy_sampling",
    "combine_filename_pred_gt",
]


def subsequent_mask(size: int, pad: int = 0):
    attn_shape = (size, size)
    output = torch.triu(torch.ones(attn_shape), diagonal=1).to(torch.bool)
    if pad and pad > 0:
        output[:pad] = False
    return output


def combine_cell_char_seq(seq: List[str]) -> str:
    """Replace empty token with <empty> in vocab. combine characters into a str"""
    if seq:
        out = "".join(seq)
    else:
        out = "<empty>"
    return out


def prepare_html_seq(seq: List[str]) -> List[str]:
    """Convert html annotations to html training template."""
    out = ["[html]", *seq, "<eos>"]
    return out


def prepare_cell_seq(seq: str) -> List[str]:
    """Convert cell sequence to training template."""
    for black in CELL_SPECIAL:
        seq = seq.replace(black, "")
    out = ["[cell]", seq, "<eos>"]

    return out


def prepare_bbox_seq(seq: List[dict]):
    tmp = [f"bbox-{round(i)}" for i in seq]
    out = ["[bbox]"] + tmp + ["<eos>"]

    return out


def random_continuous_sequence(seq: List, N: int, length: int = 10) -> List:
    """Randomly sample a continuous sub-sequence from a sequence for N times."""
    start_idx = [random.randrange(len(seq)) for _ in range(N)]
    subseq_len = [random.randrange(1, length) for _ in range(N)]
    output = [(i, min(i + j, len(seq))) for i, j in zip(start_idx, subseq_len)]

    return output


# def prepare_bbox_seq(
#     seq: List[dict],
#     N: int,
#     delimiter: str = "<sep>",
# ) -> List[List[str]]:
#     """Convert the annotation to bbox input/output sequence."""
#     out = list()
#     # bbox_loss_start_idx = list()

#     subseq_idx = random_continuous_sequence(seq, N)

#     for idx in subseq_idx:
#         entry = seq[idx[0] : idx[1]]
#         tmp = list()
#         bbox_seq = list()
#         for i in entry:
#             if "tokens" in i.keys():
#                 # pubtabnet and synthtabnet
#                 tmp.append(combine_cell_char_seq(i["tokens"]))
#                 if "bbox" in i.keys():
#                     bbox_seq.extend([f"bbox-{round(j)}" for j in i["bbox"]])
#             elif "text" in i.keys():
#                 # pubtables and icdar
#                 tmp.append(i["text"])
#                 if "bbox" in i.keys():
#                     bbox_seq.extend([f"bbox-{round(j)}" for j in i["bbox"]])

#         cell_seq = [delimiter] * len(tmp)
#         cell_seq = [q for pair in zip(tmp, cell_seq) for q in pair]
#         cell_seq = ["[bbox]", f"{len(entry)}-cell(s)", delimiter] + cell_seq

#         bbox_seq.append("<eos>")
#         # bbox_loss_start_idx.append(len(cell_seq))
#         out.append(cell_seq + bbox_seq)

#     return out


def html_str_to_token_list(
    seq: str, splitter: tk.pre_tokenizers.PreTokenizer = None
) -> List[str]:
    """Convert decode output (str) to a list of tokens for constructing html table code"""

    # works for no <eos>
    seq = seq.split("<eos>")[0]

    token_black_list = ["<eos>", "<pad>", *TASK_TOKENS]
    for i in token_black_list:
        seq = seq.replace(i, "")

    if not splitter:
        splitter = tk.pre_tokenizers.Split(pattern=" ", behavior="contiguous")

    seq = splitter.pre_tokenize_str(seq)
    # only preserve the space for spanning cell tokens
    seq = [i[0] for i in seq if len(i[0].strip()) != 0 or i[1][1] - i[1][0] != 1]

    return seq


def cell_str_to_token_list(seq: str) -> List[str]:
    seq = seq.split("<eos>")[0]

    token_black_list = ["<eos>", "<pad>", *TASK_TOKENS]
    for i in token_black_list:
        seq = seq.replace(i, "")

    seq = seq.strip()

    return seq


def build_table_from_html_and_cell(
    structure: List[str], content: List[str] = None
) -> List[str]:
    """Build table from html and cell token list"""
    assert structure is not None
    html_code = list()

    # deal with empty table
    if content is None:
        content = ["placeholder"] * len(structure)

    for tag in structure:
        if tag in ("<td>[]</td>", ">[]</td>"):
            if len(content) == 0:
                continue
            cell = content.pop(0)
            html_code.append(tag.replace("[]", cell))
        else:
            html_code.append(tag)

    return html_code


def bbox_str_to_token_list(
    seq: str, splitter: tk.pre_tokenizers.PreTokenizer = None
) -> List[List[int]]:
    """
    Note the out could be an empty list

    return
    [[ymin, xmin, ymax, xmax],
     [ymin, xmin, ymax, xmax],
    ...
    ]
    """

    seq = seq.split("<eos>")[0]

    token_black_list = ["<eos>", "<pad>", *TASK_TOKENS]
    for i in token_black_list:
        seq = seq.replace(i, "")

    if not splitter:
        splitter = tk.pre_tokenizers.Split(pattern=" ", behavior="removed")

    seq = splitter.pre_tokenize_str(seq)
    seq = [int(i[0].split("-")[1]) for i in seq]

    rounded_seq_len = len(seq) // 4 * 4
    out = [seq[i : i + 4] for i in range(0, rounded_seq_len, 4)]
    return out


def pred_token_within_range(
    pred: Tensor,
    white_list: List[int] = None,
    black_list: List[int] = None,
) -> Tensor:
    assert white_list is None or black_list is None
    if white_list:
        total = set([i for i in range(pred.shape[-1])])
        black_list = list(total.difference(set(white_list)))

    pred[..., black_list] = -float("inf")

    return pred


def greedy_sampling(logits: Tensor):
    """logits should have shape [B, |V|]."""
    probs = F.softmax(logits, dim=-1)
    next_probs, next_tokens = probs.topk(1)

    return next_probs, next_tokens


def batch_autoregressive_decode(
    device: int,
    model: EncoderDecoder,
    batch_data,
    prefix: List[int],
    max_decode_len: int,
    eos_id: int,
    valid_token_whitelist: List[int] = None,
    valid_token_blacklist: List[int] = None,
    sampling: str = "greedy",
    use_ddp: bool = True,
) -> Tensor:
    """Auto-regressively generate the output."""

    model.eval()
    with torch.no_grad():
        if use_ddp:
            memory = model.module.encode(batch_data.image)
        else:
            memory = model.encode(batch_data.image)

    B = batch_data.image.shape[0]

    context = torch.tensor(prefix, dtype=torch.int32).repeat(B, 1).to(device)

    for _ in range(max_decode_len):
        eos_flag = [eos_id in k for k in context]
        if all(eos_flag):
            break

        # as long as one sample hasn't reached <eos>, continue decoding until the max seq len
        causal_mask = subsequent_mask(context.shape[1]).to(device)

        with torch.no_grad():
            if use_ddp:
                logits = model.module.decode(
                    memory, context, tgt_mask=causal_mask, tgt_padding_mask=None
                )
                logits = model.module.generator(logits)[:, -1, :]
            else:
                logits = model.decode(
                    memory, context, tgt_mask=causal_mask, tgt_padding_mask=None
                )
                logits = model.generator(logits)[:, -1, :]

        logits = pred_token_within_range(
            logits.detach(),
            white_list=valid_token_whitelist if valid_token_whitelist else None,
            black_list=valid_token_blacklist if valid_token_blacklist else None,
        )

        if sampling == "greedy":
            next_probs, next_tokens = greedy_sampling(logits)
        else:
            raise NotImplementedError

        context = torch.cat([context, next_tokens], dim=1)

    return context


def combine_filename_pred_gt(
    filename: List[str], pred_id: Tensor, gt_id: Tensor, vocab: tk.Tokenizer, type: str
) -> dict:
    out = dict()

    assert len(filename) == len(pred_id)

    pred_id = pred_id.detach().cpu().numpy()
    gt_id = gt_id.detach().cpu().numpy()

    pred_token = vocab.decode_batch(pred_id, skip_special_tokens=False)
    gt_token = vocab.decode_batch(gt_id, skip_special_tokens=False)

    for idx, name in enumerate(filename):
        if type == "html":
            pred_token_list = html_str_to_token_list(pred_token[idx])
            gt_token_list = html_str_to_token_list(gt_token[idx])
        elif type == "cell":
            pred_token_list = cell_str_to_token_list(pred_token[idx])
            gt_token_list = cell_str_to_token_list(gt_token[idx])
        elif type == "bbox":
            pred_token_list = bbox_str_to_token_list(pred_token[idx])
            gt_token_list = bbox_str_to_token_list(gt_token[idx])
        else:
            raise ValueError(
                f"The supported tasks are html, cell and bbox, while {type} is provided."
            )

        out[name] = dict(pred=pred_token_list, gt=gt_token_list)

    return out
