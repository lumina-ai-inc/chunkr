import pickle
import shutil

import numpy as np
from os import makedirs
from os.path import join, exists
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_features.PdfFeatures import PdfFeatures

from bros.tokenization_bros import BrosTokenizer
from configuration import WORD_GRIDS_PATH

tokenizer = BrosTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")


def rectangle_to_bbox(rectangle: Rectangle):
    return [rectangle.left, rectangle.top, rectangle.width, rectangle.height]


def get_words_positions(text: str, rectangle: Rectangle):
    text = text.strip()
    text_len = len(text)

    width_per_letter = rectangle.width / text_len

    words_bboxes = [Rectangle(rectangle.left, rectangle.top, rectangle.left + 5, rectangle.bottom)]
    words_bboxes[-1].width = 0
    words_bboxes[-1].right = words_bboxes[-1].left

    for letter in text:
        if letter == " ":
            left = words_bboxes[-1].right + width_per_letter
            words_bboxes.append(Rectangle(left, words_bboxes[-1].top, left + 5, words_bboxes[-1].bottom))
            words_bboxes[-1].width = 0
            words_bboxes[-1].right = words_bboxes[-1].left
        else:
            words_bboxes[-1].right = words_bboxes[-1].right + width_per_letter
            words_bboxes[-1].width = words_bboxes[-1].width + width_per_letter

    words = text.split()
    return words, words_bboxes


def get_subwords_positions(word: str, rectangle: Rectangle):
    width_per_letter = rectangle.width / len(word)
    word_tokens = [x.replace("#", "") for x in tokenizer.tokenize(word)]

    if not word_tokens:
        return [], []

    ids = [x[-2] for x in tokenizer(word_tokens)["input_ids"]]

    right = rectangle.left + len(word_tokens[0]) * width_per_letter
    bboxes = [Rectangle(rectangle.left, rectangle.top, right, rectangle.bottom)]

    for subword in word_tokens[1:]:
        right = bboxes[-1].right + len(subword) * width_per_letter
        bboxes.append(Rectangle(bboxes[-1].right, rectangle.top, right, rectangle.bottom))

    return ids, bboxes


def get_grid_words_dict(tokens: list[PdfToken]):
    texts, bbox_texts_list, inputs_ids, bbox_subword_list = [], [], [], []
    for token in tokens:
        words, words_bboxes = get_words_positions(token.content, token.bounding_box)
        texts += words
        bbox_texts_list += [rectangle_to_bbox(r) for r in words_bboxes]
        for word, word_box in zip(words, words_bboxes):
            ids, subwords_bboxes = get_subwords_positions(word, word_box)
            inputs_ids += ids
            bbox_subword_list += [rectangle_to_bbox(r) for r in subwords_bboxes]

    return {
        "input_ids": np.array(inputs_ids),
        "bbox_subword_list": np.array(bbox_subword_list),
        "texts": texts,
        "bbox_texts_list": np.array(bbox_texts_list),
    }


def create_word_grid(pdf_features_list: list[PdfFeatures]):
    makedirs(WORD_GRIDS_PATH, exist_ok=True)

    for pdf_features in pdf_features_list:
        for page in pdf_features.pages:
            image_id = f"{pdf_features.file_name}_{page.page_number - 1}"
            if exists(join(WORD_GRIDS_PATH, image_id + ".pkl")):
                continue
            grid_words_dict = get_grid_words_dict(page.tokens)
            with open(join(WORD_GRIDS_PATH, f"{image_id}.pkl"), mode="wb") as file:
                pickle.dump(grid_words_dict, file)


def remove_word_grids():
    shutil.rmtree(WORD_GRIDS_PATH, ignore_errors=True)
