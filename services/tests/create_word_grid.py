import numpy as np
from transformers import AutoTokenizer

# We'll use a pretrained BROS tokenizer for subword token splitting
# Feel free to use any other tokenizer if desired
tokenizer = AutoTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")

def get_words_positions(text: str, bbox: list[int]) -> tuple[list[str], list[list[float]]]:
    """
    Given a token's text and bounding box [left, top, width, height],
    split the text into words and assign bounding boxes to words.

    Returns:
        words: a list of word strings
        words_bboxes: a list of bounding boxes [left, top, width, height] for each word
    """
    # Each token has {left, top, width, height}
    left, top, width, height = bbox
    text = text.strip()
    text_len = len(text)

    # If there's no text or zero length, handle gracefully
    if text_len == 0:
        return [], []

    # Approximate character width based on bounding box
    width_per_letter = width / text_len

    # Start building word bounding boxes
    # We begin with a small bounding box that we'll expand as we iterate
    words_bboxes = [[left, top, 0, height]]
    words_bboxes[-1][2] = 0  # set width = 0 for the first "seed"

    current_right = left

    for letter in text:
        if letter == " ":
            # If we see a space, we start a new bounding box for the next word
            # Next left is current_right + width_per_letter
            new_left = current_right + width_per_letter
            words_bboxes.append([new_left, top, 0, height])
            current_right = new_left
        else:
            # Expand the existing bounding box
            current_right += width_per_letter
            # Recompute the current box width
            words_bboxes[-1][2] = current_right - words_bboxes[-1][0]

    # Note: The 'height' of each word box is the same as the original token's height

    words = text.split()
    return words, words_bboxes


def get_subwords_positions(word: str, bbox: list[int]) -> tuple[list[int], list[list[float]]]:
    """
    Given a word and bounding box [left, top, width, height],
    tokenize the word into subwords (using the BROS tokenizer) and assign bounding boxes.

    Returns:
        ids: list of subword token IDs (or input IDs) from the tokenizer
        bboxes: list of bounding boxes [left, top, width, height] for each subword
    """
    left, top, width, height = bbox

    # Avoid zero division
    if len(word) == 0:
        return [], []

    # Tokenize with BROS, removing '#' from subwords
    word_tokens = [x.replace("#", "") for x in tokenizer.tokenize(word)]
    if not word_tokens:
        return [], []

    # Convert to IDs. The original code used [-2], but that was for demonstration.
    # We'll just get the entire result and flatten it for example. Modify as needed.
    tokenized_dict = tokenizer(word_tokens, add_special_tokens=False)
    # We'll store the raw input_ids here, but you might adjust depending on your use case
    ids = []
    for each_id_list in tokenized_dict["input_ids"]:
        # It's typically a single ID for each subword in this approach
        if isinstance(each_id_list, list):
            # Tokenizer might return nested lists. Flatten if needed.
            ids.extend(each_id_list)
        else:
            ids.append(each_id_list)

    # Now split the bounding box proportionally
    width_per_letter = width / len(word)
    current_right = left + len(word_tokens[0]) * width_per_letter
    bboxes = [[left, top, current_right - left, height]]

    for subword in word_tokens[1:]:
        new_left = current_right
        current_right = current_right + len(subword) * width_per_letter
        subword_width = current_right - new_left
        bboxes.append([new_left, top, subword_width, height])

    return ids, bboxes


def get_grid_words_dict(tokens: list[dict]) -> dict:
    """
    tokens: a list of tokens, each token is {
      "content": str,
      "bbox": [left, top, width, height]
    }

    Returns a dictionary describing:
      - "input_ids" : np.array of subword IDs
      - "bbox_subword_list" : np.array of subword bounding boxes
      - "texts" : list of words
      - "bbox_texts_list" : np.array of word bounding boxes
    """
    texts = []
    bbox_texts_list = []
    inputs_ids = []
    bbox_subword_list = []

    for token in tokens:
        text = token["content"]
        bounding_box = token["bbox"]

        words, words_bboxes = get_words_positions(text, bounding_box)
        texts.extend(words)
        # Convert each word bounding box to [left, top, width, height] form
        bbox_texts_list.extend(words_bboxes)

        # for each word and bounding box, get subword positions
        for w, w_bbox in zip(words, words_bboxes):
            subword_ids, subword_bboxes = get_subwords_positions(w, w_bbox)
            inputs_ids.extend(subword_ids)
            bbox_subword_list.extend(subword_bboxes)

    return {
        "input_ids": np.array(inputs_ids, dtype=np.int32),
        "bbox_subword_list": np.array(bbox_subword_list, dtype=np.float32),
        "texts": texts,
        "bbox_texts_list": np.array(bbox_texts_list, dtype=np.float32),
    }


def create_word_grid(pdfs_data: list[dict]) -> list[dict]:
    """
    pdfs_data is a list of PDF metadata. Each PDF structure is:
    {
      "file_name": str,
      "pages": [
         {
            "page_number": int,
            "tokens": [
               {"content": str, "bbox": [left, top, width, height]},
               ...
            ]
         },
         ...
      ]
    }

    Returns:
        A list of "grid" structures (dictionaries), one per page in all PDFs:
         [
            {
              "input_ids": np.array,
              "bbox_subword_list": np.array,
              "texts": list[str],
              "bbox_texts_list": np.array
            },
            ...
         ]
    """
    grid_list = []
    for pdf_data in pdfs_data:
        for page in pdf_data["pages"]:
            # get the word grid on each page
            grid_words_dict = get_grid_words_dict(page["tokens"])
            grid_list.append(grid_words_dict)
    return grid_list