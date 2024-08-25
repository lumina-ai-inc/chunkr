import json
import pickle
from os.path import join
from pathlib import Path
from statistics import mode

from fast_trainer.PdfSegment import PdfSegment
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from data_model.PdfImages import PdfImages
from configuration import ROOT_PATH, DOCLAYNET_TYPE_BY_ID
from data_model.Prediction import Prediction


def get_prediction_from_annotation(annotation, images_names, vgt_predictions_dict):
    pdf_name = images_names[annotation["image_id"]][:-4]
    category_id = annotation["category_id"]
    bounding_box = Rectangle.from_width_height(
        left=int(annotation["bbox"][0]),
        top=int(annotation["bbox"][1]),
        width=int(annotation["bbox"][2]),
        height=int(annotation["bbox"][3]),
    )

    prediction = Prediction(
        bounding_box=bounding_box, category_id=category_id, score=round(float(annotation["score"]) * 100, 2)
    )
    vgt_predictions_dict.setdefault(pdf_name, list()).append(prediction)


def get_vgt_predictions(model_name: str) -> dict[str, list[Prediction]]:
    output_dir: str = f"model_output_{model_name}"
    model_output_json_path = join(str(ROOT_PATH), output_dir, "inference", "coco_instances_results.json")
    annotations = json.loads(Path(model_output_json_path).read_text())

    test_json_path = join(str(ROOT_PATH), "jsons", "test.json")
    coco_truth = json.loads(Path(test_json_path).read_text())

    images_names = {value["id"]: value["file_name"] for value in coco_truth["images"]}

    vgt_predictions_dict = dict()
    for annotation in annotations:
        get_prediction_from_annotation(annotation, images_names, vgt_predictions_dict)

    return vgt_predictions_dict


def find_best_prediction_for_token(page_pdf_name, token, vgt_predictions_dict, most_probable_tokens_by_predictions):
    best_score: float = 0
    most_probable_prediction: Prediction | None = None
    for prediction in vgt_predictions_dict[page_pdf_name]:
        if prediction.score > best_score and prediction.bounding_box.get_intersection_percentage(token.bounding_box):
            best_score = prediction.score
            most_probable_prediction = prediction
            if best_score >= 99:
                break
    if most_probable_prediction:
        most_probable_tokens_by_predictions.setdefault(most_probable_prediction, list()).append(token)
    else:
        dummy_prediction = Prediction(bounding_box=token.bounding_box, category_id=10, score=0.0)
        most_probable_tokens_by_predictions.setdefault(dummy_prediction, list()).append(token)


def get_merged_prediction_type(to_merge: list[Prediction]):
    table_exists = any([p.category_id == 9 for p in to_merge])
    if not table_exists:
        return mode([p.category_id for p in sorted(to_merge, key=lambda x: -x.score)])
    return 9


def merge_colliding_predictions(predictions: list[Prediction]):
    predictions = [p for p in predictions if not p.score < 20]
    while True:
        new_predictions, merged = [], False
        while predictions:
            p1 = predictions.pop(0)
            to_merge = [p for p in predictions if p1.bounding_box.get_intersection_percentage(p.bounding_box) > 0]
            for prediction in to_merge:
                predictions.remove(prediction)
            if to_merge:
                to_merge.append(p1)
                p1.bounding_box = Rectangle.merge_rectangles([prediction.bounding_box for prediction in to_merge])
                p1.category_id = get_merged_prediction_type(to_merge)
                merged = True
            new_predictions.append(p1)
        if not merged:
            return new_predictions
        predictions = new_predictions


def get_pdf_segments_for_page(page, pdf_name, page_pdf_name, vgt_predictions_dict):
    most_probable_pdf_segments_for_page: list[PdfSegment] = []
    most_probable_tokens_by_predictions: dict[Prediction, list[PdfToken]] = {}
    vgt_predictions_dict[page_pdf_name] = merge_colliding_predictions(vgt_predictions_dict[page_pdf_name])

    for token in page.tokens:
        find_best_prediction_for_token(page_pdf_name, token, vgt_predictions_dict, most_probable_tokens_by_predictions)

    for prediction, tokens in most_probable_tokens_by_predictions.items():
        new_segment = PdfSegment.from_pdf_tokens(tokens, pdf_name)
        new_segment.bounding_box = prediction.bounding_box
        new_segment.segment_type = TokenType.from_text(DOCLAYNET_TYPE_BY_ID[prediction.category_id])
        most_probable_pdf_segments_for_page.append(new_segment)

    no_token_predictions = [
        prediction
        for prediction in vgt_predictions_dict[page_pdf_name]
        if prediction not in most_probable_tokens_by_predictions
    ]

    for prediction in no_token_predictions:
        segment_type = TokenType.from_text(DOCLAYNET_TYPE_BY_ID[prediction.category_id])
        page_number = page.page_number
        new_segment = PdfSegment(page_number, prediction.bounding_box, "", segment_type, pdf_name)
        most_probable_pdf_segments_for_page.append(new_segment)

    return most_probable_pdf_segments_for_page


def prediction_exists_for_page(page_pdf_name, vgt_predictions_dict):
    return page_pdf_name in vgt_predictions_dict


def get_most_probable_pdf_segments(model_name: str, pdf_images_list: list[PdfImages], save_output: bool = False):
    most_probable_pdf_segments: list[PdfSegment] = []
    vgt_predictions_dict = get_vgt_predictions(model_name)
    pdf_features_list: list[PdfFeatures] = [pdf_images.pdf_features for pdf_images in pdf_images_list]
    for pdf_features in pdf_features_list:
        for page in pdf_features.pages:
            page_pdf_name = pdf_features.file_name + "_" + str(page.page_number - 1)
            if not prediction_exists_for_page(page_pdf_name, vgt_predictions_dict):
                continue
            page_segments = get_pdf_segments_for_page(page, pdf_features.file_name, page_pdf_name, vgt_predictions_dict)
            most_probable_pdf_segments.extend(page_segments)
    if save_output:
        save_path = join(ROOT_PATH, f"model_output_{model_name}", "predicted_segments.pickle")
        with open(save_path, mode="wb") as file:
            pickle.dump(most_probable_pdf_segments, file)
    return most_probable_pdf_segments
