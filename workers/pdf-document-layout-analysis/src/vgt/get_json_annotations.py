import json
from os import makedirs
from pdf_features.PdfToken import PdfToken
from data_model.PdfImages import PdfImages
from configuration import DOCLAYNET_TYPE_BY_ID
from configuration import JSONS_ROOT_PATH, JSON_TEST_FILE_PATH


def save_annotations_json(annotations: list, width_height: list, images: list):
    images_dict = [
        {
            "id": i,
            "file_name": image_id + ".jpg",
            "width": width_height[images.index(image_id)][0],
            "height": width_height[images.index(image_id)][1],
        }
        for i, image_id in enumerate(images)
    ]

    categories_dict = [{"id": key, "name": value} for key, value in DOCLAYNET_TYPE_BY_ID.items()]

    coco_dict = {"images": images_dict, "categories": categories_dict, "annotations": annotations}

    JSON_TEST_FILE_PATH.write_text(json.dumps(coco_dict))


def get_annotation(index: int, image_id: str, token: PdfToken):
    return {
        "area": 1,
        "iscrowd": 0,
        "score": 1,
        "image_id": image_id,
        "bbox": [token.bounding_box.left, token.bounding_box.top, token.bounding_box.width, token.bounding_box.height],
        "category_id": token.token_type.get_index(),
        "id": index,
    }


def get_annotations_for_document(annotations, images, index, pdf_images, width_height):
    for page_index, page in enumerate(pdf_images.pdf_features.pages):
        image_id = f"{pdf_images.pdf_features.file_name}_{page.page_number - 1}"
        images.append(image_id)
        width_height.append((pdf_images.pdf_images[page_index].width, pdf_images.pdf_images[page_index].height))

        for token in page.tokens:
            annotations.append(get_annotation(index, image_id, token))
            index += 1


def get_annotations(pdf_images_list: list[PdfImages]):
    makedirs(JSONS_ROOT_PATH, exist_ok=True)

    annotations = list()
    images = list()
    width_height = list()
    index = 0

    for pdf_images in pdf_images_list:
        get_annotations_for_document(annotations, images, index, pdf_images, width_height)
        index += sum([len(page.tokens) for page in pdf_images.pdf_features.pages])

    save_annotations_json(annotations, width_height, images)
