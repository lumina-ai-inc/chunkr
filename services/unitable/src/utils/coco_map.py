import torch
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint


def compute_coco_map(file):
    coco_pred = list()
    coco_gt = list()
    for _, obj in file.items():
        tmp_pred = {
            "boxes": torch.tensor(obj["pred"], device=0),
            "labels": torch.tensor([0] * len(obj["pred"]), device=0),
            "scores": torch.tensor([0.999] * len(obj["pred"]), device=0),
        }

        tmp_gt = {
            "boxes": torch.tensor(obj["gt"], device=0),
            "labels": torch.tensor([0] * len(obj["gt"]), device=0),
        }

        coco_pred.append(tmp_pred)
        coco_gt.append(tmp_gt)

    metric = MeanAveragePrecision(
        iou_type="bbox",
        max_detection_thresholds=[1, 10, 1000],
        backend="faster_coco_eval",
    )
    metric.update(coco_pred, coco_gt)
    pprint(metric.compute())


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="mAP Computation")

    parser.add_argument("-f", "--file", help="path to html table results in json file")
    args = parser.parse_args()


    results_file = args.file
    with open(results_file, "r") as f:
        results_json = json.load(f)

    compute_coco_map(results_json)
