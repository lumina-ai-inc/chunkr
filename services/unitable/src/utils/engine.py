import os
import json
import argparse
from pathlib import Path
import glob

from src.utils import build_table_from_html_and_cell, html_table_template


def combine_all_json(file_dir: str) -> dict:
    total_result = dict()
    files = os.listdir(file_dir)
    try:
        files.remove("final.json")
    except ValueError:
        pass
    for file in files:
        with open(os.path.join(file_dir, file), "r") as f:
            result = json.load(f)
            total_result.update(result)

    print(f"Combined to a json with {len(total_result)} entries.")

    return total_result


def json_to_final(file_dir: str, type: str):
    if type == "html" or type == "bbox":
        result = combine_all_json(file_dir)
    elif type == "html+cell":
        result_cell = combine_all_json(file_dir)
        result_html_file = os.path.join(
            Path(file_dir).parent,
            Path(file_dir).name.split("-")[0].replace("cell", "html") + "-html",
        )
        assert Path(result_html_file).is_dir(), f"{result_html_file} does not exist."
        result = combine_all_json(result_html_file)
        assert len(result) == len(result_cell)
    else:
        # assert html and cell json files have the same length
        raise NotImplementedError

    out = dict()

    if type == "bbox":
        out = result
    else:
        for filename, obj in result.items():
            if type == "html":
                pred_html = "".join(obj["pred"])
                gt_html = "".join(obj["gt"])

                out[filename] = dict(
                    pred=html_table_template(pred_html), gt=html_table_template(gt_html)
                )
            elif type == "html+cell":
                pred_html_cell = build_table_from_html_and_cell(
                    obj["pred"], result_cell[filename]["pred"]
                )
                gt_html_cell = build_table_from_html_and_cell(
                    obj["gt"], result_cell[filename]["gt"]
                )
                out[filename] = dict(
                    pred=html_table_template(pred_html_cell),
                    gt=html_table_template(gt_html_cell),
                )
            else:
                raise NotImplementedError

    # write to file
    with open(os.path.join(file_dir, f"final.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="postprecess")

    parser.add_argument(
        "-f", "--file", help="path to all json files from difference devices"
    )
    parser.add_argument("-t", "--type", help="html, html+cell")
    args = parser.parse_args()

    json_to_final(args.file, args.type)
