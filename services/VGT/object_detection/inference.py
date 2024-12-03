import argparse

import cv2

from ditod import add_vit_config

import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from ditod.VGTTrainer import DefaultPredictor


def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--image_root",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--grid_root",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--image_name",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_root",
        help="Name of the output visualization file.",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    
    if args.dataset in ('D4LA', 'doclaynet'):
        image_path = args.image_root + args.image_name + ".png"
    else:
        image_path = args.image_root + args.image_name + ".jpg"
    
    if args.dataset == 'publaynet':
        grid_path = args.grid_root + args.image_name + ".pdf.pkl"
    elif args.dataset == 'docbank':
        grid_path = args.grid_root + args.image_name + ".pkl"
    elif args.dataset == 'D4LA':
        grid_path = args.grid_root + args.image_name + ".pkl"
    elif args.dataset == 'doclaynet':
        grid_path = args.grid_root + args.image_name + ".pdf.pkl"
        
    output_file_name = args.output_root + args.image_name + ".jpg"
    
    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)
    
    # Step 5: run inference
    img = cv2.imread(image_path)
    
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if args.dataset == 'publaynet':
        md.set(thing_classes=["text","title","list","table","figure"])
    elif args.dataset == 'docbank':
        md.set(thing_classes=["abstract","author","caption","date","equation", "figure", "footer", "list", "paragraph", "reference", "section", "table", "title"])
    elif args.dataset == 'D4LA':
        md.set(thing_classes=["DocTitle","ParaTitle","ParaText","ListText","RegionTitle", "Date", "LetterHead", "LetterDear", "LetterSign", "Question", "OtherText", "RegionKV", "Regionlist", "Abstract", "Author", "TableName", "Table", "Figure", "FigureName", "Equation", "Reference", "Footnote", "PageHeader", "PageFooter", "Number", "Catalog", "PageNumber"])
    elif args.dataset == 'doclaynet':
        md.set(thing_classes=["Caption","Footnote","Formula","List-item","Page-footer", "Page-header", "Picture", "Section-header", "Table", "Text", "Title"])

    output = predictor(img, grid_path)["instances"]
    
    # import ipdb;ipdb.set_trace()
    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    # step 6: save
    cv2.imwrite(output_file_name, result_image)

if __name__ == '__main__':
    main()

