import torch
import cv2
import numpy as np
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as utils
from detectron2.data.detection_utils import transform_instance_annotations
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
from transformers import AutoTokenizer
from ditod import add_vit_config
from create_grid_input import create_grid_dict

def create_grid_dict_from_image(tokenizer, image):
    """Create grid dictionary for a single image without PDF processing"""
    # For now, we'll create an empty grid since we don't have OCR
    # In production, you might want to add OCR here
    grid = {
        "input_ids": np.array([], dtype=np.int64),
        "bbox_subword_list": np.array([], dtype=np.float32),
        "texts": [],
        "bbox_texts_list": np.array([], dtype=np.float32)
    }
    return grid

class MemoryPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], 
            cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict): the output of the model
        """
        with torch.no_grad():
            # Get image dimensions
            height, width = original_image.shape[:2]
            
            # Apply transforms
            image, transforms = T.apply_transform_gens([self.aug], original_image)
            image_shape = image.shape[:2]  # h, w
            
            # Convert to tensor
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            # Create grid dictionary
            grid_dict = create_grid_dict_from_image(self.tokenizer, original_image)
            
            # Transform bbox coordinates if they exist
            bbox = []
            if len(grid_dict["bbox_subword_list"]) > 0:
                for bbox_per_subword in grid_dict["bbox_subword_list"]:
                    text_word = {
                        'bbox': bbox_per_subword.tolist(),
                        'bbox_mode': BoxMode.XYWH_ABS
                    }
                    utils.transform_instance_annotations(text_word, transforms, image_shape)
                    bbox.append(text_word['bbox'])

            # Prepare input dictionary
            dataset_dict = {
                "input_ids": grid_dict["input_ids"],
                "bbox": bbox,
                "image": image,
                "height": height,
                "width": width
            }

            # Run inference
            predictions = self.model([dataset_dict])[0]
            print(predictions)
            return predictions
def setup_cfg(config_file, weights_path, opts=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_file)
    
    if opts:
        cfg.merge_from_list(opts)
    
    # Set weights path
    cfg.MODEL.WEIGHTS = weights_path
    
    # Set device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg.freeze()
    return cfg

def create_predictor(config_file, weights_path, opts=None):
    """
    Creates a MemoryPredictor instance with the given config and weights.
    """
    cfg = setup_cfg(config_file, weights_path, opts)
    return MemoryPredictor(cfg)

def process_image(predictor, image, dataset_name="doclaynet"):
    """
    Process an image using the predictor and return both predictions and visualization.
    
    Args:
        predictor (MemoryPredictor): Initialized predictor
        image (np.ndarray): Input image in BGR format
        dataset_name (str): Name of the dataset to use for metadata

    Returns:
        tuple: (predictions, visualization_image)
    """
    # Get predictions
    predictions = predictor(image)

    # Set up metadata based on dataset
    md = MetadataCatalog.get(predictor.cfg.DATASETS.TEST[0])
    if dataset_name == 'publaynet':
        md.set(thing_classes=["text","title","list","table","figure"])
    elif dataset_name == 'docbank':
        md.set(thing_classes=["abstract","author","caption","date","equation", "figure", "footer", "list", "paragraph", "reference", "section", "table", "title"])
    elif dataset_name == 'D4LA':
        md.set(thing_classes=["DocTitle","ParaTitle","ParaText","ListText","RegionTitle", "Date", "LetterHead", "LetterDear", "LetterSign", "Question", "OtherText", "RegionKV", "Regionlist", "Abstract", "Author", "TableName", "Table", "Figure", "FigureName", "Equation", "Reference", "Footnote", "PageHeader", "PageFooter", "Number", "Catalog", "PageNumber"])
    elif dataset_name == 'doclaynet':
        md.set(thing_classes=["Caption","Footnote","Formula","List-item","Page-footer", "Page-header", "Picture", "Section-header", "Table", "Text", "Title"])

    # Create visualization
    v = Visualizer(
        image[:, :, ::-1],
        metadata=md,
        scale=1.0,
        instance_mode=ColorMode.SEGMENTATION
    )
    result = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    visualization = result.get_image()[:, :, ::-1]

    return predictions, visualization