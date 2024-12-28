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
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
from transformers import AutoTokenizer
from ditod import add_vit_config

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

    def __call__(self, original_images, grid_dicts):
            """
            Args:
                original_images (list[np.ndarray]): list of images, each of shape (H, W, C) (in BGR order)
                grid_dicts (list[dict]): list of grid dictionaries, one for each image

            Returns:
                predictions (list[dict]): the output of the model for each image
            """
            with torch.no_grad():
                batch_inputs = []
                
                for original_image, grid_dict in zip(original_images, grid_dicts):
                    # Get image dimensions
                    height, width = original_image.shape[:2]
                    
                    # Apply transforms
                    image, transforms = T.apply_transform_gens([self.aug], original_image)
                    image_shape = image.shape[:2]  # h, w
                    
                    # Convert to tensor
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

                    # Transform bbox coordinates if they exist
                    bbox = []
                    if len(grid_dict["bbox_subword_list"]) > 0:
                        for bbox_per_subword in grid_dict["bbox_subword_list"]:
                            text_word = {
                                'bbox': bbox_per_subword,
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
                    batch_inputs.append(dataset_dict)

                # Run inference on batch
                predictions = self.model(batch_inputs)
                return predictions

def process_image_batch(predictor, images, dataset_name="doclaynet", grid_dicts=None):
    """
    Process a batch of images using the predictor and return predictions and visualizations.
    
    Args:
        predictor (MemoryPredictor): Initialized predictor
        images (list[np.ndarray]): List of input images in BGR format
        dataset_name (str): Name of the dataset to use for metadata
        grid_dicts (list[dict]): List of grid dictionaries, one for each image

    Returns:
        tuple: (predictions, visualization_images)
    """
    # Get predictions for batch
    predictions = predictor(images, grid_dicts)

    # Set up metadata based on dataset
    md = MetadataCatalog.get(predictor.cfg.DATASETS.TEST[0])
    if dataset_name == 'doclaynet':
        md.set(thing_classes=["Caption","Footnote","Formula","List-item","Page-footer", 
                            "Page-header", "Picture", "Section-header", "Table", "Text", "Title"])
    # ... other dataset configurations ...

    # Create visualizations for each image
    visualizations = []
    # for image, pred in zip(images, predictions):
    #     v = Visualizer(
    #         image[:, :, ::-1],
    #         metadata=md,
    #         scale=1.0,
    #         instance_mode=ColorMode.SEGMENTATION
    #     )
    #     result = v.draw_instance_predictions(pred["instances"].to("cpu"))
    #     visualization = result.get_image()[:, :, ::-1]
    #     visualizations.append(visualization)

    return predictions, visualizations
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

def process_image(predictor, image, dataset_name="doclaynet", grid_dict=None):
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
    predictions = predictor(image, grid_dict)

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