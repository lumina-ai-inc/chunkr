import torch
import torch.cuda.amp
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
from config import add_vit_config

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
                    height, width = original_image.shape[:2]
                    
                    image, transforms = T.apply_transform_gens([self.aug], original_image)
                    image_shape = image.shape[:2]  # h, w
                    
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

                    bbox = []
                    if len(grid_dict["bbox_subword_list"]) > 0:
                        for bbox_per_subword in grid_dict["bbox_subword_list"]:
                            text_word = {
                                'bbox': bbox_per_subword,
                                'bbox_mode': BoxMode.XYWH_ABS
                            }
                            utils.transform_instance_annotations(text_word, transforms, image_shape)
                            bbox.append(text_word['bbox'])

                    dataset_dict = {
                        "input_ids": grid_dict["input_ids"],
                        "bbox": bbox,
                        "image": image,
                        "height": height,
                        "width": width
                    }
                    batch_inputs.append(dataset_dict)

                predictions = self.model(batch_inputs)

                return predictions

def process_image_batch(predictor, images, dataset_name="doclaynet", grid_dicts=None):
    """
    Process a batch of images using the predictor and return predictions.
    """
    predictions = predictor(images, grid_dicts)

    md = MetadataCatalog.get(predictor.cfg.DATASETS.TEST[0])
    if dataset_name == 'doclaynet':
        md.set(thing_classes=["Caption","Footnote","Formula","ListItem","PageFooter", 
                            "PageHeader", "Picture", "SectionHeader", "Table", "Text", "Title"])
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
    
    cfg.MODEL.WEIGHTS = weights_path
    
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg.freeze()
    return cfg

def create_predictor(config_file, weights_path, opts=None):
    """
    Creates a MemoryPredictor instance with the given config and weights.
    """
    cfg = setup_cfg(config_file, weights_path, opts)
    return MemoryPredictor(cfg)

