from detectron2.config import CfgNode as CN


def add_vit_config(cfg):
    """
    Add config for VIT.
    """
    _C = cfg

    _C.MODEL.VIT = CN()

    # CoaT model name.
    _C.MODEL.VIT.NAME = ""

    # Output features from CoaT backbone.
    _C.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]

    _C.MODEL.VIT.IMG_SIZE = [224, 224]

    _C.MODEL.VIT.POS_TYPE = "shared_rel"
    
    _C.MODEL.VIT.MERGE_TYPE = "Sum"

    _C.MODEL.VIT.DROP_PATH = 0.

    _C.MODEL.VIT.MODEL_KWARGS = "{}"

    _C.SOLVER.OPTIMIZER = "ADAMW"

    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0

    _C.AUG = CN()

    _C.AUG.DETR = False
    
    _C.MODEL.WORDGRID = CN()
    
    _C.MODEL.WORDGRID.VOCAB_SIZE = 30552
    
    _C.MODEL.WORDGRID.EMBEDDING_DIM = 64 
    
    _C.MODEL.WORDGRID.MODEL_PATH = ''
    
    _C.MODEL.WORDGRID.HIDDEN_SIZE = 768
    
    _C.MODEL.WORDGRID.USE_PRETRAIN_WEIGHT = True
    
    _C.MODEL.WORDGRID.USE_UNK_TEXT = False
