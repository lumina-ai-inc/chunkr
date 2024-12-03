#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""

import os
import itertools

import torch

from typing import Any, Dict, List, Set

from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping

from ditod import add_vit_config
from ditod import DetrDatasetMapper

from detectron2.data.datasets import register_coco_instances
import logging
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.engine.defaults import create_ddp_model
import weakref
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from ditod import VGTTrainer as MyTrainer
import numpy as np

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    """
    register publaynet first
    """
    # add docbank data
    register_coco_instances(
        "docbank_train",
        {},
        "/DocBank_root_path/DocBank/500K_train_VGT.json",
        "/DocBank_root_path/DocBank/DocBank_500K_ori_img"
    )

    register_coco_instances(
        "docbank_val",
        {},
        "/DocBank_root_path/DocBank/500K_valid_VGT.json",
        "/DocBank_root_path/DocBank/DocBank_500K_ori_img"
    )
    
    
    # add publaynet data
    register_coco_instances(
        "publaynet_train",
        {},
        "/publaynet_root_path/publaynet/train.json",
        "/publaynet_root_path/publaynet/train"
    )

    register_coco_instances(
        "publaynet_val",
        {},
        "/publaynet_root_path/publaynet/val.json",
        "/publaynet_root_path/publaynet/val"
    )
    

    # add D4LA data
    register_coco_instances(
        "D4LA_train",
        {},
        "/D4LA_root_path/D4LA/json/train.json",
        "/D4LA_root_path/D4LA/train_images"
    )

    register_coco_instances(
        "D4LA_val",
        {},
        "/D4LA_root_path/D4LA/json/test.json",
        "/D4LA_root_path/D4LA/test_images"
    )

    
    # add doclaynet data
    register_coco_instances(
        "doclayent_train",
        {},
        "/doclayent_root_path/DocLayNet/COCO/train.json",
        "/doclayent_root_path/DocLayNet/PNG"
    )

    register_coco_instances(
        "doclayent_val",
        {},
        "/doclayent_root_path/DocLayNet/COCO/val.json",
        "/doclayent_root_path/DocLayNet/PNG"
    )
    

    cfg = setup(args)
    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res
    
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    print("Command Line Args:", args)
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"

    if args.debug:
        import debugpy

        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
