# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from https://github.com/facebookresearch/detr/blob/main/d2/detr/dataset_mapper.py


import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import json
import pickle

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

__all__ = ["DetrDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    # if is_train:
    #     tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def build_transform_gen_w(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = 800
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    # if is_train:
    #     tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        self.tfm_gens_w = build_transform_gen_w(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        self.cfg = cfg

        logger = logging.getLogger("detectron2")
            
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        
        try:
            name = dataset_dict["file_name"][0:-4].split('/') 
            if 'publaynet' in name:
                root = '/'.join(name[:-2])
                if name[-2] == 'val':
                    name[-2] = 'dev'
                pdf_name = '/'.join(['/VGT_publaynet_grid_pkl'] + name[-2:])
                with open((root + pdf_name + '.pdf.pkl'), "rb") as f:
                    sample_inputs = pickle.load(f)
                input_ids = sample_inputs["input_ids"]
                bbox_subword_list = sample_inputs["bbox_subword_list"]
            elif 'DocBank' in name:
                root = '/'.join(name[:-2])
                pdf_name = '/'.join(['/VGT_docbank_grid_pkl'] + name[-1:])
                with open((root + pdf_name + '.pkl'), "rb") as f:
                    sample_inputs = pickle.load(f)
                input_ids = sample_inputs["input_ids"]
                bbox_subword_list = sample_inputs["bbox_subword_list"]
            elif 'D4LA' in name:
                root = '/'.join(name[:-2])
                pdf_name = '/'.join(['/VGT_D4LA_grid_pkl'] + name[-1:])
                with open((root + pdf_name + '.pkl'), "rb") as f:
                    sample_inputs = pickle.load(f)
                input_ids = sample_inputs["input_ids"]
                bbox_subword_list = sample_inputs["bbox_subword_list"]
            elif 'DocLayNet' in name:
                root = '/'.join(name[:-2])
                pdf_name = '/'.join(['/VGT_DocLayNet_grid_pkl'] + name[-1:])
                with open((root + pdf_name + '.pdf.pkl'), "rb") as f:
                    sample_inputs = pickle.load(f)
                input_ids = sample_inputs["input_ids"]
                bbox_subword_list = sample_inputs["bbox_subword_list"]
            else:
                input_ids = []
                bbox_subword_list = []
                print("no grid pkl")
        except:
            print("Wrong bbox file:", dataset_dict["file_name"])
            input_ids = []
            bbox_subword_list = []

        image_shape_ori = image.shape[:2]  # h, w

        if self.crop_gen is None:
            if image_shape_ori[0] > image_shape_ori[1]:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(self.tfm_gens_w, image)
        else:
            if np.random.rand() > 0.5:
                if image_shape_ori[0] > image_shape_ori[1]:
                    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                else:
                    image, transforms = T.apply_transform_gens(self.tfm_gens_w, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens_w[:-1] + self.crop_gen + self.tfm_gens_w[-1:], image
                )

        image_shape = image.shape[:2]  # h, w
        
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        ## 产出 text grid 
        bbox = []
        for bbox_per_subword in bbox_subword_list:
            text_word = {}
            text_word['bbox'] = bbox_per_subword.tolist()
            text_word['bbox_mode'] = BoxMode.XYWH_ABS
            utils.transform_instance_annotations(text_word, transforms, image_shape)
            bbox.append(text_word['bbox'])
                
        dataset_dict["input_ids"] = input_ids 
        dataset_dict["bbox"] = bbox

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)           

        return dataset_dict