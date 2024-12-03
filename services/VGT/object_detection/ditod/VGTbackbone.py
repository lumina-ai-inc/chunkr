# --------------------------------------------------------------------------------
# VIT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# CoaT: https://github.com/mlpc-ucsd/CoaT
# --------------------------------------------------------------------------------


import torch
import torch.nn.functional as F
import logging

from detectron2.layers import (
    ShapeSpec,
)
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone.fpn import LastLevelP6P7, LastLevelMaxPool

from .VGTbeit import beit_base_patch16, dit_base_patch16, dit_large_patch16, beit_large_patch16, VGT_dit_base_patch16
from .FeatureMerge import FeatureMerge

__all__ = [
    "build_VGT_fpn_backbone",
]


class PTM_VIT_Backbone(Backbone):
    """
    Implement VIT backbone.
    """

    def __init__(self, name, out_features, drop_path, img_size, pos_type, merge_type, model_kwargs):
        super().__init__()
        self._out_features = out_features
        if 'base' in name:
            self._out_feature_strides = {"layer3": 4, "layer5": 8, "layer7": 16, "layer11": 32}
        else:
            self._out_feature_strides = {"layer7": 4, "layer11": 8, "layer15": 16, "layer23": 32}

        if name == 'beit_base_patch16':
            model_func = beit_base_patch16
            self._out_feature_channels = {"layer3": 768, "layer5": 768, "layer7": 768, "layer11": 768}
        elif name == 'dit_base_patch16':
            model_func = dit_base_patch16
            self._out_feature_channels = {"layer3": 768, "layer5": 768, "layer7": 768, "layer11": 768}
        elif name == "deit_base_patch16":
            model_func = deit_base_patch16
            self._out_feature_channels = {"layer3": 768, "layer5": 768, "layer7": 768, "layer11": 768}
        elif name == 'VGT_dit_base_patch16':
            model_func = VGT_dit_base_patch16
            self._out_feature_channels = {"layer3": 768, "layer5": 768, "layer7": 768, "layer11": 768}
        elif name == "mae_base_patch16":
            model_func = mae_base_patch16
            self._out_feature_channels = {"layer3": 768, "layer5": 768, "layer7": 768, "layer11": 768}
        elif name == "dit_large_patch16":
            model_func = dit_large_patch16
            self._out_feature_channels = {"layer7": 1024, "layer11": 1024, "layer15": 1024, "layer23": 1024}
        elif name == "beit_large_patch16":
            model_func = beit_large_patch16
            self._out_feature_channels = {"layer7": 1024, "layer11": 1024, "layer15": 1024, "layer23": 1024}
        else:
            raise ValueError("Unsupported VIT name yet.")

        if 'beit' in name or 'dit' in name:
            if pos_type == "abs":
                self.backbone = model_func(img_size=img_size,
                                           out_features=out_features,
                                           drop_path_rate=drop_path,
                                           use_abs_pos_emb=True,
                                           **model_kwargs)
            elif pos_type == "shared_rel":
                self.backbone = model_func(img_size=img_size,
                                           out_features=out_features,
                                           drop_path_rate=drop_path,
                                           use_shared_rel_pos_bias=True,
                                           **model_kwargs)
            elif pos_type == "rel":
                self.backbone = model_func(img_size=img_size,
                                           out_features=out_features,
                                           drop_path_rate=drop_path,
                                           use_rel_pos_bias=True,
                                           **model_kwargs)
            else:
                raise ValueError()
        else:
            self.backbone = model_func(img_size=img_size,
                                       out_features=out_features,
                                       drop_path_rate=drop_path,
                                       **model_kwargs)
        
        logger = logging.getLogger("detectron2")
        logger.info("Merge using: {}".format(merge_type))
        self.FeatureMerge = FeatureMerge(feature_names = self._out_features, visual_dim=[768,768,768,768], semantic_dim=[768,768,768,768], merge_type = merge_type)

    def forward(self, x, grid):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"VIT takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        
        vis_feat_out, grid_feat_out = self.backbone.forward_features(x, grid)
        return self.FeatureMerge.forward(vis_feat_out, grid_feat_out)
        # return self.backbone.forward_features(x)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class GridFPN(FPN):
    def forward(self, x, grid):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x, grid)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}
    
def build_PTM_VIT_Backbone(cfg):
    """
    Create a VIT instance from config.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        A VIT backbone instance.
    """
    # fmt: off
    name = cfg.MODEL.VIT.NAME
    out_features = cfg.MODEL.VIT.OUT_FEATURES
    drop_path = cfg.MODEL.VIT.DROP_PATH
    img_size = cfg.MODEL.VIT.IMG_SIZE
    pos_type = cfg.MODEL.VIT.POS_TYPE
    merge_type = cfg.MODEL.VIT.MERGE_TYPE
    
    model_kwargs = eval(str(cfg.MODEL.VIT.MODEL_KWARGS).replace("`", ""))

    return PTM_VIT_Backbone(name, out_features, drop_path, img_size, pos_type, merge_type, model_kwargs)


@BACKBONE_REGISTRY.register()
def build_VGT_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a VIT w/ FPN backbone.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_PTM_VIT_Backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = GridFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
