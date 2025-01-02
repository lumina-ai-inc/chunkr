import torch
from torch import nn

class FeatureMerge(nn.Module):
    """Multimodal feature fusion used in VSR."""
    def __init__(self,
                 feature_names,
                 visual_dim,
                 semantic_dim,
                 merge_type='Sum',
                 dropout_ratio=0.1,
                 with_extra_fc=True,
                 shortcut=False
                 ):
        """Multimodal feature merge used in VSR.
        Args:
            visual_dim (list): the dim of visual features, e.g. [256]
            semantic_dim (list): the dim of semantic features, e.g. [256]
            merge_type (str): fusion type, e.g. 'Sum', 'Concat', 'Weighted'
            dropout_ratio (float): dropout ratio of fusion features
            with_extra_fc (bool): whether add extra fc layers for adaptation
            shortcut (bool): whether add shortcut connection
        """
        super().__init__()

        # merge param
        self.feature_names = feature_names
        self.merge_type = merge_type
        self.visual_dim = visual_dim
        self.textual_dim = semantic_dim
        self.with_extra_fc = with_extra_fc
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)
        
        if self.merge_type == 'Sum':
            assert len(self.visual_dim) == len(self.textual_dim)
        elif self.merge_type == 'Concat':
            assert len(self.visual_dim) == len(self.textual_dim)
            # self.concat_proj = nn.ModuleList()
            
            self.vis_proj = nn.ModuleList()
            self.text_proj = nn.ModuleList()
            self.alpha_proj = nn.ModuleList()
            
            for idx in range(len(self.visual_dim)):
                # self.concat_proj.append(nn.Conv2d(self.visual_dim[idx] + self.textual_dim[idx], self.visual_dim[idx], kernel_size = (1,1), stride=1))
                if self.with_extra_fc:
                    self.vis_proj.append(nn.Linear(self.visual_dim[idx], self.visual_dim[idx]))
                    self.text_proj.append(nn.Linear(self.textual_dim[idx], self.textual_dim[idx]))
                self.alpha_proj.append(nn.Linear(self.visual_dim[idx] + self.textual_dim[idx], self.visual_dim[idx]))
            
        elif self.merge_type == 'Weighted':
            assert len(self.visual_dim) == len(self.textual_dim)
            self.total_num = len(self.visual_dim)

            # vis projection
            self.vis_proj = nn.ModuleList()
            self.vis_proj_relu = nn.ModuleList()

            # text projection
            self.text_proj = nn.ModuleList()
            self.text_proj_relu = nn.ModuleList()

            self.alpha_proj = nn.ModuleList()
            for idx in range(self.total_num):
                if self.with_extra_fc:
                    self.vis_proj.append(nn.Linear(self.visual_dim[idx], self.visual_dim[idx]))
                    self.text_proj.append(nn.Linear(self.textual_dim[idx], self.textual_dim[idx]))
                self.alpha_proj.append(nn.Linear(self.visual_dim[idx] + self.textual_dim[idx], self.visual_dim[idx]))

        else:
            raise "Unknown merge type {}".format(self.merge_type)

        self.dropout = nn.Dropout(dropout_ratio)

        # visual context
        # self.visual_ap = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, visual_feat=None, textual_feat=None):
        """ Forward computation
        Args:
            visual_feat (list(Tensor)): visual feature maps, in shape of [L x C x H x W] x B
            textual_feat (Tensor): textual feature maps, in shape of B x L x C
        Returns:
            Tensor: fused feature maps, in shape of [B x L x C]
        """
        assert len(visual_feat) == len(textual_feat)

        # feature merge
        merged_feat = {}
        if self.merge_type == 'Sum':
            for name in self.feature_names:
                merged_feat[name] = visual_feat[name] + textual_feat[name]
        elif self.merge_type == 'Concat':
            for idx, name in enumerate(self.feature_names):
                # merged_feat[name] = self.concat_proj[idx](torch.cat((visual_feat[name],textual_feat[name]),1))
                per_vis = visual_feat[name].permute(0, 2, 3, 1)
                per_text = textual_feat[name].permute(0, 2, 3, 1)
                if self.with_extra_fc:
                    per_vis = self.relu(self.vis_proj[idx](per_vis))
                    per_text = self.relu(self.text_proj[idx](per_text))
                x_sentence = self.alpha_proj[idx](torch.cat((per_vis, per_text), -1))
                x_sentence = x_sentence.permute(0,3,1,2).contiguous()
                merged_feat[name] = x_sentence
        else:
            assert self.total_num == len(visual_feat) or self.total_num == 1
            # for per_vis, per_text in zip(visual_feat, textual_feat):
            for idx, name in enumerate(self.feature_names):
                per_vis = visual_feat[name].permute(0, 2, 3, 1)
                per_text = textual_feat[name].permute(0, 2, 3, 1)
                if self.with_extra_fc:
                    per_vis = self.relu(self.vis_proj[idx](per_vis))
                    per_text = self.relu(self.text_proj[idx](per_text))

                alpha = torch.sigmoid(self.alpha_proj[idx](torch.cat((per_vis, per_text), -1)))
                if self.shortcut:
                    # shortcut
                    x_sentence = per_vis + alpha * per_text
                else:
                    # selection
                    x_sentence = alpha * per_vis + (1 - alpha) * per_text

                x_sentence = x_sentence.permute(0,3,1,2).contiguous()
                merged_feat[name] = x_sentence

        return merged_feat