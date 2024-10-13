import math
import torch
from torch import nn, Tensor
from functools import partial

from src.model.components import ImgLinearBackbone, PositionEmbedding, Encoder


class BeitEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,  # embed_dim
        backbone: nn.Module,
        max_seq_len: int,  # for positional embedding
        codebook_tokens: int,
        dropout: float,
        encoder: Encoder,
        norm_layer: nn.Module,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.init_std = init_std

        self.backbone = backbone
        self.pos_embed = PositionEmbedding(
            max_seq_len=max_seq_len, d_model=d_model, dropout=dropout
        )

        self.encoder = encoder
        self.norm = norm_layer(d_model)
        self.generator = nn.Linear(d_model, codebook_tokens)

        self.trunc_normal = partial(
            nn.init.trunc_normal_, std=init_std, a=-init_std, b=init_std
        )
        self.apply(self._init_weights)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            self.trunc_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            self.trunc_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, PositionEmbedding):
            self.trunc_normal(m.embedding.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(
        self, x: Tensor, bool_masked_pos: Tensor, return_all_tokens: bool = False
    ):
        x = self.backbone(x)
        B, S, E = x.shape
        assert E == self.d_model

        mask_token = self.mask_token.expand(B, S, -1)

        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = self.pos_embed(x)

        x = self.encoder(x)
        x = self.norm(x)

        if return_all_tokens:
            return self.generator(x)
        else:
            return self.generator(x[bool_masked_pos])


if __name__ == "__main__":
    d_model = 512
    patch_size = 16
    nhead = 8
    dropout = 0.0
    acitvation = "gelu"
    norm_first = True
    nlayer = 12
    ff_ratio = 4
    norm_layer = partial(nn.LayerNorm, eps=1e-6)
    codebook_tokens = 8192

    img_size = 448

    max_seq_len = (img_size // patch_size) ** 2

    backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
    encoder = Encoder(
        d_model=d_model,
        nhead=nhead,
        dropout=dropout,
        activation=acitvation,
        norm_first=norm_first,
        nlayer=nlayer,
        ff_ratio=ff_ratio,
    )

    model = BeitEncoder(
        d_model=d_model,
        backbone=backbone,
        max_seq_len=max_seq_len,
        codebook_tokens=codebook_tokens,
        dropout=dropout,
        encoder=encoder,
        norm_layer=norm_layer,
    )

    print(model)

    x = torch.rand((1, 3, img_size, img_size))
    bool_masked_pos = torch.rand((1, (img_size // patch_size) ** 2)) < 0.5
    y = model(x, bool_masked_pos)
    print(torch.sum(bool_masked_pos))
    print(y.shape)
