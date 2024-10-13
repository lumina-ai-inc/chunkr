import torch
from torch import Tensor, nn
from functools import partial

from src.model.components import (
    ImgCnnBackbone,
    ImgLinearBackbone,
    ImgConvStemBackbone,
    Encoder,
    Decoder,
    PositionEmbedding,
    TokenEmbedding,
)


class EncoderDecoder(nn.Module):
    """Encoder decoder architecture that takes in a tabular image and generates the text output.
    Backbone serves as the image processor. There are three types of backbones: CNN, linear projection, and ConvStem.

    Args:
    ----
        backbone: tabular image processor
        encoder: transformer encoder
        decoder: transformer decoder
        vocab_size: size of the vocabulary
        d_model: feature size
        padding_idx: index of <pad> in the vocabulary
        max_seq_len: max sequence length of generated text
        dropout: dropout rate
        norm_layer: layernorm
        init_std: std in weights initialization
    """

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
        max_seq_len: int,
        dropout: float,
        norm_layer: nn.Module,
        init_std: float = 0.02,
    ):
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.norm = norm_layer(d_model)
        self.token_embed = TokenEmbedding(
            vocab_size=vocab_size, d_model=d_model, padding_idx=padding_idx
        )
        self.pos_embed = PositionEmbedding(
            max_seq_len=max_seq_len, d_model=d_model, dropout=dropout
        )
        self.generator = nn.Linear(d_model, vocab_size)

        self.trunc_normal = partial(
            nn.init.trunc_normal_, std=init_std, a=-init_std, b=init_std
        )
        self.apply(self._init_weights)

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
        elif isinstance(m, TokenEmbedding):
            self.trunc_normal(m.embedding.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"token_embed", "pos_embed"}

    def encode(self, src: Tensor) -> Tensor:
        src_feature = self.backbone(src)
        src_feature = self.pos_embed(src_feature)
        memory = self.encoder(src_feature)
        memory = self.norm(memory)
        return memory

    def decode(
        self, memory: Tensor, tgt: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        tgt_feature = self.pos_embed(self.token_embed(tgt))
        tgt = self.decoder(tgt_feature, memory, tgt_mask, tgt_padding_mask)

        return tgt

    def forward(
        self, src: Tensor, tgt: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        memory = self.encode(src)
        tgt = self.decode(memory, tgt, tgt_mask, tgt_padding_mask)
        tgt = self.generator(tgt)

        return tgt
