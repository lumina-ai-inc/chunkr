from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation


__all__ = [
    "ImgCnnBackbone",
    "ImgLinearBackbone",
    "ImgConvStemBackbone",
    "PositionEmbedding",
    "Encoder",
    "Decoder",
    "TokenEmbedding",
]


class ImgCnnBackbone(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        output_channels: int,
        d_model: int,
        drop_layer: Tuple = None,
    ) -> None:
        super().__init__()

        # drop layers for classification & maxpooling for higher feature resolution
        layers = list(backbone.children())
        nlayer = len(layers)
        keep_layer = set([i for i in range(nlayer)]) - set(drop_layer)
        backbone = [layers[i] for i in keep_layer]
        self.backbone = nn.Sequential(*backbone)
        self.proj = nn.Linear(output_channels, d_model)
        self.channels = output_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        assert x.shape[-1] == self.channels, "Image channels size mismatch."
        x = self.proj(x)
        return x


class ImgLinearBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
        in_chan: int = 3,
    ) -> None:
        super().__init__()

        self.conv_proj = nn.Conv2d(
            in_chan, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_proj(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        return x


class ImgConvStemBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        downsample_factor: int,
        output_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        assert downsample_factor % 2 == 0
        assert output_channels % (downsample_factor // 2) == 0
        input_channels = output_channels // (downsample_factor // 2)

        layers = [
            Conv2dNormActivation(
                3, input_channels, kernel_size=kernel_size, stride=2, padding=1
            )
        ]

        while input_channels != output_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    input_channels * 2,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                )
            )
            input_channels = input_channels * 2

        layers.append(nn.Conv2d(output_channels, d_model, kernel_size=1))

        self.conv_stem = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_stem(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        nlayer: int,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=nhead,
            dim_feedforward=ff_ratio * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        nlayer: int,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=ff_ratio * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, nlayer)

    def forward(
        self, x: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        x = self.decoder(
            x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask
        )
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # assume x is batch first
        out = self.embedding(torch.arange(x.shape[1], device=x.device))
        return self.dropout(out + x)


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        assert vocab_size > 0
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)


class PrintLayer(nn.Module):
    """Only for debugging when loss is nan."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(
            "torch.isfinite(x).all(): {}, min. {:.5f}, max. {:.5f}".format(
                torch.isfinite(x).all(), x.min(), x.max()
            )
        )
        return x


if __name__ == "__main__":
    from torchvision import models

    x = torch.rand(1, 3, 392, 392)
    model = ImgConvStemBackbone(
        d_model=512, downsample_factor=16, output_channels=64, kernel_size=5
    )
    y = model(x)
    print(model)
    print(y.shape)

    model = ImgCnnBackbone(
        backbone=models.resnet34(),
        output_channels=512,
        d_model=512,
        drop_layer=(3, 8, 9),
    )

    # print(model)
    y = model(x)
    print(y.shape)
