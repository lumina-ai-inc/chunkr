import torch
from torch import nn, Tensor, einsum
from typing import Optional, Tuple
import math
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F
from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


class ResBlock(nn.Module):
    def __init__(self, chan_in, hidden_size, chan_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan_in, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, chan_out, 1),
        )

    def forward(self, x):
        return self.net(x) + x


class BasicVAE(nn.Module):
    def get_codebook_indices(self, images):
        raise NotImplementedError()

    def decode(self, img_seq):
        raise NotImplementedError()

    def get_codebook_probs(self, img_seq):
        raise NotImplementedError()

    def get_image_tokens_size(self):
        pass

    def get_image_size(self):
        pass


class DiscreteVAE(BasicVAE):
    def __init__(
        self,
        image_size: Tuple[int, int] = [256, 256],  # input image size
        codebook_tokens: int = 512,  # codebook vocab size
        codebook_dim: int = 512,  # codebook embedding dimension
        num_layers: int = 3,  # layers of resnet blocks in encoder/decoder
        hidden_dim: int = 64,  # dimension in resnet blocks
        channels: int = 3,  # input channels
        smooth_l1_loss: bool = False,  # prevents exploding gradients
        temperature: float = 0.9,  # tau in gumbel softmax
        straight_through: bool = False,  # if True, the returned samples will be discretized as one-hot vectors, but will be differentiated as if it is the soft sample in autograd
        kl_div_loss_weight: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1, "number of layers must be greater than or equal to 1"

        self.image_size = image_size
        self.codebook_tokens = codebook_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(codebook_tokens, codebook_dim)

        encoder_layers = list()
        decoder_layers = list()

        encoder_in = channels
        decoder_in = codebook_dim

        for _ in range(num_layers):
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(encoder_in, hidden_dim, 4, stride=2, padding=1), nn.ReLU()
                )
            )
            encoder_layers.append(
                ResBlock(
                    chan_in=hidden_dim, hidden_size=hidden_dim, chan_out=hidden_dim
                )
            )
            encoder_in = hidden_dim

            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(decoder_in, hidden_dim, 4, stride=2, padding=1),
                    nn.ReLU(),
                )
            )
            decoder_layers.append(
                ResBlock(
                    chan_in=hidden_dim, hidden_size=hidden_dim, chan_out=hidden_dim
                )
            )
            decoder_in = hidden_dim

        encoder_layers.append(nn.Conv2d(hidden_dim, codebook_tokens, 1))
        decoder_layers.append(nn.Conv2d(hidden_dim, channels, 1))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

    def get_image_size(self):
        return self.image_size

    def get_image_tokens_size(self) -> int:
        ds_ratio = math.pow(2, self.num_layers)
        return int((self.image_size[0] // ds_ratio) * (self.image_size[1] // ds_ratio))

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images: Tensor):
        logits = self.forward(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1)
        return codebook_indices

    @torch.no_grad()
    @eval_decorator
    def get_codebook_probs(self, images: Tensor):
        logits = self.forward(images, return_logits=True)
        return nn.Softmax(dim=1)(logits)

    def decode(self, img_seq: Tensor):
        image_embeds = self.codebook(img_seq)
        image_embeds = image_embeds.permute((0, 3, 1, 2)).contiguous()

        # image_embeds = rearrange(image_embeds, "b h w d -> b d h w", h=h, w=w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img: Tensor,
        return_loss: bool = False,
        return_recons: bool = False,
        return_logits: bool = False,
        temp=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        assert (
            img.shape[-1] == self.image_size[0] and img.shape[-2] == self.image_size[1]
        ), f"input must have the correct image size {self.image_size}"

        logits = self.encoder(img)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(
            logits, tau=temp, dim=1, hard=self.straight_through
        )
        sampled = einsum(
            "b n h w, n d -> b d h w", soft_one_hot, self.codebook.weight
        ).contiguous()
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss
        recon_loss = self.loss_fn(img, out)

        # kl divergence
        logits = rearrange(logits, "b n h w -> b (h w) n").contiguous()
        qy = F.softmax(logits, dim=-1)

        log_qy = torch.log(qy + 1e-10)
        log_uniform = torch.log(
            torch.tensor([1.0 / self.codebook_tokens], device=img.device)
        )
        kl_div = F.kl_div(log_uniform, log_qy, None, None, "batchmean", log_target=True)

        loss = recon_loss + (kl_div * self.kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out


if __name__ == "__main__":
    input = torch.rand(1, 3, 256, 256)
    model = DiscreteVAE()
    loss, output = model(input, return_loss=True, return_recons=True)

    print(model)
    print(model.get_image_tokens_size())
    print(model.get_codebook_indices(input).shape)
    print(loss, output.shape, output.max(), output.min())
