# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on 3D data.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 3
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xyz(self, x, y, z):
        # The positions are expected to be normalized
        assert len(x) == len(y) == len(z) and x.ndim == y.ndim == z.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale
        z_embed = z * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_z = z_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_z = torch.stack(
            (pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y, pos_z

    @torch.no_grad()
    def encode_boxes(self, x, y, z, w, h, d):
        pos_x, pos_y, pos_z = self._encode_xyz(x, y, z)
        pos = torch.cat((pos_y, pos_x, pos_z, h[:, None], w[:, None], d[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, z, labels):
        (bx, nx), (by, ny), (bz, nz), (bl, nl) = x.shape, y.shape, z.shape, labels.shape
        assert bx == by == bz and nx == ny == nz and bx == bl and nx == nl
        pos_x, pos_y, pos_z = self._encode_xyz(x.flatten(), y.flatten(), z.flatten())
        pos_x, pos_y, pos_z = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1), pos_z.reshape(bz, nz, -1)
        pos = torch.cat((pos_y, pos_x, pos_z, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        cache_key = (x.shape[-3], x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1, 1)
        z_embed = (
            torch.arange(1, x.shape[-3] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1, 1)
            .repeat(x.shape[0], 1, x.shape[-2], x.shape[-1])
        )
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1, 1)
            .repeat(x.shape[0], x.shape[-3], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, 1, -1)
            .repeat(x.shape[0], x.shape[-3], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        pos_z = torch.stack(
            (pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        pos = torch.cat((pos_y, pos_x, pos_z), dim=4).permute(0, 4, 1, 2, 3)
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^3 cube and have d_1 x ... x d_n x 3 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        d, h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((d, h, w), device=device, dtype=torch.float32)
        z_embed = grid.cumsum(dim=0) - 0.5
        y_embed = grid.cumsum(dim=1) - 0.5
        x_embed = grid.cumsum(dim=2) - 0.5
        z_embed = z_embed / d
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x D x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[2]
        coords[:, :, 1] = coords[:, :, 1] / image_size[1]
        coords[:, :, 2] = coords[:, :, 2] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch


def init_t_xyz(end_x: int, end_y: int, end_z: int):
    t = torch.arange(end_x * end_y * end_z, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = (torch.div(t, end_x, rounding_mode="floor") % end_y).float()
    t_z = torch.div(t, end_x * end_y, rounding_mode="floor").float()
    return t_x, t_y, t_z


def compute_axial_cis(dim: int, end_x: int, end_y: int, end_z: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)].float() / dim))
    freqs_z = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)].float() / dim))

    t_x, t_y, t_z = init_t_xyz(end_x, end_y, end_z)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_z = torch.outer(t_z, freqs_z)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    freqs_cis_z = torch.polar(torch.ones_like(freqs_z), freqs_z)
    return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_z], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(4)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # torch.repeat on complex numbers may not be supported on non-CUDA devices
            # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(4)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)