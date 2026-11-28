# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Ollin Boer Bohan
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned TAEH3 preview decoder; native reference encoders remain unchanged."""

# Architecture adapted from madebyollin/taehv at
# e589fddc076e77f5ba8cd6baabe4ba3260b261cd. Original portions carry this license:
# MIT License
# Copyright (c) 2025 Ollin Boer Bohan
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)
TAEH3_PATH_ENV = "VLLM_OMNI_MINIMAX_H3_TAEH3_PATH"
CHECKPOINT_SHA256 = "af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba"


def _conv(in_channels: int, out_channels: int, *, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)


class _Clamp(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value / 3) * 3


class _MemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(in_channels * 2, out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat((value, past), dim=1)) + self.skip(value))


class _TGrow(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = value.shape
        return self.conv(value).reshape(-1, channels, height, width)


class H3TAEDecoder(nn.Module):
    """Lossy NCTHW decoder with request-local causal memory across batches."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(24, 256),
            nn.ReLU(inplace=True),
            _MemBlock(256, 256),
            _MemBlock(256, 256),
            _MemBlock(256, 256),
            nn.Upsample(scale_factor=2),
            _TGrow(256, 1),
            _conv(256, 128, bias=False),
            _MemBlock(128, 128),
            _MemBlock(128, 128),
            _MemBlock(128, 128),
            nn.Upsample(scale_factor=2),
            _TGrow(128, 2),
            _conv(128, 64, bias=False),
            _MemBlock(64, 64),
            _MemBlock(64, 64),
            _MemBlock(64, 64),
            nn.Upsample(scale_factor=2),
            _TGrow(64, 2),
            _conv(64, 64, bias=False),
            nn.ReLU(inplace=True),
            _conv(64, 12),
        )

    @classmethod
    def from_pretrained(cls, path: str) -> H3TAEDecoder:
        checkpoint = Path(path)
        hasher = hashlib.sha256()
        with checkpoint.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("TAEH3 checkpoint SHA256 mismatch; use the documented pinned taeh3.pth")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        decoder = cls()
        decoder.decoder.load_state_dict(
            {name.removeprefix("decoder."): value for name, value in state.items() if name.startswith("decoder.")},
            strict=True,
        )
        return decoder.eval().requires_grad_(False)

    @torch.inference_mode()
    def decode_video(self, latent: torch.Tensor, *, chunk_frames: int = 16) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 24 or min(latent.shape[2:]) <= 0:
            raise ValueError("TAEH3 requires nonempty NCTHW latents with batch size 1 and 24 channels")
        if chunk_frames <= 0:
            raise ValueError("TAEH3 chunk_frames must be positive")
        parameter = next(self.parameters())
        past: dict[int, torch.Tensor] = {}
        parts: list[torch.Tensor] = []
        for raw in latent.permute(0, 2, 1, 3, 4).split(chunk_frames, dim=1):
            value = raw[0].to(device=parameter.device, dtype=parameter.dtype).contiguous()
            for index, block in enumerate(self.decoder):
                if isinstance(block, _MemBlock):
                    previous = past.get(index)
                    if previous is None:
                        previous = torch.zeros_like(value[:1])
                    history = torch.cat((previous, value[:-1]), dim=0)
                    # Keep one frame, not a view retaining an entire activation batch.
                    past[index] = value[-1:].clone()
                    value = block(value, history)
                elif isinstance(block, nn.Upsample) and value.shape[0] > 128:
                    value = torch.cat([block(part) for part in value.split(128)], dim=0)
                else:
                    value = block(value)
            parts.append(F.pixel_shuffle(value, 2))
        value = torch.cat(parts, dim=0)
        # H3: groups of five latent tokens -> 20 raw frames -> trim 3 per
        # group, then drop the final 12 frames. 110 tokens give 362 frames.
        value = F.pad(value, (0, 0, 0, 0, 0, 0, 0, -value.shape[0] % 20))
        value = value.unflatten(0, (-1, 20))[:, 3:].flatten(0, 1)[:-12]
        return value.clamp(0, 1).permute(1, 0, 2, 3).unsqueeze(0).contiguous().float()


def install_taeh3_decoder(
    model: nn.Module, *, device: torch.device, load_device: torch.device, encode_only: bool, latent_channels: int
) -> None:
    """Read the path opt-in once and register before residency snapshots."""
    path = os.getenv(TAEH3_PATH_ENV, "").strip()
    if not path or encode_only:
        return
    if device.type != "npu" or latent_channels != 24:
        raise ValueError(f"{TAEH3_PATH_ENV} requires the native 24-channel H3 VAE on Ascend")
    model.add_module(
        "_omni_taeh3_decoder", H3TAEDecoder.from_pretrained(path).to(device=load_device, dtype=torch.float16)
    )
    logger.info("MiniMax-H3 TAEH3 installed: checkpoint_sha256=%s; lossy video decode only", CHECKPOINT_SHA256)
