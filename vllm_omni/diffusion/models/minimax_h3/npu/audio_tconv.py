# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in phase-decomposed Conv1d for the native H3 audio VAE.

Adapted from vllm-project/vllm-omni#8813 and the earlier A2 integration.
Parameters and checkpoint keys are unchanged. CPU, training, compilation and
unsupported inputs retain the original forwards. No custom NPU kernel is needed.
"""

from __future__ import annotations

import os
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)
AUDIO_TCONV_CONV1D_ENV = "VLLM_OMNI_MINIMAX_H3_AUDIO_TCONV_CONV1D"


def _conv_transpose1d_as_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """Evaluate a groups=1, dilation=1 transposed convolution by phases."""
    batch, _, length = x.shape
    kernel_size = weight.shape[-1]
    if kernel_size % stride:
        weight = F.pad(weight, (0, -kernel_size % stride))
    phase_taps = weight.shape[-1] // stride
    in_channels, out_channels = weight.shape[:2]
    phase_weight = (
        weight.reshape(in_channels, out_channels, phase_taps, stride)
        .permute(1, 3, 0, 2)
        .flip(-1)
        .reshape(out_channels * stride, in_channels, phase_taps)
    )
    out = F.conv1d(x, phase_weight, padding=phase_taps - 1)
    out = out.reshape(batch, out_channels, stride, -1).transpose(2, 3).reshape(batch, out_channels, -1)
    output_length = (length - 1) * stride - 2 * padding + kernel_size
    out = out[..., padding : padding + output_length]
    return out if bias is None else out + bias.reshape(1, -1, 1)


def _depthwise_upsample_as_conv1d(
    x: torch.Tensor,
    filter_: torch.Tensor,
    stride: int,
    pad: int,
    pad_left: int,
    pad_right: int,
    ratio: float,
) -> torch.Tensor:
    """Keep the native replicate padding, gain, length and edge crops."""
    batch, channels, _ = x.shape
    kernel = filter_.expand(channels, -1, -1)
    kernel_size = kernel.shape[-1]
    if kernel_size % stride:
        kernel = F.pad(kernel, (0, -kernel_size % stride))
    phase_taps = kernel.shape[-1] // stride
    phase_weight = (
        kernel.reshape(channels, 1, phase_taps, stride)
        .permute(0, 3, 1, 2)
        .flip(-1)
        .reshape(channels * stride, 1, phase_taps)
    )
    x = F.pad(x, (pad, pad), mode="replicate")
    # Use the unpadded kernel length, including when K % stride != 0.
    output_length = (x.shape[-1] - 1) * stride + kernel_size
    out = F.conv1d(x, phase_weight, padding=phase_taps - 1, groups=channels)
    out = out.reshape(batch, channels, stride, -1).transpose(2, 3).reshape(batch, channels, -1)
    return (out * ratio)[..., pad_left : output_length - pad_right]


def _supported_input(module: nn.Module, x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.device.type == "npu"
        and x.dtype == torch.float32
        and weight.dtype == x.dtype
        and weight.device == x.device
        and x.ndim == 3
        and not module.training
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    )


def _transpose_forward(self: nn.ConvTranspose1d, x: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
    if (
        _supported_input(self, x, self.weight)
        and self.groups == 1
        and self.dilation == (1,)
        and self.output_padding == (0,)
        and output_size is None
    ):
        return _conv_transpose1d_as_conv1d(x, self.weight, self.bias, self.stride[0], self.padding[0])
    return self._h3_tconv_original_forward(x, output_size=output_size)


def _upsample_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if _supported_input(self, x, self.filter):
        return _depthwise_upsample_as_conv1d(
            x, self.filter, self.stride, self.pad, self.pad_left, self.pad_right, self.ratio
        )
    return self._h3_tconv_original_forward(x)


def _is_upsample(module: nn.Module) -> bool:
    filter_ = getattr(module, "filter", None)
    return (
        type(module).__name__ == "UpSample1d"
        and isinstance(filter_, torch.Tensor)
        and filter_.ndim == 3
        and filter_.shape[:2] == (1, 1)
        and all(hasattr(module, name) for name in ("stride", "pad", "pad_left", "pad_right", "ratio"))
        and isinstance(module.stride, int)
        and module.stride > 0
    )


def install_audio_vae_tconv_conv1d(model: nn.Module) -> tuple[int, int]:
    """Install once at initialization; returns (upsample, transpose) counts."""
    if os.getenv(AUDIO_TCONV_CONV1D_ENV, "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return 0, 0
    upsample_count = transpose_count = 0
    for module in model.modules():
        if hasattr(module, "_h3_tconv_original_forward") or "forward" in module.__dict__:
            continue
        if (
            isinstance(module, nn.ConvTranspose1d)
            and getattr(module.forward, "__func__", None) is nn.ConvTranspose1d.forward
        ):
            module._h3_tconv_original_forward = module.forward
            module.forward = MethodType(_transpose_forward, module)
            transpose_count += 1
        elif _is_upsample(module):
            module._h3_tconv_original_forward = module.forward
            module.forward = MethodType(_upsample_forward, module)
            upsample_count += 1
    logger.info(
        "MiniMax-H3 audio Conv1d installed: stage_modules=%d, upsample_modules=%d, precision=fp32",
        transpose_count,
        upsample_count,
    )
    return upsample_count, transpose_count
