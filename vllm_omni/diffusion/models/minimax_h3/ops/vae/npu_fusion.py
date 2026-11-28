# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in native-precision RMSNorm and SwiGLU for the H3 video decoder.

This is the earlier A2 operator replacement, not VAE W8A8 quantization.
Patch forwards in place before residency snapshots; retain parameter identity,
checkpoint keys, normalization precision and the checkpoint's autocast policy.
"""

from __future__ import annotations

import os
from types import MethodType

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)
FUSION_ENV = "VLLM_OMNI_MINIMAX_H3_VAE_NPU_FUSION"


def _supported_input(module: nn.Module, x: torch.Tensor) -> bool:
    return (
        x.device.type == "npu"
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and 2 <= x.ndim <= 8
        and x.numel() > 0
        and not module.training
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    )


def _rms_forward(self: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    if (
        not _supported_input(self, x)
        or self.eps is None
        or x.shape[-1:] != self.normalized_shape
        or (self.weight is not None and (self.weight.dtype != x.dtype or self.weight.device != x.device))
    ):
        return self._h3_npu_original_forward(x)
    import torch_npu

    gamma = self.weight
    if gamma is None:
        # Registered before PinnedModuleStager snapshots the model's buffers.
        # Do not rebind the buffer: a stager may retain its original identity.
        gamma = self._h3_npu_unit_gamma.to(device=x.device, dtype=x.dtype)
    return torch_npu.npu_rms_norm(x.contiguous(), gamma=gamma, epsilon=self.eps)[0]


def _ff_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not _supported_input(self, x) or getattr(self, "_compile_forward_enabled", False):
        return self._h3_npu_original_forward(x)
    import torch_npu

    projected = self.w1(x)
    # Native gated FFN layout is [gate, value], hence silu(gate) * value.
    # Preserve w1/w2 and the surrounding autocast; do not cast or quantize them.
    return self.w2(torch_npu.npu_swiglu(projected, dim=-1))


def _patch_norm(norm: nn.Module | None, reference: torch.Tensor | None) -> bool:
    if (
        type(norm) is not nn.RMSNorm
        or len(norm.normalized_shape) != 1
        or norm.eps is None
        or hasattr(norm, "_h3_npu_original_forward")
        or "forward" in norm.__dict__
    ):
        return False
    if norm.weight is None:
        if reference is None or reference.is_meta:
            return False
        norm.register_buffer(
            "_h3_npu_unit_gamma",
            torch.ones(norm.normalized_shape, device=reference.device, dtype=torch.float32),
            persistent=False,
        )
    norm._h3_npu_original_forward = norm.forward
    norm.forward = MethodType(_rms_forward, norm)
    return True


def install_npu_vae_fusion(decoder: nn.Module, *, device: torch.device) -> bool:
    """Install eligible native decoder operators once; default is disabled."""
    if device.type != "npu" or os.getenv(FUSION_ENV, "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    if getattr(decoder, "_h3_npu_fusion_installed", False):
        return True
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        return False
    norms = feed_forwards = 0
    for block in blocks:
        if getattr(block, "_orig_mod", None) is not None or getattr(block, "_compiled_call_impl", None) is not None:
            continue
        reference = next(block.parameters(), None)
        attention = getattr(block, "attn", None)
        for owner, names in ((block, ("norm1", "norm2")), (attention, ("norm_q", "norm_k"))):
            for name in names:
                norms += _patch_norm(getattr(owner, name, None), reference)
        ff = getattr(block, "ff", None)
        if (
            isinstance(ff, nn.Module)
            and not hasattr(ff, "_h3_npu_original_forward")
            and "forward" not in ff.__dict__
            and getattr(ff, "use_gated", False)
            and isinstance(getattr(ff, "act_fn", None), nn.SiLU)
            and not ff.act_fn.inplace
            and not getattr(ff, "_compile_forward_enabled", False)
            and isinstance(getattr(ff, "w1", None), nn.Linear)
            and isinstance(getattr(ff, "w2", None), nn.Linear)
            and ff.w1.out_features == 2 * ff.w2.in_features
        ):
            ff._h3_npu_original_forward = ff.forward
            ff.forward = MethodType(_ff_forward, ff)
            feed_forwards += 1
    decoder._h3_npu_fusion_installed = bool(norms or feed_forwards)
    logger.info("MiniMax-H3 native-precision VAE fusion installed: rmsnorm=%d, swiglu=%d", norms, feed_forwards)
    return decoder._h3_npu_fusion_installed
