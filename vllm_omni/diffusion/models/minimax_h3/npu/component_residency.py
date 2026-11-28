# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in H3 auxiliary residency without changing the DiT offload policy."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm_omni.diffusion.offloader.config import TEXT_ENCODER_COMPONENT, OffloadStrategy, resolve_offload
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

RESIDENT_ENV = "VLLM_OMNI_MINIMAX_H3_KEEP_COMPONENTS_RESIDENT"
_AUXILIARIES = frozenset({"text_encoder", "video_vae", "audio_vae"})


def auxiliary_residency_enabled(config: OmniDiffusionConfig, *, device: torch.device) -> bool:
    """Validate once before loading; conflicting explicit policies fail early."""
    if os.getenv(RESIDENT_ENV, "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    if device.type != "npu":
        raise ValueError(f"{RESIDENT_ENV} requires an Ascend NPU target")
    policy = resolve_offload(config)
    if policy.strategy is OffloadStrategy.MODEL_LEVEL:
        raise ValueError(f"{RESIDENT_ENV} supports layerwise/DLO or no offload, not module-level offload")
    if policy.public is not None and TEXT_ENCODER_COMPONENT in policy.components:
        raise ValueError(f"{RESIDENT_ENV} conflicts with explicit text_encoder offload; select only dit instead")
    return True


def resident_auxiliary_plan(plan: OffloadPlan) -> OffloadPlan:
    """Keep DiT stacks, child hooks and transfer declarations intact."""
    return replace(
        plan,
        encoder_block_attrs={
            name: paths for name, paths in plan.encoder_block_attrs.items() if name not in _AUXILIARIES
        },
        on_demand_component_paths=plan.on_demand_component_paths - _AUXILIARIES,
    )


def is_resident_auxiliary(pipeline: nn.Module, component: nn.Module) -> bool:
    """Recognize native VAE split proxies without installing movement hooks."""
    for name in _AUXILIARIES:
        root = getattr(pipeline, name, None)
        if root is None:
            continue
        if component is root:
            return True
        if name == "video_vae" and any(
            component is getattr(root, part, None) for part in ("encoder_component", "decoder_component")
        ):
            return True
    return False
