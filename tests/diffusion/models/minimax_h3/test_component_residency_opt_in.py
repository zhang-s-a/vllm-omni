# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Auxiliary residency policy preserves the independent DiT transfer plan."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.npu.component_residency import (
    RESIDENT_ENV,
    auxiliary_residency_enabled,
    is_resident_auxiliary,
    resident_auxiliary_plan,
)
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass
class _Config:
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    enable_distributed_layerwise_offload: bool = False
    diffusion_offload_config: dict[str, object] | None = None
    dlo_use_allgather: bool = True


@dataclass
class _Target:
    type: str = "npu"


@pytest.mark.parametrize("value", [None, "0", "false", "off", ""])
def test_disabled_does_not_change_other_platforms_or_policies(monkeypatch, value):
    monkeypatch.delenv(RESIDENT_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(RESIDENT_ENV, value)
    assert not auxiliary_residency_enabled(_Config(enable_cpu_offload=True), device=torch.device("cpu"))


@pytest.mark.parametrize(
    "options", [{}, {"enable_layerwise_offload": True}, {"enable_distributed_layerwise_offload": True}]
)
@pytest.mark.parametrize("value", ["1", " ON "])
def test_enabled_supports_current_legacy_layer_backends(monkeypatch, options, value):
    monkeypatch.setenv(RESIDENT_ENV, value)
    assert auxiliary_residency_enabled(_Config(**options), device=_Target())


def test_explicit_dit_only_policy_is_compatible(monkeypatch):
    monkeypatch.setenv(RESIDENT_ENV, "1")
    config = _Config(diffusion_offload_config={"mode": "layer", "components": ["dit"]})
    assert auxiliary_residency_enabled(config, device=_Target())


@pytest.mark.parametrize(
    "config, message",
    [
        (_Config(enable_cpu_offload=True), "module-level"),
        (_Config(diffusion_offload_config={"mode": "layer", "components": ["dit", "text_encoder"]}), "text_encoder"),
    ],
)
def test_conflicting_policy_is_not_silently_overridden(monkeypatch, config, message):
    monkeypatch.setenv(RESIDENT_ENV, "1")
    with pytest.raises(ValueError, match=message):
        auxiliary_residency_enabled(config, device=_Target())


def test_enabled_rejects_non_npu_target(monkeypatch):
    monkeypatch.setenv(RESIDENT_ENV, "1")
    with pytest.raises(ValueError, match="Ascend"):
        auxiliary_residency_enabled(_Config(), device=torch.device("cpu"))


def test_auxiliary_plan_is_instance_local_and_leaves_dit_intact():
    original = OffloadPlan(
        block_attrs={"transformer": ("layers",)},
        offload_submodules={"token_refiner": "blocks"},
        resident_dit_paths=frozenset({"transformer"}),
        encoder_block_attrs={"text_encoder": ("text_model.layers",), "other_encoder": ("blocks",)},
        on_demand_component_paths=frozenset({"text_encoder", "video_vae", "audio_vae", "other_encoder"}),
    )
    result = resident_auxiliary_plan(original)
    assert result.block_attrs is original.block_attrs
    assert result.offload_submodules is original.offload_submodules
    assert result.resident_dit_paths is original.resident_dit_paths
    assert result.encoder_block_attrs == {"other_encoder": ("blocks",)}
    assert result.on_demand_component_paths == {"other_encoder"}
    assert "text_encoder" in original.encoder_block_attrs
    assert "video_vae" in original.on_demand_component_paths


def test_split_vae_proxies_and_only_loaded_auxiliaries_are_recognized():
    pipeline = nn.Module()
    pipeline.text_encoder = nn.Linear(2, 2)
    pipeline.audio_vae = nn.Linear(2, 2)
    pipeline.video_vae = nn.Module()
    pipeline.video_vae.encoder_component = nn.Identity()
    pipeline.video_vae.decoder_component = nn.Identity()
    for component in (
        pipeline.text_encoder,
        pipeline.audio_vae,
        pipeline.video_vae,
        pipeline.video_vae.encoder_component,
        pipeline.video_vae.decoder_component,
    ):
        assert is_resident_auxiliary(pipeline, component)
    assert not is_resident_auxiliary(pipeline, nn.Identity())
    assert not is_resident_auxiliary(nn.Module(), nn.Identity())
