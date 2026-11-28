# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Version-matched release contracts; no device kernels or weights required."""

from dataclasses import dataclass

import pytest
import torch
from torch.nn import functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.kv_offload_config import parse_reference_kv_config, validate_kv_isolate
from vllm_omni.diffusion.models.minimax_h3.reference_isolation import (
    MiniMaxH3ReferenceIsolation,
    reference_isolation_enabled,
)
from vllm_omni.diffusion.models.minimax_h3.reference_isolation_cache import create_isolated_reference_cache
from vllm_omni.diffusion.models.minimax_h3.reference_kv_cache import MiniMaxH3ReferenceKVTier2State

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass
class _Options:
    kv_isolate: bool | None = False


@pytest.mark.parametrize("value", [None, False, True])
def test_isolation_uses_serialized_boolean_and_ignores_old_environment(monkeypatch, value):
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_ISOLATION", "0" if value else "1")
    assert reference_isolation_enabled(_Options(value)) is (value is True)
    assert reference_isolation_enabled() is False


@pytest.mark.parametrize("value", ["true", "1", 0, 1, [], {}])
def test_isolation_rejects_truthy_non_booleans(value):
    with pytest.raises(ValueError, match="boolean"):
        validate_kv_isolate(value)


def test_original_on_mode_and_defaults_are_retained():
    assert parse_reference_kv_config(None) is None
    config = parse_reference_kv_config({"mode": "on"})
    assert config.kv_refresh_interval == 2
    assert config.kv_host_quantization == "none"
    assert config.cache_dit_policy == "refresh" and config.max_refresh_interval is None
    hit = parse_reference_kv_config({"mode": "on", "kv_refresh_interval": 16, "cache_dit_policy": "hit"})
    assert hit.max_refresh_interval == 32


@pytest.mark.parametrize(
    "updates",
    [
        {"mode": "tier2"},
        {"unknown": 1},
        {"kv_refresh_interval": True},
        {"kv_refresh_interval": -1},
        {"kv_host_quantization": "mxfp8"},
        {"cache_dit_policy": "invalid"},
        {"kv_refresh_interval": 0, "cache_dit_policy": "hit"},
        {"cache_dit_policy": "hit", "max_refresh_interval": 1},
        {"max_refresh_interval": 32},
    ],
)
def test_invalid_offload_options_are_rejected(updates):
    with pytest.raises(ValueError):
        parse_reference_kv_config({"mode": "on", **updates})


def test_isolate_factory_keeps_original_once_int8_ring_policy():
    assert create_isolated_reference_cache(num_layers=3, global_reference_rows=0, device=torch.device("cpu")) is None
    state = create_isolated_reference_cache(num_layers=3, global_reference_rows=2, device=torch.device("cpu"))
    assert state.refresh_interval == 0 and state.host_quantization == "int8" and state.ring_size == 2
    assert not state.cache_dit_schedule_active and state.cache_dit_policy == "refresh"
    state.close()


@pytest.mark.parametrize("quantization", ["none", "int8"])
def test_layer_major_pool_compact_rows_and_cleanup(quantization):
    state = MiniMaxH3ReferenceKVTier2State(
        num_layers=3,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=16,
        pin_memory=False,
        host_quantization=quantization,
    )
    mask = torch.tensor([False, True, False, True, False, False])
    torch.manual_seed(42)
    originals = [(torch.randn(1, 6, 2, 4) + index, torch.randn(1, 6, 2, 4) - index) for index in range(3)]
    state.begin_step(0)
    state.set_global_reference_mask(mask)
    for layer, (key, value) in enumerate(originals):
        state.process_post_parallel_layer(layer, key, value, compact=False, parallel_strategy="ulysses")
    state.end_step()
    assert state._host_pool.shape == (3, 2, 2, 2, 4)
    assert all(state._host_pool[layer].is_contiguous() for layer in range(3))
    if quantization == "int8":
        assert state._host_pool.dtype == torch.int8 and state._host_scale_pool.shape == (3, 2, 2)
    state.begin_step(1)
    for layer, (key, value) in enumerate(originals):
        actual = state.process_post_parallel_layer(
            layer, key[:, ~mask], value[:, ~mask], compact=True, parallel_strategy="ulysses"
        )
        for result, source in zip(actual, (key, value)):
            expected = torch.cat((source[:, mask], source[:, ~mask]), dim=1)
            tolerance = float(source.abs().max()) / 254 + 1e-5 if quantization == "int8" else 0
            torch.testing.assert_close(result, expected, rtol=0, atol=tolerance)
    state.end_step()
    assert state.stats.refresh_steps == 1 and state.stats.captured_layers == 3
    assert state.stats.substituted_layers == 3
    state.close()
    assert state._host_pool is None and state._device_ring is None


def _dense(query, key, value, metadata):
    assert metadata.attn_mask is None
    assert metadata.extra["max_seqlen_q"] == query.shape[1]
    assert metadata.extra["max_seqlen_k"] == key.shape[1]
    return F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)).transpose(
        1, 2
    )


@pytest.mark.parametrize("references", [(), (1,), (1, 4), (0, 2, 6)])
def test_mask_free_split_matches_explicit_four_region_oracle(references):
    state = MiniMaxH3ReferenceIsolation(
        torch.tensor(references, dtype=torch.long),
        sequence_rows=9,
        used_rows=7,
        condition_timestep=0.999,
        device=torch.device("cpu"),
    )
    torch.manual_seed(71)
    query, key, value = [torch.randn(1, 9, 2, 4, dtype=torch.float64) for _ in range(3)]
    result = state.attend(query, key, value, AttentionMetadata(), compact=False, dense_forward=_dense)
    reference = torch.zeros(7, dtype=torch.bool)
    reference[list(references)] = True
    allowed = torch.ones(7, 7, dtype=torch.bool)
    allowed[reference] = reference
    oracle = F.scaled_dot_product_attention(
        query[:, :7].transpose(1, 2), key[:, :7].transpose(1, 2), value[:, :7].transpose(1, 2), attn_mask=allowed
    ).transpose(1, 2)
    torch.testing.assert_close(result[:, :7], oracle, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(result[:, 7:]) == 0
    changed_value = value.clone()
    changed_value[:, state.dynamic_positions] += 100
    changed = state.attend(query, key, changed_value, AttentionMetadata(), compact=False, dense_forward=_dense)
    torch.testing.assert_close(
        result[:, state.reference_positions], changed[:, state.reference_positions], rtol=0, atol=0
    )
    state.close()
