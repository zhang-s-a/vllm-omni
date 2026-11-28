# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TAEH3 opt-in, causal batch boundaries and normalized-latent contracts."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.models.minimax_h3.npu import taeh3

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass
class _NPUTarget:
    type: str = "npu"


def test_disabled_and_encoder_only_never_load_checkpoint(monkeypatch):
    monkeypatch.delenv(taeh3.TAEH3_PATH_ENV, raising=False)
    model = nn.Module()
    taeh3.install_taeh3_decoder(
        model, device=torch.device("cpu"), load_device=torch.device("cpu"), encode_only=False, latent_channels=24
    )
    assert not list(model.parameters())
    monkeypatch.setenv(taeh3.TAEH3_PATH_ENV, "/intentionally/missing.pth")
    taeh3.install_taeh3_decoder(
        model, device=_NPUTarget(), load_device=torch.device("cpu"), encode_only=True, latent_channels=24
    )
    assert not list(model.parameters())


@pytest.mark.parametrize("target,channels", [(torch.device("cpu"), 24), (_NPUTarget(), 48)])
def test_enabled_rejects_incompatible_target_or_latent(monkeypatch, target, channels):
    monkeypatch.setenv(taeh3.TAEH3_PATH_ENV, "/intentionally/missing.pth")
    with pytest.raises(ValueError, match="24-channel"):
        taeh3.install_taeh3_decoder(
            nn.Module(), device=target, load_device=torch.device("cpu"), encode_only=False, latent_channels=channels
        )


def test_checksum_is_checked_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "wrong.pth"
    path.write_bytes(b"wrong checkpoint")

    def must_not_load(*args, **kwargs):
        pytest.fail("unverified checkpoint must not be deserialized")

    monkeypatch.setattr(torch, "load", must_not_load)
    with pytest.raises(ValueError, match="SHA256"):
        taeh3.H3TAEDecoder.from_pretrained(str(path))


def test_registration_preserves_native_modules_and_includes_tae_in_state(monkeypatch):
    monkeypatch.setenv(taeh3.TAEH3_PATH_ENV, "/fixture/taeh3.pth")
    decoder = nn.Linear(2, 2)
    monkeypatch.setattr(taeh3.H3TAEDecoder, "from_pretrained", lambda path: decoder)
    model = nn.Module()
    model.encoder = nn.Linear(2, 2)
    model.decoder = nn.Linear(2, 2)
    original_parameters = list(model.parameters())
    taeh3.install_taeh3_decoder(
        model, device=_NPUTarget(), load_device=torch.device("cpu"), encode_only=False, latent_channels=24
    )
    assert list(model.parameters())[:4] == original_parameters
    assert model._omni_taeh3_decoder is decoder
    assert decoder.weight.dtype == torch.float16
    assert any(name.startswith("_omni_taeh3_decoder.") for name in model.state_dict())
    model.to(dtype=torch.float32)
    assert decoder.weight.dtype == torch.float32  # Normal module movement owns TAE too.


def _parallel_reference(decoder, latent):
    """Upstream vectorized causal rule and H3 temporal trim, independent batching."""
    value = latent.permute(0, 2, 1, 3, 4)[0]
    for block in decoder.decoder:
        if isinstance(block, taeh3._MemBlock):
            history = F.pad(value[:-1], (0, 0, 0, 0, 0, 0, 1, 0))
            value = block(value, history)
        else:
            value = block(value)
    value = F.pixel_shuffle(value, 2)
    value = F.pad(value, (0, 0, 0, 0, 0, 0, 0, -value.shape[0] % 20))
    value = value.unflatten(0, (-1, 20))[:, 3:].flatten(0, 1)[:-12]
    return value.clamp(0, 1).permute(1, 0, 2, 3).unsqueeze(0).contiguous().float()


@pytest.mark.parametrize("length,expected_frames", [(5, 5), (8, 22), (10, 22), (110, 362)])
def test_causal_batches_match_parallel_temporal_reference(length, expected_frames):
    torch.manual_seed(42)
    decoder = taeh3.H3TAEDecoder().eval()
    latent = torch.randn(1, 24, length, 1, 1)
    with torch.inference_mode():
        expected = _parallel_reference(decoder, latent)
        actual = decoder.decode_video(latent, chunk_frames=3)
        repeated = decoder.decode_video(latent, chunk_frames=16)
    assert actual.shape == (1, 3, expected_frames, 16, 16)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(repeated, actual, rtol=1e-4, atol=1e-6)
    assert actual.dtype == torch.float32 and actual.min() >= 0 and actual.max() <= 1


@pytest.mark.parametrize("shape", [(2, 24, 10, 1, 1), (1, 48, 10, 1, 1), (1, 24, 0, 1, 1)])
def test_invalid_latents_fail_before_compute(shape):
    with pytest.raises(ValueError, match="NCTHW"):
        taeh3.H3TAEDecoder().decode_video(torch.empty(shape))
