# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU mathematical and opt-in contracts; these do not validate an A5 kernel."""

from types import MethodType

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.models.minimax_h3.npu import audio_tconv

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.stride = self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer("filter", torch.rand(1, 1, kernel_size))

    def forward(self, x):
        padded = F.pad(x, (self.pad, self.pad), mode="replicate")
        result = self.ratio * F.conv_transpose1d(
            padded, self.filter.expand(x.shape[1], -1, -1), stride=self.stride, groups=x.shape[1]
        )
        return result[..., self.pad_left : result.shape[-1] - self.pad_right]


@pytest.mark.parametrize("kernel,stride,padding", [(9, 5, 2), (4, 2, 1), (7, 2, 0), (8, 3, 1)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_full_phase_decomposition(kernel, stride, padding, has_bias):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 13, dtype=torch.float64)
    weight = torch.randn(3, 5, kernel, dtype=torch.float64)
    bias = torch.randn(5, dtype=torch.float64) if has_bias else None
    expected = F.conv_transpose1d(x, weight, bias, stride=stride, padding=padding)
    actual = audio_tconv._conv_transpose1d_as_conv1d(x, weight, bias, stride, padding)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kernel,stride", [(12, 2), (8, 4), (7, 2), (8, 3)])
def test_depthwise_length_gain_and_crop(kernel, stride):
    torch.manual_seed(42)
    module = UpSample1d(stride, kernel).double()
    x = torch.randn(2, 3, 17, dtype=torch.float64)
    actual = audio_tconv._depthwise_upsample_as_conv1d(
        x, module.filter, module.stride, module.pad, module.pad_left, module.pad_right, module.ratio
    )
    torch.testing.assert_close(actual, module(x), rtol=0, atol=1e-12)


@pytest.mark.parametrize("value", [None, "0", "false", "off", "", "invalid"])
def test_default_and_disabled_leave_forwards_untouched(monkeypatch, value):
    monkeypatch.delenv(audio_tconv.AUDIO_TCONV_CONV1D_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(audio_tconv.AUDIO_TCONV_CONV1D_ENV, value)
    module = nn.ConvTranspose1d(3, 5, 4, stride=2)
    original = module.forward.__func__
    assert audio_tconv.install_audio_vae_tconv_conv1d(module) == (0, 0)
    assert module.forward.__func__ is original


@pytest.mark.parametrize("value", ["1", "true", "yes", " ON "])
def test_install_is_idempotent_and_preserves_storage(monkeypatch, value):
    monkeypatch.setenv(audio_tconv.AUDIO_TCONV_CONV1D_ENV, value)
    model = nn.Sequential(nn.ConvTranspose1d(3, 5, 4, stride=2), UpSample1d())
    parameters = {name: id(p) for name, p in model.named_parameters()}
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    assert audio_tconv.install_audio_vae_tconv_conv1d(model) == (1, 1)
    assert audio_tconv.install_audio_vae_tconv_conv1d(model) == (0, 0)
    assert parameters == {name: id(p) for name, p in model.named_parameters()}
    assert state.keys() == model.state_dict().keys()
    for name, expected in state.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("extra", [{"groups": 2}, {"dilation": 2}, {"output_padding": 1}, {}])
def test_original_fallback_preserves_explicit_output_size(monkeypatch, extra):
    monkeypatch.setenv(audio_tconv.AUDIO_TCONV_CONV1D_ENV, "1")
    module = nn.ConvTranspose1d(4, 4, 4, stride=2, padding=1, **extra).eval()
    x = torch.randn(2, 4, 11)
    output_size = list(module(x).shape)
    if not extra.get("output_padding"):
        output_size[-1] += 1
    expected = module(x, output_size=output_size)
    audio_tconv.install_audio_vae_tconv_conv1d(module)
    torch.testing.assert_close(module(x, output_size=output_size), expected, rtol=0, atol=0)


def test_cpu_forward_and_backward_remain_native(monkeypatch):
    monkeypatch.setenv(audio_tconv.AUDIO_TCONV_CONV1D_ENV, "1")
    module = UpSample1d()
    original = module.forward
    x = torch.randn(2, 3, 11, requires_grad=True)
    expected = original(x)
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    audio_tconv.install_audio_vae_tconv_conv1d(module)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], expected_grad, rtol=0, atol=0)


def test_forced_dispatch_exercises_both_rewrites(monkeypatch):
    # Exercise wrapper plumbing with CPU math; this is not NPU validation.
    monkeypatch.setenv(audio_tconv.AUDIO_TCONV_CONV1D_ENV, "1")
    monkeypatch.setattr(audio_tconv, "_supported_input", lambda module, x, weight: True)
    for module in (nn.ConvTranspose1d(3, 5, 9, stride=5, padding=2), UpSample1d(2, 7)):
        module.eval()
        x = torch.randn(2, 3, 11)
        expected = module(x)
        audio_tconv.install_audio_vae_tconv_conv1d(module)
        torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-6)


def test_existing_forward_override_is_not_replaced(monkeypatch):
    monkeypatch.setenv(audio_tconv.AUDIO_TCONV_CONV1D_ENV, "1")
    module = nn.ConvTranspose1d(3, 5, 4)
    module.forward = MethodType(lambda self, x: x, module)
    saved = module.forward
    assert audio_tconv.install_audio_vae_tconv_conv1d(module) == (0, 0)
    assert module.forward is saved
