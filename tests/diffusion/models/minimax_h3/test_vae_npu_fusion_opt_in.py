# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native-precision operator contracts with CPU NPU-operator doubles."""

import sys
from types import MethodType, ModuleType

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.models.minimax_h3.ops.vae import npu_fusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _NPUTarget:
    type = "npu"


class _FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1, self.w2 = nn.Linear(8, 24), nn.Linear(12, 8)
        self.act_fn = nn.SiLU()
        self.use_gated = True
        self._compile_forward_enabled = False

    def forward(self, x):
        gate, value = self.w1(x).chunk(2, dim=-1)
        return self.w2(self.act_fn(gate) * value)


def _decoder():
    decoder = nn.Module()
    block = nn.Module()
    block.norm1 = nn.RMSNorm(8, eps=1e-5)
    block.norm2 = nn.RMSNorm(8, eps=None)
    block.attn = nn.Module()
    block.attn.norm_q = nn.RMSNorm(4, eps=1e-6, elementwise_affine=False)
    block.attn.norm_k = nn.RMSNorm(4, eps=1e-6, elementwise_affine=False)
    block.ff = _FeedForward()
    decoder.transformer_blocks = nn.ModuleList([block])
    return decoder.eval()


@pytest.fixture
def cpu_operator_doubles(monkeypatch):
    calls: list[tuple[object, ...]] = []
    module = ModuleType("torch_npu")

    def rms_norm(x, *, gamma, epsilon):
        calls.append(("rmsnorm", x.dtype, epsilon, x.is_contiguous()))
        return F.rms_norm(x, (x.shape[-1],), gamma, epsilon), None

    def swiglu(x, dim):
        calls.append(("swiglu", x.dtype))
        gate, value = x.chunk(2, dim=dim)
        return F.silu(gate) * value

    monkeypatch.setattr(module, "npu_rms_norm", rms_norm, raising=False)
    monkeypatch.setattr(module, "npu_swiglu", swiglu, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    monkeypatch.setattr(npu_fusion, "_supported_input", lambda owner, x: True)
    return calls


@pytest.mark.parametrize("value", [None, "0", "false", "off", "invalid"])
def test_default_and_disabled_are_no_ops(monkeypatch, value):
    monkeypatch.delenv(npu_fusion.FUSION_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(npu_fusion.FUSION_ENV, value)
    decoder = _decoder()
    norm = decoder.transformer_blocks[0].norm1
    original = norm.forward.__func__
    assert not npu_fusion.install_npu_vae_fusion(decoder, device=_NPUTarget())
    assert norm.forward.__func__ is original
    assert not hasattr(norm, "_h3_npu_original_forward")


def test_cpu_target_does_not_install(monkeypatch):
    monkeypatch.setenv(npu_fusion.FUSION_ENV, "1")
    assert not npu_fusion.install_npu_vae_fusion(_decoder(), device=torch.device("cpu"))


@pytest.mark.parametrize("value", ["1", "true", "yes", " ON "])
def test_parameter_identity_state_dtype_and_idempotence(monkeypatch, value):
    monkeypatch.setenv(npu_fusion.FUSION_ENV, value)
    decoder = _decoder()
    parameters = {name: (id(p), p.dtype) for name, p in decoder.named_parameters()}
    state = {name: p.clone() for name, p in decoder.state_dict().items()}
    assert npu_fusion.install_npu_vae_fusion(decoder, device=_NPUTarget())
    saved = decoder.transformer_blocks[0].ff.forward
    assert npu_fusion.install_npu_vae_fusion(decoder, device=_NPUTarget())
    assert decoder.transformer_blocks[0].ff.forward is saved
    assert parameters == {name: (id(p), p.dtype) for name, p in decoder.named_parameters()}
    assert state.keys() == decoder.state_dict().keys()
    for name, expected in state.items():
        torch.testing.assert_close(decoder.state_dict()[name], expected, rtol=0, atol=0)


def test_non_affine_gamma_is_registered_before_residency(monkeypatch, cpu_operator_doubles):
    monkeypatch.setenv(npu_fusion.FUSION_ENV, "1")
    decoder = _decoder()
    npu_fusion.install_npu_vae_fusion(decoder, device=_NPUTarget())
    norm = decoder.transformer_blocks[0].attn.norm_q
    gamma = dict(norm.named_buffers())["_h3_npu_unit_gamma"]
    assert "_h3_npu_unit_gamma" not in norm.state_dict()
    x = torch.randn(2, 3, 4)
    with torch.inference_mode():
        expected = F.rms_norm(x, (4,), None, 1e-6)
        torch.testing.assert_close(norm(x), expected)
        norm(x)
    assert norm._h3_npu_unit_gamma is gamma


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_norm_epsilon_noncontiguous_and_gate_order(monkeypatch, cpu_operator_doubles, dtype):
    monkeypatch.setenv(npu_fusion.FUSION_ENV, "1")
    torch.manual_seed(42)
    decoder = _decoder().to(dtype=dtype)
    block = decoder.transformer_blocks[0]
    x = torch.randn(2, 3, 16, dtype=dtype)[..., ::2]
    assert not x.is_contiguous()
    expected_norm = block.norm1(x)
    expected_ff = block.ff(x)
    npu_fusion.install_npu_vae_fusion(decoder, device=_NPUTarget())
    with torch.inference_mode():
        torch.testing.assert_close(block.norm1(x), expected_norm)
        torch.testing.assert_close(block.ff(x), expected_ff)
    assert cpu_operator_doubles[0] == ("rmsnorm", dtype, 1e-5, True)
    assert cpu_operator_doubles[1] == ("swiglu", dtype)


def test_cpu_fallback_keeps_forward_and_gradient(monkeypatch):
    monkeypatch.setenv(npu_fusion.FUSION_ENV, "1")
    block = _decoder().transformer_blocks[0]
    x = torch.randn(2, 3, 8, requires_grad=True)
    expected = block.norm1(x) + block.ff(x)
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
    decoder = nn.Module()
    decoder.transformer_blocks = nn.ModuleList([block])
    npu_fusion.install_npu_vae_fusion(decoder, device=_NPUTarget())
    actual = block.norm1(x) + block.ff(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], expected_gradient, rtol=0, atol=0)


def test_compiled_ff_and_existing_overrides_are_not_replaced(monkeypatch):
    monkeypatch.setenv(npu_fusion.FUSION_ENV, "1")
    decoder = _decoder()
    block = decoder.transformer_blocks[0]
    block.ff._compile_forward_enabled = True
    block.norm1.forward = MethodType(lambda self, x: x, block.norm1)
    norm_forward, ff_forward = block.norm1.forward, block.ff.forward.__func__
    npu_fusion.install_npu_vae_fusion(decoder, device=_NPUTarget())
    assert block.norm1.forward is norm_forward
    assert block.ff.forward.__func__ is ff_forward


def test_selected_kernel_failure_is_not_hidden(monkeypatch, cpu_operator_doubles):
    monkeypatch.setenv(npu_fusion.FUSION_ENV, "1")
    decoder = _decoder()
    npu_fusion.install_npu_vae_fusion(decoder, device=_NPUTarget())

    def fail(*args, **kwargs):
        raise RuntimeError("test NPU kernel failure")

    monkeypatch.setattr(sys.modules["torch_npu"], "npu_rms_norm", fail)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="test NPU kernel failure"):
        decoder.transformer_blocks[0].norm1(torch.randn(2, 3, 8))


def test_empty_or_replacement_decoder_is_not_patched(monkeypatch):
    monkeypatch.setenv(npu_fusion.FUSION_ENV, "1")
    assert not npu_fusion.install_npu_vae_fusion(nn.Identity(), device=_NPUTarget())


def test_dynamic_epsilon_keeps_the_native_implementation(monkeypatch):
    monkeypatch.setenv(npu_fusion.FUSION_ENV, "1")
    decoder = _decoder().half()
    norm = decoder.transformer_blocks[0].norm2
    original = norm.forward.__func__
    x = torch.randn(2, 3, 8, dtype=torch.float16)
    expected = norm(x)
    npu_fusion.install_npu_vae_fusion(decoder, device=_NPUTarget())
    assert norm.forward.__func__ is original
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)
