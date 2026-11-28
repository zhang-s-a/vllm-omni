# MiniMax-H3 native-precision video VAE operators on Ascend (v0.32)

This release-branch opt-in reproduces the earlier A2 RMSNorm/SwiGLU operator
replacement for the native video decoder, including A5 targets:

```bash
export VLLM_OMNI_MINIMAX_H3_VAE_NPU_FUSION=1
```

The default is `0`. `1`, `true`, `yes` and `on` enable it, ignoring case and
surrounding spaces. Set it before starting the existing service and restart
after changing it. It is independent of the audio Conv1d opt-in.

The installer targets block norm1/norm2, Attention norm_q/norm_k, and eligible
gated SiLU FFNs. It uses `torch_npu.npu_rms_norm` and `torch_npu.npu_swiglu`.
These are separate operators: RMSNorm is not fused together with SwiGLU.
There is no W8A8 conversion, fused quantized MatMul, extra MindIE-SD operator,
RoPE replacement, change to DiT/attention scheduling, or weight dtype change.

Original parameters and state-dict keys are retained. Non-affine norms use a
nonpersistent all-ones buffer registered before residency staging. The original
explicit epsilon, FP32 normalization input and autocast remain in force.
Norms with `eps=None` keep their native forward rather than guessing the
backend's accumulation-dtype-dependent default epsilon.
CPU, training, gradients, compilation, unsupported inputs and
pre-existing instance forward overrides retain their original forwards.
Changing the switch requires recreating the VAE; it does not unpatch a live one.

Installation happens after loading and before temporal patches and residency
snapshots. Startup counts show installed modules, not proof of execution.
For A5 validation, profile a warmed native VAE decode and check RMSNorm/SwiGLU
dispatch and input dtypes; compare the same fixed latent with the switch off
and on. A replacement decoder such as TAEH3 bypasses these native blocks.

CPU regression tests (matching v0.32 development dependencies):

```bash
pytest -q tests/diffusion/models/minimax_h3/test_vae_npu_fusion_opt_in.py
```

The port has CPU contract validation only. A5 numerical behavior, video
quality, end-to-end latency and peak memory still require device validation.
Floating-point kernel changes are not claimed to be bit-identical, and no
historical A2 result is claimed as an A5 benchmark of this branch.

This can be combined with DiT-only MXFP8:

```bash
--diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}'
```

That configuration selects the transformer only; it does not quantize the
native audio/video VAE or disable this VAE opt-in.
