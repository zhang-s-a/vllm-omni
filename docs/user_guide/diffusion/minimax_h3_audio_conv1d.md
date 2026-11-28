# MiniMax-H3 audio Conv1d on Ascend (v0.32)

This release-branch opt-in reproduces the earlier A2 audio-VAE optimization
on the native MiniMax-H3 audio decoder, including A5 targets. It does not
quantize weights or change the video decoder, DiT, sampling or offload policy.

Set the switch before starting the existing service:

```bash
export VLLM_OMNI_MINIMAX_H3_AUDIO_TCONV_CONV1D=1
```

The default is `0` (original implementation). `1`, `true`, `yes` and `on`
enable it, ignoring case and surrounding spaces. Restart the service after
changing the switch; it is read during audio-VAE initialization.

The implementation evaluates stage ConvTranspose1d and compatible checkpoint
UpSample1d filters using phase-decomposed Conv1d and interleaving. It preserves
the existing FP32 input/weight dtype, parameter objects and state-dict keys.
It is algebraically equivalent, although floating-point kernels can round
differently. CPU, training, gradients, compilation, unsupported dtypes,
grouped/dilated stage convolutions and explicit output sizes retain their
original implementation. It requires no additional MindIE-SD custom operator.

The startup log reports installed stage and resampler counts. Installation
does not prove execution: on A5, profile a warmed audio decode and confirm the
eligible transposed convolutions execute through Conv1d, then compare the same
fixed latent against the original decoder and check real audio outputs.

CPU regression tests (matching v0.32 development dependencies):

```bash
pytest -q tests/diffusion/models/minimax_h3/test_audio_conv1d_opt_in.py
```

The port has CPU validation only; A5 numerical, end-to-end, latency and peak
memory validation are still required. No A2 or upstream benchmark is claimed
as a measurement of this branch. The phase decomposition follows
[upstream PR #8813](https://github.com/vllm-project/vllm-omni/pull/8813).

This can be combined with DiT-only MXFP8:

```bash
--diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}'
```

That configuration selects the transformer only; it does not quantize the
native audio/video VAE or disable this VAE opt-in.
