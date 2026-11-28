# MiniMax-H3 TAEH3 preview decoding on Ascend (v0.32)

TAEH3 is a lossy lightweight video decoder. This opt-in ports the earlier A2
path to the native H3 adapter for A5 evaluation. Reference image/video encoding
and the audio VAE remain native; DiT, sampling and quantization are unchanged.
Generated quality can differ from the native decoder.

Download the same pinned upstream weights used by the earlier implementation:

```bash
curl -fL --retry 3 \
  https://raw.githubusercontent.com/madebyollin/taehv/e589fddc076e77f5ba8cd6baabe4ba3260b261cd/taeh3.pth \
  -o /data/taeh3.pth
echo 'af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba  /data/taeh3.pth' | sha256sum -c -
export VLLM_OMNI_MINIMAX_H3_TAEH3_PATH=/data/taeh3.pth
```

The path is the single opt-in switch: unset/empty means the original decoder;
a nonempty local checkpoint path enables TAEH3. Set it before startup and
restart after changing it. There is no automatic download. The loader verifies
the pinned SHA256 before `torch.load(weights_only=True)` and strictly loads
decoder keys. Missing/incompatible weights fail instead of silently using the
native decoder.

The adapter requires native 24-channel H3 latents and an Ascend target. TAEH3
consumes the normalized diffusion latent directly; native latent mean/std and
pixel revert must not be applied again. FP16 decoder weights are registered
before residency snapshots and included in the split decoder stager, so the
existing load/offload lifecycle owns them. The original native decoder remains
loaded for compatibility; this change does not remove its weights.

Temporal batches retain each causal MemBlock's previous input across boundaries,
with request-local state. H3's five-token groups, per-group three-frame trim and
final twelve-frame drop are preserved: 110 latent tokens produce 362 frames.
Output is FP32 `[B,C,T,H,W]` RGB in `[0,1]`. TAE runs rank-locally; native tile
parallel settings do not spatially partition its work.

This port supports the normal full-video output path. Keep
`extra_params.preencode_mp4=false` (the default); chunked MP4 decoding raises
explicitly rather than silently switching back to the native decoder. Native
RMSNorm/SwiGLU optimization may remain configured but is bypassed for output
video when TAEH3 is enabled. Native reference encoding and audio still execute.
Transformer-only MXFP8 can remain enabled independently.

CPU regression tests, with matching v0.32 development dependencies installed:

```bash
pytest -q tests/diffusion/models/minimax_h3/test_taeh3_opt_in.py
```

CPU checks cover opt-in/loading guards and causal/temporal behavior; they do
not establish A5 execution, generated-video quality, memory or latency. No A2
speedup or upstream GPU benchmark is claimed as an A5 measurement.

Architecture and license source: [madebyollin/taehv at the pinned revision](https://github.com/madebyollin/taehv/tree/e589fddc076e77f5ba8cd6baabe4ba3260b261cd).
The adapted source retains its MIT notice. No model weights are included in this PR.
