# MiniMax-H3 reference KV and active isolation on Ascend (v0.32)

This release port combines the previously used reference-KV offload, contiguous
host layout, active reference isolation and Cache-DiT hit-priority integration.
It is based on official v0.32.0 and targets evaluation with
`quay.io/ascend/vllm-omni:v0.32.0-a5`. No A5 device result is claimed.

The two original CLI options and their enable semantics are preserved:

| Options | Behavior |
| --- | --- |
| Both omitted | Original path; no reference-KV state or isolation |
| `--kv-offload-config` only | Original validated reference-KV refresh/host-quantization policy |
| `--kv-isolate` only | Active visual-reference isolation plus one-time capture, Host INT8 and two-slot asynchronous prefetch |
| Both supplied | Isolation takes precedence over the entire offload JSON policy |

Append to your existing service command for offload without isolation:

```bash
--kv-offload-config '{"mode":"on","kv_refresh_interval":16,"kv_host_quantization":"int8"}'
```

`mode` remains `on`, not the older experimental `tier1`/`tier2` names. The
original defaults are refresh interval 2, host quantization `none`, and
`cache_dit_policy=refresh`. Host quantization also accepts `int8` or `fp8`.
The original optional hit-priority policy is retained:

```bash
--kv-offload-config '{"mode":"on","kv_refresh_interval":16,"kv_host_quantization":"int8","cache_dit_policy":"hit","max_refresh_interval":32}'
```

To use active isolation instead, append:

```bash
--kv-isolate
```

This flag alone selects the existing fixed request-local policy: one fill,
pinned host storage, INT8, two staging slots and asynchronous per-layer
prefetch. Periodic refresh and hit-priority scheduling do not also run when
isolation is enabled. A supplied offload JSON policy is ignored by that mode;
CLI JSON syntax must still be valid. Omit `--kv-isolate` to disable it. It is a
boolean flag, not `--kv-isolate false`. Legacy environment switches do not
enable either feature.

After Ulysses, the cache stores the balanced K/V head shard in layer-major
host storage. Reuse steps skip reference-row projections and stage the required
layer through the device ring. State, row geometry and cache history are scoped
to the request/branch and released at the end, including error paths.

Isolation changes attention semantics: visual-reference queries read only
visual-reference K/V, while dynamic text/audio/target queries read all valid
K/V. The implementation splits those regions without allocating a quadratic
mask. The reference condition timestep remains fixed. This is experimental and
quality-changing; periodic stale-KV reuse is also not a lossless-equivalence
claim. INT8/FP8 host quantization has numerical effects and logged fallbacks.

Supported attention is noncausal, one document, local or pure Ulysses with
ring degree 1. Dense FLASH_ATTN/SDPA and the existing rectangular RainFusion
path are retained. FASTVIDEO_VSA, Ring/other SP combinations, VDN/ControlNet,
latent editing and step-execution isolation paths reject where unsupported.
This does not add a new A5 sparse kernel or change RainFusion sparsity/start
step. A5's installed MindIE-SD rectangular operator ABI still needs validation.

Transformer-only MXFP8, auxiliary residency, audio Conv1d and TAEH3 can be
configured independently. Transformer MXFP8 does not select reference-KV host
quantization. Native video-decoder fusion is bypassed by TAEH3 as documented
by that feature. Neither feature here promises lower peak HBM or faster A5
inference: quantify actual exposed waits, memory and generated quality with
matched warmed inputs and the slowest rank's timings.

With matching v0.32 development dependencies installed, CPU regression tests:

```bash
pytest -q tests/diffusion/models/minimax_h3/test_refkv_isolation_release.py
```

Additional local validation exercises the version-matched original CPU suites
and Cache-DiT 1.5.0 with CPU/AST doubles. Those checks do not establish full
service imports, A5 kernels/streams/events, distributed execution, quality,
latency or peak memory. This fork PR remains a draft pending those checks.
