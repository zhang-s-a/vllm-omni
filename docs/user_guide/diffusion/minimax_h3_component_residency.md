# MiniMax-H3 auxiliary residency on Ascend (v0.32)

Keep the loaded text encoder and native video/audio VAEs on the NPU between
requests, while preserving the existing DiT layerwise/DLO transfer policy:

```bash
export VLLM_OMNI_MINIMAX_H3_KEEP_COMPONENTS_RESIDENT=1
```

Default: `0`. `1`, `true`, `yes` and `on` enable it, ignoring case and whitespace.
Set it before service startup; restart after changing it. This is an independent
release opt-in and does not require the audio Conv1d or VAE fusion PRs.

The switch removes auxiliary encoder-block streaming and on-demand placement
from the per-instance offload plan, loads VAEs directly onto the NPU, and skips
manual per-request auxiliary staging, including video encoder/decoder proxies.
DiT stacks, child modules, weight transfer, quantization, attention and sampling
settings are retained. It does not intercept explicit offload, sleep or teardown.
No model parameters or persistent buffers are replaced.

Supported policies are legacy layerwise/DLO, explicit layer mode selecting only
`dit`, and no offload (where this is normally already the residency behavior).
Module-level offload and explicit `text_encoder` offload conflict with the switch
and raise before loading. If using `diffusion-offload-config`, select only `dit`
instead of requesting both residency and text-encoder offload. Do not copy the
older A2 `SUBMODULE` enum: v0.32 resolves the current layerwise/DLO policies.

This saves only auxiliary weight movement and related waits, increases live
HBM usage, and does not remove VAE/encoder computation or DiT weight transfers.
No A5 latency or memory benefit is claimed. Use the same warmed inputs/config
with the switch off/on, inspect the slowest rank's exposed transfer waits, and
record actual HBM peaks. Placement alone does not establish a speedup.

CPU regression tests, with matching v0.32 development dependencies installed:

```bash
pytest -q tests/diffusion/models/minimax_h3/test_component_residency_opt_in.py
```

Validation covers the default path, current policy parsing, conflicting
selection, split proxies and preservation of the DiT plan. Actual A5 loading,
distributed execution, sleep/wake, numerical output and performance require
device validation; CPU policy tests do not establish those results.
