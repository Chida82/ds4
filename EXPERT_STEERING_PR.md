# PR: SteerMoE expert steering for DS4

This PR adds **expert-level steering** ("SteerMoE", Fayyaz et al., Adobe
Research, 2024) to the DS4 (DeepSeek V4 Flash) inference engine as a second
lever orthogonal to the existing **directional** steering.

## What it does

For every routed-MoE layer, the engine now optionally adds a per-`(layer, expert)`
bias to the router selection score *before* the top-k cut, while the
down-stream weighting keeps the unbiased router probabilities — the paper's
"soft" rule.

```
selection[e] = router_probs[e]
             + (model has bias  ? model_bias[e]               : 0)
             + (steering loaded ? scale * steering_bias[l, e] : 0)
selected     = top-k(selection, 6)
```

The bias is a flat little-endian binary of `DS4_N_LAYER * DS4_N_EXPERT`
(`43 * 256 = 11008`) `float32` values.  The first `DS4_N_HASH_LAYER = 3`
hash-routed layers are unaffected.  The user controls magnitude with
`--expert-steering-scale` (≈1 for soft steering, ≈1e6 for hard force-on/off).

## Backends

| Backend | Status                                               |
|---------|------------------------------------------------------|
| CPU     | Implemented in `layer_topk_selected_experts*`.        |
| Metal   | Implemented in the fused `kernel_dsv4_router_finalize_one` (single-token fast path) and via a chained Metal `add` on the selection scratch buffer (batch / non-fused path). |
| CUDA    | Accepts the new params and warns once if a non-NULL bias is supplied; the production target is Apple Silicon.|

## Surface area

* New CLI flags on `ds4` and `ds4-server`:
  * `--expert-steering-file FILE`
  * `--expert-steering-scale F` (default 1 when file is given)
* `ds4_engine_options` extended with `expert_steering_file` /
  `expert_steering_scale`.
* `ds4_gpu_router_select_tensor` and `ds4_gpu_router_select_batch_tensor` gain
  two trailing parameters: `(const ds4_gpu_tensor *expert_steering_bias, uint32_t expert_steering_layer)`.
  Existing call sites in `ds4.c` pass them through; CPU/CUDA/Metal stubs
  updated.
* New private helper `ds4_engine_load_expert_steering()` reads the file and
  pre-multiplies by the user scale at engine open.
* New Metal kernel arg `has_steering` plus `device const float *steering [[buffer(6)]]`
  on `kernel_dsv4_router_finalize_one`.

Files touched: `ds4.h`, `ds4.c`, `ds4_gpu.h`, `ds4_metal.m`, `ds4_cuda.cu`,
`metal/dsv4_misc.metal`, `ds4_cli.c`, `ds4_server.c`.

## New deliverables

* [`expert-steering/README.md`](expert-steering/README.md) — design and workflow overview.
* [`expert-steering/tools/build_expert_steering.py`](expert-steering/tools/build_expert_steering.py)
  — Python tool that uses the existing Metal `DS4_METAL_GRAPH_DUMP_*` hooks to
  capture top-k selections for two contrast prompt sets, computes per-expert
  risk differences, and writes the `.f32` bias file.
* [`EXPERT_STEERING.md`](EXPERT_STEERING.md) — user guide explaining the
  dual-lever model, file format, CLI flags, build/use workflow, and how to
  combine with directional steering.
* [`EXPERT_STEERING_PR.md`](EXPERT_STEERING_PR.md) — this document.

## Tests

* `make` builds `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval` cleanly with no
  new warnings.
* `make test` passes (long-context, tool-call quality, logprob vectors, Metal
  kernels, server) — confirming the changes are non-disruptive in the default
  no-steering path.
* End-to-end smoke tests (Metal and CPU backends, single-token decode and
  prefill paths via `-p`):
  * **Zero bias** + any scale produces **bit-identical** tokens to baseline
    (no-op invariant).
  * **Random bias** with scale 1 noticeably perturbs the output text on both
    backends.

## How to use

```sh
# 1) Build a bias file from a contrast pair
python3 expert-steering/tools/build_expert_steering.py \
    --ds4 ./ds4 --model gguf/<model> \
    --pos-file dir-steering/examples/succinct.txt \
    --neg-file dir-steering/examples/verbose.txt \
    --out expert-steering/out/succinct.f32 --top-n 4 --strict

# 2) Run with hard SteerMoE
./ds4 -m gguf/<model> \
      --expert-steering-file expert-steering/out/succinct.f32 \
      --expert-steering-scale 1e6 \
      -p "Explain why databases use indexes."

# 3) Combine with directional steering (independent levers)
./ds4 -m gguf/<model> \
      --dir-steering-file dir-steering/out/persona.f32 --dir-steering-ffn -1 \
      --expert-steering-file expert-steering/out/succinct.f32 --expert-steering-scale 1e6 \
      -p "..."
```

See [`EXPERT_STEERING.md`](EXPERT_STEERING.md) for the full guide.
