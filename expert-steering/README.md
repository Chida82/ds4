# Expert Steering (SteerMoE) for DS4

This directory hosts the tooling needed to **build** SteerMoE bias files for
the DS4 (DeepSeek V4 Flash) inference engine.  See
[../EXPERT_STEERING.md](../EXPERT_STEERING.md) for the user guide describing
how to *use* the resulting files at runtime.

## Background

DS4 implements two orthogonal steering levers:

1. **Directional steering** (existing, see [`../dir-steering/`](../dir-steering/)) —
   surgery at the **activation level**.  A learned direction `v_l` is removed
   from (or added to) the residual stream after attention and/or FFN of every
   layer.  This shifts the model's representation in a continuous subspace.

2. **Expert steering** (this directory, **new**, "SteerMoE") — surgery at the
   **routing level**.  Per `(layer, expert)` we add a bias to the routed-MoE
   selection score *before* the top-k cut, while the down-stream weighting
   keeps the unbiased router probabilities (the "soft" rule from the paper).
   Large positive biases force an expert ON (A+); large negative biases force
   it OFF (A-).

Both levers can be enabled together:

```sh
./ds4 -m gguf/<model> \
      --dir-steering-file dir-steering/out/persona.f32 --dir-steering-ffn -1 \
      --expert-steering-file expert-steering/out/verbosity.f32 --expert-steering-scale 1e6 \
      -p "Explain why databases use indexes."
```

## File format

A single flat binary, **little-endian**, of exactly `DS4_N_LAYER * DS4_N_EXPERT
= 43 * 256 = 11008` `float32` entries (= 44032 bytes).

Index layout: `bias[layer * 256 + expert]`.

* The first `DS4_N_HASH_LAYER = 3` layers use hash-routed experts (a fixed
  `token_id -> expert_id` table) and **are unaffected** by SteerMoE — those
  256-wide slices may be left at zero.
* At runtime the engine pre-multiplies every entry by `--expert-steering-scale`,
  so a stored bias of `+1` together with `--expert-steering-scale 1e6` produces
  the paper's hard force-on (A+); a stored value of `-0.05` together with
  `--expert-steering-scale 1` produces a soft suppression.

## Workflow

The end-to-end pipeline mirrors the directional-steering workflow:

1. Pick a *contrast pair*: two prompt files that differ only along the trait
   you want to control (verbose vs succinct, refusal vs comply, ...).  The
   verbose/succinct pair from `../dir-steering/examples/` is a good starting
   point.
2. Run [`tools/build_expert_steering.py`](tools/build_expert_steering.py) to
   produce the `.f32` file.  Internally the tool runs `ds4` twice with
   `DS4_METAL_GRAPH_DUMP_*` enabled to capture the **actually selected** top-k
   experts at every routed layer for both prompt sets, computes the per-expert
   risk difference `Δ[l, e] = p_pos[l,e] - p_neg[l,e]`, and writes:
     * `+|Δ|` (or `+1` in `--strict` mode) for the top-N most positive experts
       per layer (force these ON when steering toward "pos");
     * `-|Δ|` (or `-1` in `--strict` mode) for the top-N most negative experts
       per layer (force these OFF when steering toward "pos").
3. Pass the resulting file to `ds4` / `ds4-server` with
   `--expert-steering-file` and `--expert-steering-scale`.

A verbose vs succinct example is documented in
[../EXPERT_STEERING.md](../EXPERT_STEERING.md).

## References

* Fayyaz et al., *"Steering MoE LLMs via Expert (De)Activation"*, Adobe
  Research 2024 — <https://github.com/adobe-research/SteerMoE>.
