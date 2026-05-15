# Expert Steering (SteerMoE) — User Guide

DS4 ships **two orthogonal steering levers** that you can use independently or
together to shape the model's behaviour without retraining or LoRA:

| Lever                     | Where it acts                       | What it modifies                                                                 | CLI                                                          |
|---------------------------|-------------------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------|
| **Directional steering**  | Residual stream (per layer)         | Removes / adds a learned direction `v_l` from activations after attention/FFN.   | `--dir-steering-file F.f32 --dir-steering-attn A --dir-steering-ffn B` |
| **Expert steering** (new) | Routed-MoE selection score          | Adds a per-`(layer, expert)` bias *before* top-k, biasing which experts are picked.| `--expert-steering-file F.f32 --expert-steering-scale S`     |

The two operate at completely different points of the network so they compose
cleanly:

```sh
./ds4 -m gguf/<model> \
      --dir-steering-file dir-steering/out/persona.f32  --dir-steering-ffn -1 \
      --expert-steering-file expert-steering/out/verbosity.f32 --expert-steering-scale 1e6 \
      -p "Explain why databases use indexes."
```

This document covers the **expert** lever.  See
[`dir-steering/README.md`](dir-steering/README.md) for the directional lever.

---

## 1. What SteerMoE does (paper recap)

The DeepSeek V4 Flash routed-MoE block produces, for every token, a vector of
256 router probabilities and picks the top-6 experts.  Fayyaz et al. ("Steering
MoE LLMs via Expert (De)Activation", Adobe Research) show that you can shape
the model's behaviour by *biasing the router selection score* before that
top-k cut, while still using the unbiased probabilities as the down-stream
weights ("soft" rule).  Large positive biases force an expert ON (A+); large
negative biases force it OFF (A-).

DS4 implements the additive-bias rule **on both the CPU and Metal backends**:

```
selection[e] = router_probs[e]
             + (model->ffn_exp_probs_b ? model_bias[e] : 0)
             + (steering ? scale * steering_bias[layer, e] : 0)
selected     = top-k(selection, 6)
weights      = router_probs[selected] normalized
```

The first `DS4_N_HASH_LAYER = 3` layers use a fixed `token_id -> expert_id`
table (hash routing), so SteerMoE only applies to layers 3..42.  The CUDA
backend currently warns and ignores expert steering (router_select on CUDA is
not yet wired through; the CPU and Metal paths are the production ones for
the Apple Silicon target).

## 2. File format

A flat little-endian binary of exactly `DS4_N_LAYER * DS4_N_EXPERT = 43 * 256
= 11008` `float32` values, indexed as `bias[layer * 256 + expert]`.  Total
size: **44032 bytes**.

The engine pre-multiplies every entry by `--expert-steering-scale` at load
time.  Two common modes:

* **Soft steering** — `--expert-steering-scale 1`, file values in roughly
  `[-1, +1]`.  Nudges the router but does not override it.
* **Hard force-on / force-off** (paper's A+/A- variant) — store ±1 in the file
  and pass `--expert-steering-scale 1e6`.  The bias dominates the router score
  so the chosen experts effectively flip.

## 3. CLI reference

Both `ds4` and `ds4-server` accept:

```
--expert-steering-file FILE
    Load a per-(layer, expert) f32 bias for the routed-MoE selection score.

--expert-steering-scale F
    Multiplier on the file values. Soft steering ~1; hard force-on/off ~1e6.
    Default with file: 1
```

If you pass `--expert-steering-scale` without a file, startup aborts with a
clear message.  If you pass the file alone, scale defaults to `1.0`.

## 4. Building a bias file

Use [`expert-steering/tools/build_expert_steering.py`](expert-steering/tools/build_expert_steering.py)
with a *contrast pair* of prompt files — one embodying the trait you want to
**promote** (`--pos-file`), one embodying the trait you want to **suppress**
(`--neg-file`).  The tool runs `ds4` twice with the Metal graph dump
infrastructure to capture the actually-selected top-k experts at every routed
layer, computes the per-expert risk difference

```
Δ[l, e] = p_pos[l, e] - p_neg[l, e]
```

and writes, **per layer**:

* the top-N experts with the largest positive `Δ` -> `+|Δ|` (or `+1` with
  `--strict`) — these will be force-ON when the bias is applied with positive
  scale;
* the top-N with the largest negative `Δ` -> `-|Δ|` (or `-1` with `--strict`)
  — these will be force-OFF.

Hash-routed layers (0..2) are left at zero.

### Example: succinct vs verbose

The DS4 directional-steering examples already include the prompt pair we need:

```sh
python3 expert-steering/tools/build_expert_steering.py \
    --ds4 ./ds4 \
    --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --pos-file dir-steering/examples/succinct.txt \
    --neg-file dir-steering/examples/verbose.txt \
    --out expert-steering/out/succinct.f32 \
    --top-n 4 --strict
```

Use it:

```sh
# Strong push toward "succinct" routing
./ds4 -m gguf/<model> \
      --expert-steering-file expert-steering/out/succinct.f32 \
      --expert-steering-scale 1e6 \
      -p "Explain why databases use indexes."

# Strong push toward "verbose" — flip the sign
./ds4 -m gguf/<model> \
      --expert-steering-file expert-steering/out/succinct.f32 \
      --expert-steering-scale -1e6 \
      -p "Explain why databases use indexes."

# Soft mix with directional steering (independent levers)
./ds4 -m gguf/<model> \
      --dir-steering-file dir-steering/out/persona.f32 --dir-steering-ffn -0.5 \
      --expert-steering-file expert-steering/out/succinct.f32 --expert-steering-scale 5 \
      -p "Explain why databases use indexes."
```

## 5. How to interpret the bias

`Δ[l, e] > 0` means experts that fire **more often** when the model produces
"pos"-style outputs than "neg"-style outputs.  Force them ON to push toward
"pos".  `Δ < 0` is the opposite — experts associated with "neg"; force them
OFF to suppress "neg".

The `--strict` mode mirrors the paper's hard A+/A- protocol: you only carry
*which* experts to flip, not *by how much*, and rely on a very large scale to
override the router.  Without `--strict` the magnitude `|Δ|` is preserved,
giving a soft, frequency-weighted steering signal that mixes well with low
scales.

## 6. Verification

After building, two quick sanity checks should hold:

1. **Zero bias is a no-op.** Generating with a file of all-zeros and any
   scale must produce the **exact same tokens** as generating without the
   flag (same seed, prompt, etc.).
2. **Non-zero bias changes routing.** A small random bias should perturb the
   output noticeably; with `scale 1e6` and a binary ±1 file the output should
   change a lot.

Both are verified by the smoke tests in `make test`.

## 7. References

* Fayyaz et al., *Steering MoE LLMs via Expert (De)Activation*, Adobe Research,
  2024 — <https://github.com/adobe-research/SteerMoE>.
* DS4 directional steering: [`dir-steering/README.md`](dir-steering/README.md).
