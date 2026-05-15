# Expert Steering

Expert steering is a second runtime steering lever for DS4's routed MoE FFN.
Directional steering edits 4096-wide activations after attention or FFN output;
expert steering edits the router decision itself by promoting or suppressing
behavior-associated experts.

The method follows the SteerMoE idea: compare expert activation rates between
paired prompt sets, compute a risk difference for each `(layer, expert)`, then
modify routing at inference time. DS4 adapts the paper to DeepSeek V4 Flash's
router, which uses `sqrt(softplus(logit))` scores and top-6 routed experts.
For promoted experts, DS4 sets the router score just above the current maximum
by a small margin. For suppressed experts, DS4 drops the score below the current
selection minimum and sets its route weight source to zero.

## Runtime Options

```text
--expert-steering-file FILE   load a 43 x 256 f32 expert steering map
--expert-steering-scale F     promote/suppress margin; default with file: 0.01
```

The `.f32` file is a flat `43 x 256` matrix. Positive entries promote experts,
negative entries suppress experts, and zeros leave normal routing untouched.
A negative runtime scale reverses the map, so a target-behavior map can be used
in either direction.

Expert steering is independent from directional steering and can be combined:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale 0.01 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain why databases use indexes."
```

The first three DS4 layers use token-id hash routing. The builder skips them by
default (`--layers 3-42`) because activation-style expert promotion is only a
true router intervention on score-routed layers. Runtime suppression can still
reduce the weights of hash-selected experts if a nonzero entry is present, but
the recommended path is to leave layers `0-2` unset.

## Building A Map

Use paired prompt files with one prompt per non-empty line. The first file is
the target behavior; the second file is the contrast behavior.

```sh
python3 dir-steering/tools/build_expert_steering.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out dir-steering/out/verbosity-experts.json \
  --ctx 512 \
  --activate-per-layer 2 \
  --deactivate-per-layer 2
```

This writes:

```text
dir-steering/out/verbosity-experts.json
dir-steering/out/verbosity-experts.f32
```

The JSON contains the risk-difference report and selected experts. The `.f32`
file is the runtime map passed to `ds4` or `ds4-server`.

## Trying Scales

Start small. For most experiments, use `0.005`, `0.01`, and `0.02` before going
larger. Because the intervention guarantees promoted experts enter above the
current score maximum, bigger values are not usually necessary.

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 180 \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale 0.01 \
  -p "Explain what DNS does."
```

Reverse the same map with a negative scale:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 180 \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale -0.01 \
  -p "Explain what DNS does."
```

For server use, pass the same flags at startup:

```sh
./ds4-server --ctx 100000 \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale 0.01
```

## Notes

- The builder currently captures routing through the Metal graph dump hook.
  Runtime application works on CPU and Metal.
- The generated map is sparse by design: only the top positive and negative RD
  experts per enabled layer are nonzero.
- If output becomes repetitive, loses factual content, or ignores the prompt,
  lower the scale or reduce `--activate-per-layer` / `--deactivate-per-layer`.