# Steering Levers PR Summary

This change extends DS4 steering with a second independent runtime lever for
routed MoE expert selection.

## What This Adds

- existing directional steering remains available as a `43 x 4096` activation
  edit;
- new expert steering adds a `43 x 256` sparse routed-expert map;
- both levers can be used alone or together in `ds4` and `ds4-server`.

## Purpose

Directional steering controls hidden-state behavior after attention and/or FFN.
Expert steering controls the routed MoE decision itself by promoting or
suppressing experts associated with a target behavior.

The goal is to support two complementary control surfaces:

- a semantic activation-space lever;
- a routed-MoE expert-selection lever.

## User-Facing Flags

Directional steering:

```text
--dir-steering-file FILE
--dir-steering-ffn F
--dir-steering-attn F
```

Expert steering:

```text
--expert-steering-file FILE
--expert-steering-scale F
```

## Usage Examples

`ds4`, directional only:

```sh
./ds4 -m ds4flash.gguf \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain what DNS does."
```

`ds4`, expert only:

```sh
./ds4 -m ds4flash.gguf \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale 0.01 \
  -p "Explain what DNS does."
```

`ds4`, both levers:

```sh
./ds4 -m ds4flash.gguf \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale 0.01 \
  -p "Explain what DNS does."
```

`ds4-server`, both levers:

```sh
./ds4-server --ctx 100000 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale 0.01
```

## Implementation Notes

- expert steering follows a SteerMoE-style approach adapted to DS4's DeepSeek
  V4 Flash routed MoE;
- runtime support is implemented for CPU and Metal;
- CUDA currently reports expert steering as unsupported rather than accepting
  the flags and ignoring them;
- a builder script generates the expert map from paired prompt files.

## Validation

- Metal build passes;
- CPU build passes;
- `make test` passes;
- the new builder generates valid `.json` and `.f32` artifacts;
- runtime smoke tests confirm expert-only and dual-lever execution.

## Related Docs

- [README.md](README.md)
- [EXPERT_STEERING.md](EXPERT_STEERING.md)
- [STEERING_LEVERS.md](STEERING_LEVERS.md)