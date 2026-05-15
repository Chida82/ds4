# DS4 Steering Levers

This document explains the two runtime steering mechanisms now available in
DS4, what each one is meant to control, and how to pass the steering files to
`ds4` or `ds4-server` either individually or together.

## Why Two Levers

DS4 now exposes two different intervention points:

- directional steering edits a layer activation along a learned 4096-wide
  direction;
- expert steering edits routed MoE selection by promoting or suppressing
  specific experts in the router.

They are intentionally separate because they act on different parts of the
model:

- directional steering is a broad latent edit and is useful for behavior,
  tone, verbosity, topic emphasis, or concept suppression;
- expert steering is a router-time intervention and is useful when the desired
  behavior seems tied to recurring expert usage patterns.

In practice, directional steering is the higher-level semantic lever, while
expert steering is the more structural routed-MoE lever.

## What Each Lever Does

### Directional Steering

Directional steering uses a flat `43 x 4096` `.f32` file. Each row is one layer
direction. At runtime DS4 applies the edit after attention output, after FFN
output, or both.

Typical use cases:

- shorter vs longer answers;
- more or less explanatory style;
- concept amplification or suppression;
- coarse style transfer without fine-tuning.

Runtime flags:

```text
--dir-steering-file FILE
--dir-steering-ffn F
--dir-steering-attn F
```

See [README.md](README.md) for vector building and scale suggestions.

### Expert Steering

Expert steering uses a flat `43 x 256` `.f32` file. Each entry is normally one
of:

- `+1`: promote that expert;
- `-1`: suppress that expert;
- `0`: leave the router untouched.

The file is built by comparing paired prompt sets and measuring per-layer,
per-expert activation differences. DS4 then applies the map at inference time
inside the routed MoE path.

Typical use cases:

- bias the model toward a behavior associated with specific experts;
- reduce a contrasting behavior by suppressing the experts that correlate with
  it;
- combine router steering with an activation-space direction for stronger or
  more stable control.

Runtime flags:

```text
--expert-steering-file FILE
--expert-steering-scale F
```

See [EXPERT_STEERING.md](EXPERT_STEERING.md) for the builder, map format, and
expert-selection details.

## Single Lever Usage With `ds4`

### Directional Steering Only

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain why databases use indexes."
```

This passes only the direction file. The model follows the normal router path,
while FFN activations are steered by the requested scale.

### Expert Steering Only

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale 0.01 \
  -p "Explain why databases use indexes."
```

This passes only the expert map. No activation direction is applied, but the
routed MoE router is modified according to the `43 x 256` map.

## Double Lever Usage With `ds4`

Both files can be passed together. This is the intended setup when you want one
lever to act in activation space and the other to act on expert selection.

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale 0.01 \
  -p "Explain why databases use indexes."
```

This configuration means:

- the directional file pushes the hidden state toward the desired verbosity
  direction;
- the expert file biases the MoE router toward experts associated with the same
  target behavior.

The two levers are independent. You can disable one without changing the other
simply by omitting its file or by setting its runtime scale to zero.

## Single Lever Usage With `ds4-server`

Server usage is the same idea, but flags are passed at process startup.

### Directional Steering Only

```sh
./ds4-server --ctx 100000 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1
```

### Expert Steering Only

```sh
./ds4-server --ctx 100000 \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale 0.01
```

## Double Lever Usage With `ds4-server`

```sh
./ds4-server --ctx 100000 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  --expert-steering-file dir-steering/out/verbosity-experts.f32 \
  --expert-steering-scale 0.01
```

Once the server starts, every request on that process uses the configured
steering setup.

## How To Think About The Two Levers

Use only directional steering when:

- the behavior is well represented by a broad latent direction;
- you want a simpler experiment with one file and one or two scales;
- you are exploring style/verbosity/topic control first.

Use only expert steering when:

- the behavior appears to be strongly tied to routed expert choice;
- you want to test a SteerMoE-style intervention directly;
- you want a sparse map instead of a dense activation direction.

Use both together when:

- one lever alone is too weak;
- you want the router and the hidden state to push in the same direction;
- you want to combine a semantic control signal with a routed-MoE control
  signal.

## Development Scope

This development adds a full expert-steering path on top of the existing
directional steering stack.

Implemented scope:

- runtime flags in `ds4` and `ds4-server`;
- CPU support;
- Metal support;
- builder script that extracts router activation statistics from paired prompt
  files;
- documentation for building and using the new map.

Current backend note:

- CPU and Metal support the runtime feature;
- CUDA currently rejects expert steering explicitly instead of silently doing
  the wrong thing.

## Recommended Workflow

1. Build a directional vector if you want a broad semantic/style lever.
2. Build an expert map if you want a router-level lever.
3. Test each lever separately.
4. Combine them only after you know the individual effect of each file.
5. Start from small scales and increase only if the model remains stable.

For the bundled verbosity example, the natural pairing is:

- directional file: `dir-steering/out/verbosity.f32`
- expert file: `dir-steering/out/verbosity-experts.f32`

This lets DS4 control verbosity both through activation editing and through MoE
expert preference.