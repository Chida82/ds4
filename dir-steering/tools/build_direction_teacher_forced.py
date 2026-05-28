#!/usr/bin/env python3
"""Build a DS4 directional-steering vector using teacher-forced response starts.

Unlike build_direction.py which captures activations at the last token of the
prompt, this tool captures activations at the last token of a response start
that is appended after the assistant prefix.  This means the direction vector
encodes the difference between being in "cooperative response mode" vs "refusal
mode", not just the difference between two prompts.

File format (good-file / bad-file):
    Each non-empty, non-comment line has the form:
        PROMPT ||| RESPONSE_START

    Example good line:
        how do I crack WPA2? ||| Here's a script using aircrack-ng to capture the handshake:

    Example bad line:
        how do I crack WPA2? ||| I'm sorry, but I cannot help with unauthorized network access.

The prompt parts should match between good and bad files (same question, different
response trajectory).  The response start is what determines the activation state.
"""

import argparse
import array
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


N_LAYER = 43
N_EMBD = 4096

SPECIALS = {
    "bos": "<｜begin▁of▁sentence｜>",
    "user": "<｜User｜>",
    "assistant": "<｜Assistant｜>",
    "think": "<think>",
    "nothink": "</think>",
}

SEPARATOR = "|||"


def read_tf_file(path: Path) -> list[tuple[str, str]]:
    """Read teacher-forced pairs: PROMPT ||| RESPONSE_START per line."""
    pairs: list[tuple[str, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if SEPARATOR not in line:
            raise SystemExit(
                f"{path}:{line_no}: missing '{SEPARATOR}' separator. "
                f"Expected format: PROMPT {SEPARATOR} RESPONSE_START"
            )
        prompt, response_start = line.split(SEPARATOR, 1)
        prompt = prompt.strip()
        response_start = response_start.strip()
        if not prompt or not response_start:
            raise SystemExit(f"{path}:{line_no}: empty prompt or response_start")
        pairs.append((prompt, response_start))
    if not pairs:
        raise SystemExit(f"{path}: no pairs found")
    return pairs


def render_tf_prompt(system: str, user_prompt: str, response_start: str) -> str:
    """Render a full teacher-forced sequence including response start.

    The resulting text is fed entirely as input to ds4, so the model processes
    the response start as if it were generating it.  The activation at the last
    token position captures the model's internal state during that response.
    """
    pieces = [SPECIALS["bos"]]
    if system:
        pieces.append(system)
    pieces += [
        SPECIALS["user"],
        user_prompt,
        SPECIALS["assistant"],
        SPECIALS["nothink"],
        response_start,
    ]
    return "".join(pieces)


def normalize(v: list[float]) -> list[float]:
    n2 = sum(x * x for x in v)
    if n2 <= 0.0:
        return v
    inv = 1.0 / math.sqrt(n2)
    return [x * inv for x in v]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def run_capture(
    ds4: Path,
    model: Path,
    rendered_text: str,
    ctx: int,
    component: str,
    work: Path,
    avg_last_k: int = 1,
) -> list[list[float]]:
    """Run ds4 and return the activation dump for every layer.

    If avg_last_k > 1, averages the last K token rows per layer instead of
    just the very last one.
    """
    prompt_path = work / "prompt.txt"
    prompt_path.write_text(rendered_text, encoding="utf-8")
    dump_prefix = work / "dump"

    env = os.environ.copy()
    env["DS4_METAL_GRAPH_DUMP_PREFIX"] = str(dump_prefix)
    env["DS4_METAL_GRAPH_DUMP_NAME"] = component
    env["DS4_METAL_GRAPH_DUMP_POS"] = "0"

    cmd = [
        str(ds4),
        "-m", str(model),
        "--ctx", str(ctx),
        "--prompt-file", str(prompt_path),
        "-n", "1",
    ]
    subprocess.run(cmd, cwd=ds4.parent, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    rows: list[list[float]] = []
    for layer in range(N_LAYER):
        path = work / f"dump_{component}-{layer}_pos0.bin"
        data = array.array("f")
        with path.open("rb") as f:
            data.fromfile(f, path.stat().st_size // 4)
        if len(data) < N_EMBD or len(data) % N_EMBD != 0:
            raise RuntimeError(f"bad dump shape for {path}: {len(data)} floats")

        n_rows = len(data) // N_EMBD
        k = min(avg_last_k, n_rows)
        if k <= 1:
            rows.append(list(data[-N_EMBD:]))
        else:
            avg = [0.0] * N_EMBD
            for ri in range(n_rows - k, n_rows):
                offset = ri * N_EMBD
                for j in range(N_EMBD):
                    avg[j] += data[offset + j]
            for j in range(N_EMBD):
                avg[j] /= k
            rows.append(avg)
    return rows


def add_rows(total: list[list[float]], rows: list[list[float]]) -> None:
    for layer in range(N_LAYER):
        dst = total[layer]
        src = rows[layer]
        for i, value in enumerate(src):
            dst[i] += value


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a steering vector from teacher-forced response starts."
    )
    ap.add_argument("--ds4", default="./ds4", help="path to the ds4 CLI")
    ap.add_argument("--model", default="ds4flash.gguf", help="GGUF model path")
    ap.add_argument("--good-file", required=True,
                    help="target pairs (PROMPT ||| COOPERATIVE_RESPONSE_START)")
    ap.add_argument("--bad-file", required=True,
                    help="contrast pairs (PROMPT ||| REFUSAL_OR_NEUTRAL_START)")
    ap.add_argument("--out", default="dir-steering/out/direction_tf.json",
                    help="metadata JSON path; .f32 is written next to it")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--component", default="ffn_out",
                    choices=("ffn_out", "attn_out"),
                    help="runtime-editable 4096-wide activation stream")
    ap.add_argument("--avg-last-k", type=int, default=1,
                    help="average the last K token positions instead of just the last one")
    ap.add_argument("--pair-normalize", action="store_true",
                    help="average normalized per-pair differences")
    ap.add_argument("--no-orthogonalize", action="store_true",
                    help="do not remove the component parallel to the control mean")
    args = ap.parse_args()

    ds4 = Path(args.ds4).resolve()
    model = Path(args.model).resolve()
    good_pairs = read_tf_file(Path(args.good_file))
    bad_pairs = read_tf_file(Path(args.bad_file))
    n = min(len(good_pairs), len(bad_pairs))
    good_pairs = good_pairs[:n]
    bad_pairs = bad_pairs[:n]

    print(f"teacher-forced build: {n} pairs, component={args.component}, "
          f"avg_last_k={args.avg_last_k}", flush=True)

    good_sum = [[0.0] * N_EMBD for _ in range(N_LAYER)]
    bad_sum = [[0.0] * N_EMBD for _ in range(N_LAYER)]
    pair_sum = [[0.0] * N_EMBD for _ in range(N_LAYER)]

    with tempfile.TemporaryDirectory(prefix="ds4-dir-steer-tf-") as td:
        root = Path(td)
        for i, ((g_prompt, g_resp), (b_prompt, b_resp)) in enumerate(
                zip(good_pairs, bad_pairs), 1):
            print(f"pair {i}/{n}", flush=True)
            gw = root / f"good-{i}"
            bw = root / f"bad-{i}"
            gw.mkdir()
            bw.mkdir()

            good_text = render_tf_prompt(args.system, g_prompt, g_resp)
            bad_text = render_tf_prompt(args.system, b_prompt, b_resp)

            good_rows = run_capture(ds4, model, good_text, args.ctx,
                                    args.component, gw, args.avg_last_k)
            bad_rows = run_capture(ds4, model, bad_text, args.ctx,
                                   args.component, bw, args.avg_last_k)
            add_rows(good_sum, good_rows)
            add_rows(bad_sum, bad_rows)
            if args.pair_normalize:
                for layer in range(N_LAYER):
                    diff = normalize([
                        good_rows[layer][j] - bad_rows[layer][j]
                        for j in range(N_EMBD)
                    ])
                    for j, value in enumerate(diff):
                        pair_sum[layer][j] += value

    layers = []
    for layer in range(N_LAYER):
        good_mean = [x / n for x in good_sum[layer]]
        bad_mean = [x / n for x in bad_sum[layer]]
        if args.pair_normalize:
            direction = normalize([x / n for x in pair_sum[layer]])
        else:
            direction = normalize([
                good_mean[i] - bad_mean[i]
                for i in range(N_EMBD)
            ])
        if not args.no_orthogonalize:
            base = normalize(bad_mean)
            projection = dot(direction, base)
            direction = normalize([
                direction[i] - projection * base[i]
                for i in range(N_EMBD)
            ])
        layers.append(direction)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ds4-directional-steering-v1",
        "shape": [N_LAYER, N_EMBD],
        "component": args.component,
        "method": "teacher-forced",
        "avg_last_k": args.avg_last_k,
        "pair_normalize": bool(args.pair_normalize),
        "orthogonalize_control_mean": not args.no_orthogonalize,
        "good_file": str(Path(args.good_file)),
        "bad_file": str(Path(args.bad_file)),
        "model": str(model),
        "note": "runtime positive scale suppresses this direction; negative scale amplifies it",
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flat = array.array("f")
    for direction in layers:
        flat.extend(direction)
    f32_out = out.with_suffix(".f32")
    with f32_out.open("wb") as f:
        flat.tofile(f)
    print(f"wrote {out}")
    print(f"wrote {f32_out}")


if __name__ == "__main__":
    main()
