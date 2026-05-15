#!/usr/bin/env python3
"""Build a SteerMoE per-(layer, expert) bias file for DS4.

Pipeline
--------
For each contrast prompt set ("pos" = the trait we want to *promote*, "neg" =
the trait we want to *suppress*) we run the local ``ds4`` binary in
prompt-only / no-generation mode with the Metal graph dump infrastructure
enabled.  The dumps tell us which experts were actually picked at every routed
layer for every prompt token::

    DS4_METAL_GRAPH_DUMP_PREFIX=<dir>/run
    DS4_METAL_GRAPH_DUMP_NAME=ffn_moe_topk
    DS4_METAL_GRAPH_DUMP_LAYER=all

producing files of the form ``<prefix>_ffn_moe_topk-<layer>_pos<pos>.i32``
each containing ``n_tokens * 6`` int32 selected expert ids.

We aggregate counts per ``(layer, expert)`` independently for "pos" and "neg",
turn them into per-expert activation probabilities, compute the risk
difference

    Δ[l, e] = p_pos[l, e] - p_neg[l, e]

and pick, **per layer**, the top-N experts with the largest positive Δ as
"force ON" (positive bias) and the top-N with the largest negative Δ as
"force OFF" (negative bias).  In ``--strict`` mode the magnitudes are clamped
to ±1 (the engine multiplies by ``--expert-steering-scale`` at load time).

The first ``DS4_N_HASH_LAYER = 3`` layers use hash routing instead of top-k
routing; their slices are left at zero.

Output: a flat little-endian binary of ``DS4_N_LAYER * DS4_N_EXPERT`` float32
entries, ready to be passed to ``ds4 --expert-steering-file ... --expert-steering-scale ...``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

DS4_N_LAYER = 43
DS4_N_EXPERT = 256
DS4_N_EXPERT_USED = 6
DS4_N_HASH_LAYER = 3


def _run_dump(ds4_bin: Path, model: Path, prompts: list[str], out_dir: Path,
              ctx: int, system: str | None) -> int:
    """Run ds4 once per prompt, capturing topk dumps under ``out_dir``.

    Returns the total number of decoded prompt tokens (sum across runs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    for i, prompt in enumerate(prompts):
        prefix = out_dir / f"run{i:04d}"
        env = os.environ.copy()
        env["DS4_METAL_GRAPH_DUMP_PREFIX"] = str(prefix)
        env["DS4_METAL_GRAPH_DUMP_NAME"] = "ffn_moe_topk"
        env["DS4_METAL_GRAPH_DUMP_LAYER"] = "all"
        cmd = [str(ds4_bin), "-m", str(model), "-c", str(ctx), "-n", "1", "-p", prompt]
        if system:
            cmd += ["--system", system]
        proc = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
            raise SystemExit(f"ds4 failed for prompt #{i} (rc={proc.returncode})")

        # Each .i32 dump file holds n_tokens * 6 int32 entries.
        for path in sorted(out_dir.glob(f"run{i:04d}_ffn_moe_topk-*_pos*.i32")):
            total_tokens += path.stat().st_size // (DS4_N_EXPERT_USED * 4)
            # We tally per layer outside; here we just count tokens once.
    return total_tokens


def _accumulate(out_dir: Path, counts: list[list[int]]) -> int:
    """Add the dumped expert selections under ``out_dir`` into ``counts``.

    counts: ``DS4_N_LAYER`` lists each of length ``DS4_N_EXPERT``.
    Returns the per-layer token count (constant across layers)."""
    layer_token_count = 0
    for path in sorted(out_dir.glob("run*_ffn_moe_topk-*_pos*.i32")):
        # filename pattern: <prefix>_ffn_moe_topk-<layer>_pos<pos>.i32
        stem = path.stem  # drops .i32
        layer_str = stem.rsplit("_ffn_moe_topk-", 1)[1].split("_pos", 1)[0]
        layer = int(layer_str)
        if layer < 0 or layer >= DS4_N_LAYER:
            continue
        data = path.read_bytes()
        if len(data) % (DS4_N_EXPERT_USED * 4) != 0:
            raise SystemExit(f"corrupt dump file: {path}")
        n_tok = len(data) // (DS4_N_EXPERT_USED * 4)
        ints = struct.unpack(f"<{n_tok * DS4_N_EXPERT_USED}i", data)
        for e in ints:
            if 0 <= e < DS4_N_EXPERT:
                counts[layer][e] += 1
        if layer >= DS4_N_HASH_LAYER:
            layer_token_count = max(layer_token_count, n_tok)
    return layer_token_count


def _read_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # One prompt per non-empty, non-comment line; or, if the file is a single
    # paragraph, the whole text is one prompt.
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
    if len(lines) >= 4 and all(len(l) < 1024 for l in lines):
        return lines
    return [text]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds4", default="./ds4", type=Path,
                    help="Path to the ds4 binary (default: ./ds4)")
    ap.add_argument("--model", required=True, type=Path,
                    help="Path to the GGUF model file")
    ap.add_argument("--pos-file", required=True, type=Path,
                    help="Prompt file embodying the trait we want to PROMOTE")
    ap.add_argument("--neg-file", required=True, type=Path,
                    help="Prompt file embodying the trait we want to SUPPRESS")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output .f32 path (43*256 little-endian float32)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--system", default=None,
                    help="Optional system prompt forwarded to ds4")
    ap.add_argument("--top-n", type=int, default=4,
                    help="How many force-ON / force-OFF experts per layer (default: 4)")
    ap.add_argument("--strict", action="store_true",
                    help="Use ±1 instead of ±|Δ| (combine with --expert-steering-scale 1e6 "
                         "for the paper's hard A+/A- variant)")
    ap.add_argument("--keep-dumps", action="store_true",
                    help="Keep the temporary dump directory for inspection")
    args = ap.parse_args()

    if not args.ds4.exists():
        ap.error(f"ds4 binary not found: {args.ds4}")
    if not args.model.exists():
        ap.error(f"model not found: {args.model}")

    pos_prompts = _read_prompts(args.pos_file)
    neg_prompts = _read_prompts(args.neg_file)
    print(f"[expert-steering] {len(pos_prompts)} positive / {len(neg_prompts)} negative prompts",
          file=sys.stderr)

    work = Path(tempfile.mkdtemp(prefix="ds4-expert-steering-"))
    try:
        pos_dir = work / "pos"
        neg_dir = work / "neg"
        _run_dump(args.ds4, args.model, pos_prompts, pos_dir, args.ctx, args.system)
        _run_dump(args.ds4, args.model, neg_prompts, neg_dir, args.ctx, args.system)

        pos_counts = [[0] * DS4_N_EXPERT for _ in range(DS4_N_LAYER)]
        neg_counts = [[0] * DS4_N_EXPERT for _ in range(DS4_N_LAYER)]
        _accumulate(pos_dir, pos_counts)
        _accumulate(neg_dir, neg_counts)

        bias = [0.0] * (DS4_N_LAYER * DS4_N_EXPERT)
        for l in range(DS4_N_HASH_LAYER, DS4_N_LAYER):
            pos_total = sum(pos_counts[l]) or 1
            neg_total = sum(neg_counts[l]) or 1
            delta = [(pos_counts[l][e] / pos_total) - (neg_counts[l][e] / neg_total)
                     for e in range(DS4_N_EXPERT)]
            order = sorted(range(DS4_N_EXPERT), key=lambda e: delta[e])
            for e in order[:args.top_n]:                  # most negative -> force OFF
                bias[l * DS4_N_EXPERT + e] = -1.0 if args.strict else float(delta[e])
            for e in order[-args.top_n:]:                 # most positive -> force ON
                bias[l * DS4_N_EXPERT + e] = +1.0 if args.strict else float(delta[e])

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("wb") as fp:
            fp.write(struct.pack(f"<{len(bias)}f", *bias))
        print(f"[expert-steering] wrote {args.out} ({len(bias)} f32 entries, "
              f"{args.out.stat().st_size} bytes, top_n={args.top_n}, strict={args.strict})",
              file=sys.stderr)
        return 0
    finally:
        if not args.keep_dumps:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"[expert-steering] kept dumps under {work}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
