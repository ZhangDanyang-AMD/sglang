#!/usr/bin/env python
"""Roofline model for fused_sigmoid_gating_delta_rule_update_kernel_opt.

This script is intended to be a lightweight, reproducible way to build a
USE_INITIAL_STATE=True roofline estimate for the Triton kernel in
   `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`.

Assumptions (tunable via CLI):
- Bandwidth (HBM) default: 5.3 TB/s
- Peak compute default: 163.4 TFLOP/s
- This roofline model assumes no tiling in K/V for simplicity: BK=K and BV=V.

Notes:
- FLOPs are estimated from the algorithm structure (dominant recurrent math) and
  do not attempt to assign FLOP-equivalents to transcendental/SFU ops.

Optional:
- Pass `--plot` to save a roofline figure (requires matplotlib).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_BW_BYTES_PER_S = 5.3e12  # 5.3 TB/s
DEFAULT_PEAK_FLOPS = 163.4e12  # 163.4 TFLOP/s


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def dtype_size_bytes(dtype: str) -> int:
    dt = dtype.lower()
    if dt in {"fp16", "float16", "f16"}:
        return 2
    if dt in {"bf16", "bfloat16"}:
        return 2
    if dt in {"fp32", "float32", "f32"}:
        return 4
    if dt in {"fp8", "e4m3", "e5m2"}:
        return 1
    if dt in {"int8", "i8", "uint8", "u8"}:
        return 1
    if dt in {"int32", "i32", "uint32", "u32"}:
        return 4
    raise ValueError(f"Unknown dtype: {dtype}")


@dataclass(frozen=True)
class RooflineInputs:
    # Shapes
    N: int
    T: int
    H: int
    HV: int
    K: int
    V: int
    # Dtypes
    dtype_q: str
    dtype_k: str
    dtype_v: str
    dtype_o: str
    dtype_gating: str
    dtype_h0: str
    # Behavior
    use_qk_l2norm_in_kernel: bool = False
    use_initial_state: bool = True


@dataclass(frozen=True)
class RooflineResult:
    bytes_total: int
    flops_total: float
    arithmetic_intensity: float
    ridge_ai: float
    roofline_flops_per_s: float
    bound: str
    est_time_s: float


def _logspace(start_log10: float, end_log10: float, num: int) -> list[float]:
    if num <= 1:
        return [10.0 ** end_log10]
    step = (end_log10 - start_log10) / (num - 1)
    return [10.0 ** (start_log10 + i * step) for i in range(num)]


def plot_roofline(
    *,
    bw_bytes_per_s: float,
    peak_flops: float,
    ai: float,
    achieved_flops_per_s: float,
    out_path: Path,
    show: bool,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for --plot. Install it (e.g. `pip install matplotlib`) and retry."
        ) from e

    ridge_ai = peak_flops / bw_bytes_per_s
    # Choose a log-range that covers the modeled point and ridge.
    eps = 1e-12
    ai_center = max(ai, eps)
    x_min = max(min(ai_center / 100.0, ridge_ai / 100.0, 1e-4), 1e-8)
    x_max = max(ai_center * 100.0, ridge_ai * 100.0, 1e2)

    xs = _logspace(math.log10(x_min), math.log10(x_max), 400)
    mem_line = [bw_bytes_per_s * x for x in xs]
    peak_line = [peak_flops for _ in xs]
    roof_line = [min(p, m) for p, m in zip(peak_line, mem_line)]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(xs, [y / 1e12 for y in mem_line], label=f"BW line ({bw_bytes_per_s/1e12:.2f} TB/s)")
    ax.plot(xs, [y / 1e12 for y in peak_line], label=f"Peak ({peak_flops/1e12:.1f} TFLOP/s)")
    ax.plot(xs, [y / 1e12 for y in roof_line], linewidth=2.5, label="Roofline")

    # Ridge point
    ax.axvline(ridge_ai, linestyle="--", linewidth=1.0, label=f"Ridge AI={ridge_ai:.3g} FLOP/B")

    # Modeled point
    ax.scatter([ai], [achieved_flops_per_s / 1e12], s=35, zorder=5, label="Modeled point")

    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)


def estimate_bytes_use_initial_state(inp: RooflineInputs) -> int:
    sh = inp.HV // inp.H
    if inp.HV % inp.H != 0:
        raise ValueError(f"HV must be divisible by H (got HV={inp.HV}, H={inp.H})")

    q_bytes = dtype_size_bytes(inp.dtype_q)
    k_bytes = dtype_size_bytes(inp.dtype_k)
    v_bytes = dtype_size_bytes(inp.dtype_v)
    o_bytes = dtype_size_bytes(inp.dtype_o)
    g_bytes = dtype_size_bytes(inp.dtype_gating)
    h0_bytes = dtype_size_bytes(inp.dtype_h0)

    # Model assumes BV == V (no V tiling). Per timestep per program:
    bytes_qk = (inp.K * q_bytes) + (inp.K * k_bytes)
    bytes_v_in = sh * inp.V * v_bytes
    bytes_o_out = sh * inp.V * o_bytes

    # Gating parameters are per shared_h lane (scalars), loaded each timestep.
    # (A_log, a, dt_bias, b)
    bytes_gating = sh * 4 * g_bytes

    bytes_per_step = bytes_qk + bytes_v_in + bytes_o_out + bytes_gating

    bytes_init = 0
    bytes_final = 0
    if inp.use_initial_state:
        tile_elems = sh * inp.K * inp.V
        bytes_init = tile_elems * h0_bytes
        bytes_final = tile_elems * h0_bytes

    programs = inp.N * inp.H
    total = programs * (bytes_init + inp.T * bytes_per_step + bytes_final)
    return int(total)


def estimate_flops_algorithmic(inp: RooflineInputs) -> float:
    """Estimate FLOPs from the dominant recurrent math.

    We only count add/mul/fma-like operations (not exp/log/div), because roofline TFLOPs
    is typically defined for FMA throughput on FP types.
    """

    sh = inp.HV // inp.H

    # Per timestep per shared_h lane:
    # b_h *= exp(g): K*V mul
    # b_v -= sum(b_h * b_k): ~2*K*V flops
    # b_v *= beta: V mul
    # b_h += b_k * b_v: 2*K*V flops
    # b_o = sum(b_h * b_q): ~2*K*V flops
    flops_per_sh_step = (1.0 * inp.K * inp.V) + (2.0 * inp.K * inp.V) + (1.0 * inp.V) + (2.0 * inp.K * inp.V) + (2.0 * inp.K * inp.V)

    # Optional q/k L2norm adds extra math; keep it coarse.
    # For each of q and k: sum(K mul + (K-1) add) + sqrt + div => count only mul/add + div as ~K mul/add.
    if inp.use_qk_l2norm_in_kernel:
        flops_per_sh_step += 4.0 * inp.K  # very rough; does not scale with BV

    flops_step = sh * flops_per_sh_step

    programs = inp.N * inp.H
    return float(programs * inp.T * flops_step)


def roofline(bytes_total: int, flops_total: float, bw_bytes_per_s: float, peak_flops: float) -> RooflineResult:
    ai = flops_total / bytes_total if bytes_total > 0 else math.inf
    ridge = peak_flops / bw_bytes_per_s
    mem_bound = ai * bw_bytes_per_s
    roof = min(peak_flops, mem_bound)
    bound = "memory" if mem_bound < peak_flops else "compute"
    t = flops_total / roof if roof > 0 else math.inf
    return RooflineResult(
        bytes_total=bytes_total,
        flops_total=flops_total,
        arithmetic_intensity=ai,
        ridge_ai=ridge,
        roofline_flops_per_s=roof,
        bound=bound,
        est_time_s=t,
    )


def format_human_bytes(n: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024.0:
            return f"{n:.3g} {unit}"
        n /= 1024.0
    return f"{n:.3g} PiB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, default=DEFAULT_BW_BYTES_PER_S, help="HBM bandwidth in bytes/s (default 5.3e12).")
    ap.add_argument("--peak", type=float, default=DEFAULT_PEAK_FLOPS, help="Peak compute in FLOP/s (default 163.4e12).")

    ap.add_argument("--plot", action="store_true", help="Save a roofline plot (requires matplotlib).")
    ap.add_argument("--plot-path", type=str, default="roofline.png", help="Output image path for --plot.")
    ap.add_argument("--no-show", action="store_true", help="Do not open an interactive window when plotting.")

    ap.add_argument("--N", type=int, required=True, help="Number of sequences (B or varlen N).")
    ap.add_argument("--T", type=int, required=True, help="Sequence length.")
    ap.add_argument("--H", type=int, required=True, help="Number of heads.")
    ap.add_argument("--HV", type=int, required=True, help="Value heads (v.shape[2]).")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--V", type=int, required=True)

    ap.add_argument("--dtype-q", type=str, default="fp16")
    ap.add_argument("--dtype-k", type=str, default="fp16")
    ap.add_argument("--dtype-v", type=str, default="fp16")
    ap.add_argument("--dtype-o", type=str, default="fp16")
    ap.add_argument("--dtype-gating", type=str, default="fp32", help="dtype for (A_log,a,dt_bias,b).")
    ap.add_argument("--dtype-h0", type=str, default="fp16")

    ap.add_argument("--no-initial-state", action="store_true")
    ap.add_argument("--use-qk-l2norm", action="store_true")

    args = ap.parse_args()

    inp = RooflineInputs(
        N=args.N,
        T=args.T,
        H=args.H,
        HV=args.HV,
        K=args.K,
        V=args.V,
        dtype_q=args.dtype_q,
        dtype_k=args.dtype_k,
        dtype_v=args.dtype_v,
        dtype_o=args.dtype_o,
        dtype_gating=args.dtype_gating,
        dtype_h0=args.dtype_h0,
        use_qk_l2norm_in_kernel=args.use_qk_l2norm,
        use_initial_state=not args.no_initial_state,
    )

    bytes_total = estimate_bytes_use_initial_state(inp)
    flops_total = estimate_flops_algorithmic(inp)
    res = roofline(bytes_total, flops_total, args.bw, args.peak)
    sh = args.HV // args.H

    print("=== Roofline model (USE_INITIAL_STATE path) ===")
    print(f"N={args.N}, T={args.T}, H={args.H}, HV={args.HV} (shared_h={sh}), K={args.K}, V={args.V}")
    print(f"Dtypes: q={args.dtype_q}, k={args.dtype_k}, v={args.dtype_v}, o={args.dtype_o}, gating={args.dtype_gating}, h0={args.dtype_h0}")
    print(f"use_initial_state={not args.no_initial_state}, use_qk_l2norm={args.use_qk_l2norm}")
    print(f"Total bytes (modeled): {format_human_bytes(res.bytes_total)} ({res.bytes_total:.3e} B)")
    print(f"Total FLOPs (modeled): {res.flops_total:.3e} FLOP")
    print(f"Arithmetic intensity: {res.arithmetic_intensity:.3f} FLOP/B")
    print(f"Ridge point: {res.ridge_ai:.3f} FLOP/B (peak/bw)")
    print(f"Ceiling (BW): {res.arithmetic_intensity * args.bw / 1e12:.3f} TFLOP/s")
    print(f"Ceiling (Peak): {args.peak / 1e12:.3f} TFLOP/s")
    print(f"Roofline perf: {res.roofline_flops_per_s / 1e12:.3f} TFLOP/s  -> {res.bound}-bound")
    print(f"Estimated time: {res.est_time_s * 1e3:.3f} ms")

    if args.plot:
        out_path = Path(args.plot_path)
        title = f"Roofline (N={args.N}, T={args.T}, H={args.H}, HV={args.HV}, K={args.K}, V={args.V})"
        plot_roofline(
            bw_bytes_per_s=args.bw,
            peak_flops=args.peak,
            ai=res.arithmetic_intensity,
            achieved_flops_per_s=res.roofline_flops_per_s,
            out_path=out_path,
            show=not args.no_show,
            title=title,
        )
        print(f"Saved roofline plot to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
