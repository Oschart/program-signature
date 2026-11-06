"""Appendix runtime figure (fig:encoder_runtime): forward-pass wall-clock time of the four trace encoders vs trace
length T.

Builds only the trace encoder of each encoder's paper Karel configuration (its full-volume run in reproduce.py; List's
trace encoders are the same) -- no state encoder, set encoder or decoder, which every encoder shares -- feeds it one
random trace of T frames (B = K = 1, width 256) and times the forward pass: mean and population std of --n_repeat timed
passes after --n_warmup untimed ones, at --n_T evenly spaced T up to --T_max. An encoder that runs out of GPU memory
stops at the last T that fit (recorded in the cache's meta). The paper's timings were taken on one NVIDIA A100 (40 GB).

Timing writes --cache and draws the figure; --plot_only redraws it from the cache. Drawn at FIG_WIDTH_IN, so
\\plotinput[0.6\\textwidth]{...} keeps its font sizes.

    python plot_encoder_runtime.py --device=cuda   # time -> cache/encoder_runtime.json, then plot
    python plot_encoder_runtime.py --plot_only     # plot from the cache
"""
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from common.config_utils import BaseOptions
from common.paper_figures import TEXT_WIDTH_IN, save_figure
from models.encoders import encoder_map
from plot_ablations_50pct import COLORS, ENCODERS, MARKERS, style_axes
from plot_data_efficiency import run_name
from reproduce import FULL_K, run as paper_run

FIG_WIDTH_IN = 0.6 * TEXT_WIDTH_IN


def encoder_config(enc):
    """Config of `enc`'s paper Karel run: the flags of its reproduce.py command on BaseOptions' defaults."""
    _, argv = paper_run("karel", enc, FULL_K["karel"], run_name("karel", enc, FULL_K["karel"]), 0)
    cfg = vars(BaseOptions().parser.parse_known_args(argv[3:])[0])  # argv: python -m train <flags>
    cfg["max_demo_length"] = 100  # initial size of the positional table; it grows with T
    return cfg


def time_forward(encoder, T, n_warmup, n_repeat, device):
    """Wall-clock seconds of n_repeat forward passes of encoder on one random trace of T frames."""
    batch = {"traces": th.randn(1, 1, T, encoder.feat_dim, device=device),
             "trace_lens": th.full((1, 1), T, dtype=th.long, device=device)}
    sync = th.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    with th.inference_mode():
        for _ in range(n_warmup):
            encoder(batch)
        times = []
        for _ in range(n_repeat):
            sync()
            t0 = time.perf_counter()
            encoder(batch)
            sync()
            times.append(time.perf_counter() - t0)
    return times


def measure(args):
    """{"meta": ..., "seconds": {encoder: {T: [seconds per pass]}}} for the paper Karel config of every encoder."""
    th.manual_seed(0)
    T_values = np.linspace(args.T_max / args.n_T, args.T_max, args.n_T).round().astype(int).tolist()
    runs = {enc: run_name("karel", enc, FULL_K["karel"]) + "-r0" for enc in ENCODERS}
    seconds, out_of_memory = {}, {}
    for enc, run in runs.items():
        cfg = encoder_config(enc)
        encoder = encoder_map[cfg["trace_encoder"]](cfg).to(args.device).eval()
        print(f"=== {enc}: {run} (input width {encoder.feat_dim})")
        seconds[enc] = {}
        for T in T_values:
            try:
                seconds[enc][T] = time_forward(encoder, T, args.n_warmup, args.n_repeat, args.device)
            except th.cuda.OutOfMemoryError:
                print(f"  T={T:>6d}  out of GPU memory: no longer traces timed")
                out_of_memory[enc] = T
                th.cuda.empty_cache()
                break
            print(f"  T={T:>6d}  {1000 * np.mean(seconds[enc][T]):10.3f} ± {1000 * np.std(seconds[enc][T]):.3f} ms")
    device = th.cuda.get_device_name() if args.device.startswith("cuda") else args.device
    meta = {"device": device, "torch": th.__version__, "runs": runs, "n_warmup": args.n_warmup,
            "n_repeat": args.n_repeat, "out_of_memory_at": out_of_memory}
    return {"meta": meta, "seconds": seconds}


def plot(results, out_path):
    rc = {"font.size": 8, "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 2.3))
        for enc in ENCODERS:
            by_T = {int(T): ms for T, ms in results["seconds"][enc].items()}  # JSON keys are strings
            T = sorted(by_T)
            ms = 1000 * np.array([by_T[t] for t in T])
            mean, std = ms.mean(axis=1), ms.std(axis=1)
            ax.fill_between(T, mean - std, mean + std, color=COLORS[enc], alpha=0.2, linewidth=0)
            ax.plot(T, mean, color=COLORS[enc], marker=MARKERS[enc], linewidth=1.5, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.6, label=enc, zorder=4 if enc == "Logsig" else 2)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Trace length $T$", fontweight="bold")
        ax.set_ylabel("Forward time (ms)", fontweight="bold")
        style_axes(ax)
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--T_max", type=int, default=100000)
    p.add_argument("--n_T", type=int, default=20)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_repeat", type=int, default=5)
    p.add_argument("--cache", default="cache/encoder_runtime.json")
    p.add_argument("--plot_only", action="store_true", help="plot --cache without timing")
    p.add_argument("--out", default="plots/encoder_runtime.png")
    args = p.parse_args()

    if args.plot_only:
        with open(args.cache) as f:
            results = json.load(f)
    else:
        results = measure(args)
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        with open(args.cache, "w") as f:
            json.dump(results, f, indent=1)
        print(f"Wrote {args.cache}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot(results, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
