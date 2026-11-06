"""Finding 1 figure: test semantic accuracy (program total match) vs training data volume on Karel v2, List v2 and the
RASP sweep, paper configs (runs: `python reproduce.py finding1`).

x = training volume as a percentage of the domain's full training set, log scale: Karel / List train on 10K / 25K /
50K / 100K of 100K, RASP on 4K / 8K / 16K / 32K of 32K, so the panels share a scale that absolute volumes would not.
y = mean over seeds 0-2 of each run's best-epoch test PTM (the epoch best_model.pth was saved at; Karel / List v2
and the RASP sweep have no validation split, so this selects on the test set); shading = population std over seeds.
The RASP LSTM uses --weight_decay=1.0, every other run none (state this in the caption). Drawn at the NeurIPS text
width, so \\plotinput[\\textwidth]{...} keeps its font sizes.

    python plot_data_efficiency.py   # -> plots/data_efficiency.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.paper_figures import TEXT_WIDTH_IN, save_figure
from plot_ablations_50pct import COLORS, ENCODERS, MARKERS, best_ptm, style_axes

FULL_SET_K = {"karel": 100, "list": 100, "rasp": 32}  # full training set, thousands of samples
VOLUMES_K = {"karel": [10, 25, 50, 100], "list": [10, 25, 50, 100], "rasp": [4, 8, 16, 32]}
TITLES = {"karel": "Karel", "list": "List", "rasp": "RASP"}
SEEDS = [0, 1, 2]
YLABEL = "Semantic Accuracy"
XLABEL = "Training Volume (% of full set)"


def run_name(domain, enc, k):
    """Run name (without -r<seed>) of the paper-ready config of `enc` trained on k thousand samples."""
    if domain in ("karel", "list"):
        tag = {"Logsig": "logsig-unitstep", "LSTM": "lstm-time", "Transformer": "trans",
               "Fourier": "fourier-m9-noset"}[enc]
        return f"{domain.upper()}-v2-{k}k-{tag}-grid"
    stem = {"Logsig": "logsig_concat_onehot_fn3_p32_k4_settrans_unitstep",
            "LSTM": "lstm_add_sin_fn3_p256_k4_settrans_wd1",
            "Transformer": "transformer_add_sin_fn3_p256_k4_settrans",
            "Fourier": "fourier_add_sin_m9_fn3_p256_k4"}[enc]
    return f"rasp_ta_{stem}_sweep32k_cap{k}k"


def collect(run_logs):
    """(domain, encoder) -> (means, stds) of best-epoch test PTM over VOLUMES_K[domain]. Fails loud on a missing seed
    or short log."""
    stats = {}
    for dom, volumes in VOLUMES_K.items():
        for enc in ENCODERS:
            vals = np.array([[best_ptm(run_logs, run_name(dom, enc, k), seed) for seed in SEEDS] for k in volumes])
            stats[dom, enc] = vals.mean(axis=1), vals.std(axis=1)
    return stats


def print_table(stats):
    """Plain-text table of every plotted value (the figure's table view)."""
    print(f"{'':12s}" + "".join(f"{enc:>16s}" for enc in ENCODERS))
    for dom, volumes in VOLUMES_K.items():
        for i, k in enumerate(volumes):
            cells = "".join(f"{stats[dom, enc][0][i]:9.3f} ± {stats[dom, enc][1][i]:.3f}" for enc in ENCODERS)
            print(f"{TITLES[dom]:5s} {k:>3d}K  {cells}")


def plot(stats, out_path):
    rc = {"font.size": 8, "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(VOLUMES_K), figsize=(TEXT_WIDTH_IN, 2.1), sharey=True)
        for ax, (dom, volumes) in zip(axes, VOLUMES_K.items()):
            pct = [100 * k / FULL_SET_K[dom] for k in volumes]
            for enc in ENCODERS:
                mean, std = stats[dom, enc]
                ax.fill_between(pct, mean - std, mean + std, color=COLORS[enc], alpha=0.2, linewidth=0)
                ax.plot(pct, mean, color=COLORS[enc], marker=MARKERS[enc], linewidth=1.5, markersize=4,
                        markeredgecolor="white", markeredgewidth=0.6, label=enc,
                        zorder=4 if enc == "Logsig" else 2)
            ax.set_xscale("log")
            ax.set_xticks(pct)
            ax.set_xticklabels([f"{p:g}" for p in pct])
            ax.minorticks_off()
            ax.set_title(TITLES[dom], fontweight="bold")
            style_axes(ax)
        axes[0].set_ylabel(YLABEL, fontweight="bold")
        axes[0].set_ylim(bottom=0)
        axes[len(axes) // 2].set_xlabel(XLABEL, fontweight="bold")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(ENCODERS), frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_logs", default="run_logs")
    ap.add_argument("--out", default="plots/data_efficiency.png")
    args = ap.parse_args()

    stats = collect(args.run_logs)
    print_table(stats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot(stats, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
