"""Figures for the 50%-volume ablations on Karel v2 / List v2 (50K) and the RASP sweep (16K), paper configs, one seed
(runs: `python reproduce.py ablations`):
    sig_vs_logsig.png       depth-2 signature vs log-signature, at each domain's paper-config width, at FULL volume
                            (Karel / List 100K, RASP 32K): per domain, two parallel axes, accuracy and number of
                            features, one line per encoder
    depth_x_width.png       logsig truncation depth M in {1,2,3} x state embedding width d in {16,32,64}
    width_vs_baselines.png  program total match vs width d for Logsig / LSTM / Transformer / Fourier

Each value is the test-set program total match of run_logs/<run>-r0.csv at its best epoch (= best_model.pth).
width_vs_baselines and sig_vs_logsig are drawn at the NeurIPS text width for the paper appendix
(\\plotinput[\\textwidth]{...}).
Width d = --pre_enc_dim (RASP: the per-frame network's output). The RASP LSTM uses --weight_decay=1.0, every other
run none (state this in the caption).

Example:
    python plot_ablations_50pct.py --out_dir plots/ablations_50pct
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.paper_figures import TEXT_WIDTH_IN, save_figure

DOMAINS = ["karel", "list", "rasp"]
TITLES = {"karel": "Karel, 50K", "list": "List, 50K", "rasp": "RASP, 16K"}
DOMAIN_NAMES = {"karel": "Karel", "list": "List", "rasp": "RASP"}
ENCODERS = ["Logsig", "LSTM", "Transformer", "Fourier"]  # fixed paper series order
COLORS = {"Logsig": "firebrick", "LSTM": "navy", "Transformer": "darkgreen", "Fourier": "darkorange"}
# Firebrick and darkgreen are close under protanopia (ΔE 7.2), so every encoder also gets its own marker.
MARKERS = {"Logsig": "o", "LSTM": "s", "Transformer": "^", "Fourier": "D"}
DEPTH_COLORS = {1: "#fc9b7c", 2: "#f14432", 3: "#ac1117"}  # one-hue ramp, light -> dark
SIG_COLOR = "dimgray"
# sig_vs_logsig: top of the accuracy axis, of each domain's feature axis, and the two axes' colours (muted, none of
# the encoder colours)
ACC_MAX = 0.8
FEAT_MAX = {"karel": 75000, "list": 75000, "rasp": 3000}
AXIS_COLORS = ("#4f6d8a", "#9c7a54")
WIDTHS = [16, 32, 64, 128, 256]
DEPTH_WIDTHS = [16, 32, 64]
YLABEL = "Semantic Accuracy"


def run_name(domain, enc, d, depth=2, full=False):
    """Run name of one config at 50% volume (`full`: 100%), or None if it was not run (RASP d=256: depth-2 logsig
    only)."""
    if domain in ("karel", "list"):
        tag = {"Logsig": "logsig-unitstep" + ("" if depth == 2 else f"-m{depth}"), "Sig": "sig-unitstep",
               "LSTM": "lstm-time", "Transformer": "trans", "Fourier": "fourier-m9-noset"}[enc]
        return f"{domain.upper()}-v2-{100 if full else 50}k-{tag}{'' if d == 256 else f'-p{d}'}-grid"
    cap = f"_sweep32k_cap{32 if full else 16}k"
    if enc in ("Logsig", "Sig"):
        if d == 256 and (enc == "Sig" or depth != 2):
            return None
        name = "logsig" if enc == "Logsig" else "sig"
        return f"rasp_ta_{name}_concat_onehot_fn3_p{d}_k4_settrans_unitstep{'' if depth == 2 else f'_m{depth}'}{cap}"
    if enc == "Fourier":
        return f"rasp_ta_fourier_add_sin_m9_fn3_p{d}_k4{cap}"
    if enc == "LSTM":  # without weight decay it never learns on RASP (<= 0.03 at every width)
        return f"rasp_ta_lstm_add_sin_fn3_p{d}_k4_settrans_wd1{cap}"
    return f"rasp_ta_{enc.lower()}_add_sin_fn3_p{d}_k4_settrans{cap}"


def paper_width(domain, enc):
    """Width d of the paper-ready config (RASP logsig/sig: 32, everything else: 256)."""
    return 32 if domain == "rasp" and enc in ("Logsig", "Sig") else 256


def n_features(domain, enc):
    """Depth-2 feature count of the paper-config path, width D = d (+ RASP's 18 one-hot time channels; Karel / List add
    their time code to the d channels): signature D(D+1), log-signature D(D+1)/2. Matches the checkpoints' post_sig."""
    D = paper_width(domain, enc) + (18 if domain == "rasp" else 0)
    return D * (D + 1) // (1 if enc == "Sig" else 2)


def test_ptm_curve(run_logs, run, seed=0):
    """Test program total match of every logged epoch of the log's last run (a rerun appends to the same CSV from
    epoch 0). Fails loud on a missing or short log."""
    path = Path(run_logs) / f"{run}-r{seed}.csv"
    rows = list(csv.DictReader(open(path)))
    starts = [i for i, r in enumerate(rows) if r.get("epoch") == "0"]
    rows = rows[starts[-1]:] if starts else rows
    col = [c for c in rows[0] if c.startswith("ID[") and c.endswith("-program_total_match")]
    if len(col) != 1:
        raise ValueError(f"{path}: expected one ID[...]-program_total_match column, got {col}")
    vals = [float(r[col[0]]) for r in rows if r[col[0]] not in ("", None)]
    if len(vals) < 10:
        raise ValueError(f"{path}: only {len(vals)} test evaluations")
    return vals


def best_ptm(run_logs, run, seed=0):
    """Test program total match of the best epoch (= the epoch best_model.pth was saved at)."""
    return max(test_ptm_curve(run_logs, run, seed))


def style_axes(ax):
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def label_bars(ax, bars, vals):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)


def plot_sig_vs_logsig(ptm, out_path):
    """Two parallel zero-based axes per domain, accuracy (0 - ACC_MAX) and number of features (0 - FEAT_MAX[dom]),
    one line per encoder from its accuracy to its feature count."""
    encs = [("Logsig", "Log-signature", COLORS["Logsig"], "o"), ("Sig", "Signature", SIG_COLOR, "s")]
    rc = {"font.size": 8, "axes.titlesize": 9, "xtick.labelsize": 7, "legend.fontsize": 8}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(DOMAINS), figsize=(TEXT_WIDTH_IN, 2.4))
        for ax, dom in zip(axes, DOMAINS):
            accs = {enc: ptm(run_name(dom, enc, paper_width(dom, enc), full=True)) for enc, *_ in encs}
            for enc, label, color, marker in encs:
                acc, feat = accs[enc], n_features(dom, enc)
                y_feat = feat / FEAT_MAX[dom] * ACC_MAX  # the feature axis drawn in accuracy units
                ax.plot([0, 1], [acc, y_feat], color=color, linewidth=1.8, marker=marker, markersize=4.5,
                        markeredgecolor="white", markeredgewidth=0.6, label=label, clip_on=False, zorder=3)
                above = acc >= accs["Sig" if enc == "Logsig" else "Logsig"]  # the two accuracies nearly coincide
                ax.annotate(f"{acc:.3f}", (0, acc), xytext=(-5, 4 if above else -4), textcoords="offset points",
                            ha="right", va="bottom" if above else "top", fontsize=6.5, color=color)
                ax.annotate(f"{feat:,}", (1, y_feat), xytext=(5, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=6.5, color=color)
            axis_ends = ((0, f"{ACC_MAX:g}", "right", -4), (1, f"{FEAT_MAX[dom] / 1000:g}K", "left", 4))
            for (x, top, ha, dx), color in zip(axis_ends, AXIS_COLORS):
                ax.axvline(x, color=color, linewidth=1.2, zorder=1)
                ax.annotate(top, (x, ACC_MAX), xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=6, color=color)
                ax.annotate("0", (x, 0), xytext=(dx, 0), textcoords="offset points", ha=ha, va="center",
                            fontsize=6, color=color)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, ACC_MAX)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Accuracy", "Features"], fontweight="bold")
            for tick, color in zip(ax.get_xticklabels(), AXIS_COLORS):
                tick.set_color(color)
            ax.tick_params(axis="x", length=0, pad=4)
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{DOMAIN_NAMES[dom]} (d = {paper_width(dom, 'Logsig')})", fontweight="bold", pad=14)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0.02, 0, 0.97, 0.88), w_pad=4.5)
        save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_depth_x_width(ptm, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    x = np.arange(len(DEPTH_WIDTHS))
    w = 0.26
    for ax, dom in zip(axes, DOMAINS):
        for i, m in enumerate((1, 2, 3)):
            vals = [ptm(run_name(dom, "Logsig", d, depth=m)) for d in DEPTH_WIDTHS]
            bars = ax.bar(x + (i - 1) * w, vals, w, color=DEPTH_COLORS[m], edgecolor="white", linewidth=1,
                          label=f"M = {m}")
            label_bars(ax, bars, vals)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in DEPTH_WIDTHS])
        ax.set_xlabel("State embedding width d", fontweight="bold")
        ax.set_title(TITLES[dom], fontweight="bold", fontsize=10)
        style_axes(ax)
    axes[0].set_ylabel(YLABEL, fontweight="bold")
    axes[0].set_ylim(0, 0.62)
    fig.legend(*axes[0].get_legend_handles_labels(), title="Truncation depth", loc="upper center", ncol=3,
               frameon=False, fontsize=8, title_fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_width_vs_baselines(ptm, out_path, ymax=0.62):
    rc = {"font.size": 8, "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.1), sharey=True)
        for ax, dom in zip(axes, DOMAINS):
            for enc in ENCODERS:
                vals = [ptm(run_name(dom, enc, d)) for d in WIDTHS]
                ax.plot(WIDTHS, vals, color=COLORS[enc], marker=MARKERS[enc], linewidth=1.5, markersize=4,
                        markeredgecolor="white", markeredgewidth=0.6, label=enc, zorder=4 if enc == "Logsig" else 2)
            ax.set_xscale("log", base=2)
            ax.set_xticks(WIDTHS)
            ax.set_xticklabels([str(d) for d in WIDTHS])
            ax.minorticks_off()
            ax.set_title(DOMAIN_NAMES[dom], fontweight="bold")
            style_axes(ax)
        axes[0].set_ylabel(YLABEL, fontweight="bold")
        axes[0].set_ylim(0, ymax)
        axes[1].set_xlabel("State embedding width $d$", fontweight="bold")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(ENCODERS), frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def print_table(ptm):
    """Plain-text table of every plotted value (the figures' table view)."""
    print("width vs baselines (d = " + " / ".join(map(str, WIDTHS)) + ")")
    for dom in DOMAINS:
        for enc in ENCODERS:
            cells = [f"{ptm(r):.3f}" if (r := run_name(dom, enc, d)) else "  -  " for d in WIDTHS]
            print(f"  {dom:5s} {enc:11s} " + "  ".join(cells))
    print("depth x width (d = " + " / ".join(map(str, DEPTH_WIDTHS)) + ")")
    for dom in DOMAINS:
        for m in (1, 2, 3):
            print(f"  {dom:5s} M={m}  " + "  ".join(f"{ptm(run_name(dom, 'Logsig', d, m)):.3f}" for d in DEPTH_WIDTHS))
    print("logsig / sig at full volume, paper width: accuracy, features")
    for dom in DOMAINS:
        print(f"  {dom:5s} " + " / ".join(f"{ptm(run_name(dom, enc, paper_width(dom, enc), full=True)):.3f}"
                                          for enc in ("Logsig", "Sig"))
              + "   " + " / ".join(f"{n_features(dom, enc):,}" for enc in ("Logsig", "Sig")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_logs", default="run_logs")
    ap.add_argument("--out_dir", default="plots/ablations_50pct")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ptm = lambda run: best_ptm(args.run_logs, run)
    plot_sig_vs_logsig(ptm, out / "sig_vs_logsig.png")
    plot_depth_x_width(ptm, out / "depth_x_width.png")
    plot_width_vs_baselines(ptm, out / "width_vs_baselines.png")
    print_table(ptm)
    print(f"Wrote {out}/{{sig_vs_logsig,depth_x_width,width_vs_baselines}}.png")


if __name__ == "__main__":
    main()
