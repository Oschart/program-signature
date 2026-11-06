"""
latent_view.py — Finding 3: t-SNE of the embedded traces each trace encoder reads.

For each model in `--model_specs` (default: the paper configs, model_specs/latent_paper_configs.json):
  * Load the full training config from <checkpoint_dir>/config.json
    (only minimal runtime fields like device/test_only are overridden).
  * Run the model's state encoder and trace encoder on the first test batch (`--batch_size` programs) and keep
    the embedded traces the encoder reads, with every encoder's state embeddings rebuilt with unit-length steps
    before the time code (--sig_step_norm=unit, as the log-signature encoder does), so all panels are comparable.
  * Report, per model, the share of each state's five nearest states (full space, all captured programs) that
    lie on its own trace (printed and written to <plot_dir>/geometry_pre_enc_traces_unitstep_full.json).
  * Fit t-SNE on the first `--n_samples` programs per model and draw every model in one figure at the paper's
    text width: one row per domain, one column per encoder (paths_combined_TSNE_pre_enc_traces_unitstep_full.png,
    plus its LaTeX copy, see common/paper_figures.py).

RASP models get RASP_VOCAB_SIZE / RASP_MAX_SEQ_LEN from their datadir tag.

Paper figures: appendix fig:path_tsne_full (all domains), then main-text fig:path_tsne (Karel row):
    python latent_view.py --device=cuda --plot_dir=plots/latent_view/final
    python latent_view.py --device=cuda --models=Karel-Logsig,Karel-LSTM,Karel-Transformer,Karel-Fourier \
        --plot_dir=plots/latent_view/final/karel
"""
import argparse
import json
import os
import pickle as pkl
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from common.config_utils import parse_lr
from common.paper_figures import save_figure
from karel_env.dsl import get_DSL, dsl_version_for
from plot_data_efficiency import TEXT_WIDTH_IN
from trainers.trace2code import Trace2Code


# Column labels of the figure, from the model names.
GROUP_TAGS = {
    "logsig": "Logsig",
    "trans": "Transformer",
    "lstm": "LSTM",
    "fourier": "Fourier",
}
# Column order of the figure (paper series order); other labels follow.
ENCODER_ORDER = ["Logsig", "LSTM", "Transformer", "Fourier"]
# Nearest neighbours per state in the geometry numbers.
GEOMETRY_KNN = 5
# Program colours: muted, colour-blind-safe hues; a 6th program repeats the first colour.
PATH_COLORS = ["#3d70b3", "#cc5f3d", "#21906c", "#c39320", "#a8619a"]
# Fonts of the paper-width figure.
PAPER_RC = {"font.size": 8, "axes.titlesize": 8}
# Output file tag: the latents (unit-step embedded traces) and the view (every state of the trace).
TAG = "TSNE_pre_enc_traces_unitstep_full"


def short_label(model_name: str) -> str:
    low = model_name.lower()
    for tag, label in GROUP_TAGS.items():
        if tag in low:
            return label
    return model_name


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

# Runtime overrides FORCED on every loaded config (override saved values).
_RUNTIME_OVERRIDES = {
    "no_wandb": True,
    "test_only": True,
    "num_workers": 0,
    "wandb_key": "",
    "overwrite_logs": False,
}

# Defaults set only when the saved config didn't already have them.
_RUNTIME_DEFAULTS = {
    "prefetch_factor": 2,
    "log_every": 20,
}


def _set_rasp_vocab(datadir: str):
    """RASP token vocab is built from RASP_VOCAB_SIZE / RASP_MAX_SEQ_LEN, read once when
    rasp_grammar is first imported. Export them from the datadir tag rasp_v<V>s<L>; raise
    if the vocab already imported differs (two RASP vocabularies cannot share one process)."""
    m = re.match(r"rasp_v(\d+)s(\d+)", datadir)
    if m is None:
        raise ValueError(f"Cannot read V/L from RASP datadir '{datadir}' (expected rasp_v<V>s<L>-...)")
    os.environ.setdefault("RASP_VOCAB_SIZE", m.group(1))
    os.environ.setdefault("RASP_MAX_SEQ_LEN", m.group(2))
    from rasp_env import rasp_grammar as G
    have, want = (G.RASP_VOCAB_SIZE, G.RASP_MAX_SEQ_LEN), (int(m.group(1)), int(m.group(2)))
    if have != want:
        raise ValueError(f"{datadir} needs RASP vocab V/L={want}, this process has {have}. "
                         f"Run RASP datasets with different V/L in separate invocations.")


def load_model_config(checkpoint_path: str, device: str, batch_size: int) -> dict:
    """Build a Trace2Code-ready config from the saved sidecar config.json.

    The saved config (see trainers/base.py:save_config_json) excludes runtime-derived
    objects (tokenizer, env_spec, ...) and secrets — we set sensible inference-time
    defaults for those, then re-derive what trainer init expects.
    """
    cfg_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing sidecar config: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    cfg.update(_RUNTIME_OVERRIDES)
    for k, v in _RUNTIME_DEFAULTS.items():
        cfg.setdefault(k, v)

    cfg["device"] = device
    cfg["batch_size"] = batch_size
    cfg["checkpoint"] = checkpoint_path
    cfg["log_dir"] = f"logs/_latent_view/{cfg.get('exp_name', 'unknown')}"
    os.makedirs(cfg["log_dir"], exist_ok=True)

    # Re-derive what BaseOptions.parse() normally adds at the end.
    cfg.setdefault("base_lr", parse_lr(cfg.get("lr_schedule", "constant(1e-4)")))
    cfg.setdefault("num_demos_test", cfg.get("num_demos", 5))

    if cfg.get("environment") == "rasp":
        _set_rasp_vocab(cfg["datadir"])

    if "dsl_tokens" not in cfg:
        env = cfg.get("environment", "karel")
        cfg.setdefault("dsl_version", dsl_version_for(cfg["datadir"]))
        dsl = get_DSL(environment=env, version=cfg["dsl_version"])
        cfg["dsl_tokens"] = list(dsl.int2token)

    return cfg


def same_trace_share(paths: th.Tensor, lens: th.Tensor, k: int = GEOMETRY_KNN) -> float:
    """Share of each state's k nearest other states (Euclidean, full space, no 2D projection) that lie on its own
    trace, over all states. paths: (B, K, T, D), lens: (B, K)."""
    B, K, T, D = paths.shape
    lens = lens.reshape(B * K).long()
    flat = paths.reshape(B * K, T, D).double()
    x = th.cat([flat[i, : lens[i]] for i in range(B * K)], dim=0)
    trace_id = th.repeat_interleave(th.arange(B * K), lens)
    dist = th.cdist(x, x)
    dist.fill_diagonal_(float("inf"))
    nn_idx = dist.topk(k, dim=1, largest=False).indices
    return float((trace_id[nn_idx] == trace_id[:, None]).double().mean())


# ---------------------------------------------------------------------------
# LatentView
# ---------------------------------------------------------------------------

class LatentView:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.batch_size = args.batch_size
        self.plot_dir = args.plot_dir
        self.cache_path = args.cache
        os.makedirs(self.plot_dir, exist_ok=True)
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)

        with open(args.model_specs) as f:
            full_registry = json.load(f)["models"]

        if args.models:
            wanted = [m.strip() for m in args.models.split(",") if m.strip()]
            missing = [m for m in wanted if m not in full_registry]
            if missing:
                raise KeyError(
                    f"--models references unknown entries: {missing}. "
                    f"Registry: {list(full_registry)}"
                )
            self.registry = {m: full_registry[m] for m in wanted}
        else:
            self.registry = full_registry

        latents = self.load_models()
        self.report_geometry(latents)
        self.plot_paths(latents, n_samples=args.n_samples)

    # -------------------- model loading --------------------
    @staticmethod
    def capture(model, batch, cfg: dict) -> dict:
        """The embedded traces model's trace encoder reads, rebuilt with unit-length steps before the time code
        (--sig_step_norm=unit) for every encoder."""
        enc = model.trace_encoders[0]
        saved, enc.sig_step_norm = enc.sig_step_norm, "unit"
        try:
            with th.inference_mode():
                model.tracer(batch)
                enc(batch)
        finally:
            enc.sig_step_norm = saved
        return {
            "paths": batch["pre_enc_traces"].detach().cpu(),   # (B, K, T, D)
            "lens": batch["trace_lens"].detach().cpu(),        # (B, K)
            "datadir": cfg["datadir"],
        }

    def load_models(self):
        """Load each model of the registry and capture its latents on the first test batch of its dataset.
        Cached to disk by checkpoint path and batch size, so a later run (e.g. the Karel row after all domains)
        reuses them."""
        cached = {}
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                cached = pkl.load(f)
        usable = {} if self.args.refresh else cached

        latents, fresh = {}, {}
        valid_cache = {}   # datadir -> loaded test set, shared by the models on it
        for name, entry in self.registry.items():
            ckpt_path = entry["checkpoint"]
            key = f"{ckpt_path}|bs={self.batch_size}"
            if key in usable:
                print(f"[cache hit] {name}")
                latents[name] = usable[key]
                continue
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"{name}: checkpoint not found at {ckpt_path}")

            print(f"\n{'=' * 60}\n  Loading {name}\n{'=' * 60}")
            cfg = load_model_config(ckpt_path, self.device, self.batch_size)
            cache_key = f"valid_dataset::{cfg['datadir']}"
            if cfg["datadir"] in valid_cache:
                cfg[cache_key] = valid_cache[cfg["datadir"]]
            solver = Trace2Code(cfg)
            valid_cache[cfg["datadir"]] = solver.config[cache_key]
            solver.model.eval()

            batch = solver.move_batch(next(iter(solver.test_loader)))
            if batch["s_h"].shape[0] < self.args.n_samples:
                raise ValueError(f"{name}: {self.args.n_samples} programs requested but the batch holds "
                                 f"{batch['s_h'].shape[0]}; raise --batch_size.")
            latents[name] = fresh[key] = self.capture(solver.model, batch, cfg)
            print(f"  done [{name}] — paths {tuple(latents[name]['paths'].shape)}")
            del solver

        if self.cache_path and fresh:
            # Merge into the on-disk cache; written to a temp file and renamed, so a concurrent reader never
            # sees a partial file.
            tmp_path = f"{self.cache_path}.{os.getpid()}.tmp"
            with open(tmp_path, "wb") as f:
                pkl.dump({**cached, **fresh}, f)
            os.replace(tmp_path, self.cache_path)
            print(f"\nWrote cache: {self.cache_path}")
        return latents

    # -------------------- geometry --------------------
    def report_geometry(self, latents):
        geometry = {name: {"same_trace": same_trace_share(data["paths"], data["lens"])}
                    for name, data in latents.items()}
        print(f"\nShare of each state's {GEOMETRY_KNN} nearest states on its own trace "
              f"(all {self.batch_size} captured programs, full space):")
        print("| model | same_trace |\n|---|---|")
        for name, g in geometry.items():
            print(f"| {name} | {g['same_trace']:.2f} |")
        out_path = os.path.join(self.plot_dir, "geometry_pre_enc_traces_unitstep_full.json")
        with open(out_path, "w") as f:
            json.dump(geometry, f, indent=2)
        print(f"  wrote {out_path}")

    # -------------------- path figure --------------------
    @staticmethod
    def tsne(path, perplexity=5):
        from sklearn.manifold import TSNE
        perp = max(2, min(perplexity, len(path) - 1))
        return TSNE(n_components=2, random_state=0, init="pca", perplexity=perp).fit_transform(path)

    @staticmethod
    def draw_paths(ax, pts, lens, K, linewidth, markersize):
        """One line per trace in temporal order, a dot per state; the K traces of a program share one color."""
        offset = 0
        for i, ln in enumerate(lens):
            seg = pts[offset : offset + ln]
            offset += ln
            program_idx = i // K
            # Only the first trace of each program carries a legend label.
            label = f"Program {program_idx + 1}" if i % K == 0 else None
            ax.plot(*seg.T, "-o", markersize=markersize, linewidth=linewidth,
                    markeredgewidth=0, solid_capstyle="round",
                    color=PATH_COLORS[program_idx % len(PATH_COLORS)], label=label)

    def plot_paths(self, latents, n_samples):
        """t-SNE (perplexity 5) of the states of the first n_samples programs, fitted per model, drawn in one
        figure TEXT_WIDTH_IN wide (paper fonts at \\plotinput[\\textwidth]): thin lines with small state dots,
        tiny numeric ticks, a small legend (one entry per program) set clear of the panels.

        Each panel is a square symmetric about the origin, where t-SNE centres its output, so the ticks come in
        symmetric pairs (e.g. -50, 0, 50) and no end tick falls just outside a panel."""
        per_model = {}
        for name, data in latents.items():
            paths, lens = data["paths"], data["lens"]
            B, K, T, D = paths.shape
            lens = lens.reshape(-1)[: n_samples * K]
            flat = th.cat([paths.reshape(B * K, T, D)[i, : int(ln)] for i, ln in enumerate(lens)], dim=0).numpy()
            per_model[name] = {"pts": self.tsne(flat), "lens": lens.tolist(), "K": K, "datadir": data["datadir"]}

        with plt.rc_context(PAPER_RC):
            fig, panels = self.paper_grid(per_model)
            for name, ax in panels.items():
                info = per_model[name]
                reach = 1.05 * float(np.abs(info["pts"]).max())
                self.draw_paths(ax, info["pts"], info["lens"], info["K"], linewidth=0.5, markersize=1.1)
                ax.set_xlim(-reach, reach)
                ax.set_ylim(-reach, reach)
                ax.set_aspect("equal", adjustable="box")
                for axis in (ax.xaxis, ax.yaxis):
                    axis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10], integer=True))
                    # Minus as an en dash, a text glyph: LaTeX's math minus has no size below 5 pt and would be
                    # set at twice the label size.
                    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\N{EN DASH}")))
                ax.tick_params(labelsize=2.5, length=1, width=0.3, pad=0.6, color="0.6", labelcolor="0.35")
            handles, labels = next(iter(panels.values())).get_legend_handles_labels()
            # Legend keys as plain, thicker lines: the 0.5-pt dotted trace line is too faint as a key.
            keys = [Line2D([], [], color=h.get_color(), linewidth=1.0) for h in handles]
            fig.legend(keys, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=5, handlelength=1.5)
            # Keep 0.25 in free for the legend, whatever the number of rows: about 0.1 in of it is the gap above it.
            fig.tight_layout(pad=0.3, rect=(0, 0.25 / fig.get_figheight(), 1, 1))
            fname = f"paths_combined_{TAG}.png"
            save_figure(fig, os.path.join(self.plot_dir, fname), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  wrote {fname}")

    @staticmethod
    def paper_grid(per_model):
        """One square panel per model: one row per datadir, one column per encoder (paper order), encoder names
        in bold over the top row, domains beside the rows (none on a one-row figure, whose caption names the
        domain), light spines, empty cells blank. Returns the figure and {model name: axes}."""
        cells, rows, cols = {}, [], []
        for name, info in per_model.items():
            row, col = info["datadir"], short_label(name)
            if (row, col) in cells:
                raise ValueError(f"'{cells[(row, col)]}' and '{name}' both map to {row} / {col}. "
                                 f"Use --models to plot one of them.")
            cells[(row, col)] = name
            if row not in rows:
                rows.append(row)
            if col not in cols:
                cols.append(col)
        cols.sort(key=lambda c: ENCODER_ORDER.index(c) if c in ENCODER_ORDER else len(ENCODER_ORDER))

        panel = TEXT_WIDTH_IN / len(cols)
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(TEXT_WIDTH_IN, panel * len(rows)), squeeze=False)
        panels = {}
        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                ax = axes[r][c]
                if r == 0:
                    ax.set_title(col, fontweight="bold")
                name = cells.get((row, col))
                if name is None:
                    ax.axis("off")
                    continue
                for spine in ax.spines.values():
                    spine.set(color="lightgray", linewidth=0.5)
                if c == 0 and len(rows) > 1:
                    ax.set_ylabel(name.split("-")[0], fontweight="normal")
                panels[name] = ax
        return fig, panels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Finding 3: t-SNE of the embedded traces of the paper's models.")
    p.add_argument("--model_specs", default="model_specs/latent_paper_configs.json")
    p.add_argument("--models", default="",
                   help="Comma-separated subset of model names from the registry to load. "
                        "Empty => use all entries.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=20,
                   help="Test programs captured per model (one batch); the geometry numbers use all of them.")
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of programs drawn per panel.")
    p.add_argument("--plot_dir", default="plots/latent_view")
    p.add_argument("--cache", default="cache/latent_view.pkl",
                   help="Pickle path; pass empty string to disable caching.")
    p.add_argument("--refresh", action="store_true",
                   help="Ignore the cache and re-run all models.")
    p.add_argument("--seed", type=int, default=2023)
    return p.parse_args()


def main():
    th.set_grad_enabled(False)
    args = parse_args()
    if not args.cache:
        args.cache = None

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    LatentView(args)


if __name__ == "__main__":
    main()
