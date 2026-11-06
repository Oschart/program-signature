"""Print the training commands behind the paper's figures, one per line.

    python reproduce.py finding1             # accuracy vs training volume: 3 domains x 4 encoders x 4 volumes x 3 seeds
    python reproduce.py ablations            # appendix ablations (signature vs log-signature, depth x width, width), seed 0
    python reproduce.py finding2             # bootstrapping on Karel+ / List+ / RASP+: 3 domains x 4 encoders, one seed each
    python reproduce.py finding1 | bash      # run them one after another on this machine

Each run writes run_logs/<run>-r<seed>.csv and checkpoints/<run>-r<seed>/ under the names that plot_data_efficiency.py,
plot_ablations_50pct.py, bootstrap_table.py and model_specs/latent_paper_configs.json read. Commands are independent,
so they can also be sent to a cluster scheduler one per job. Generate the datasets first (scripts/generate_data.sh;
finding2 needs the redundancy-injected sets as well).
"""

import argparse
import shlex

from bootstrap_table import run_name as bootstrap_run_name
from bootstrap_table import seed as bootstrap_seed
from plot_ablations_50pct import DEPTH_WIDTHS, ENCODERS, WIDTHS, paper_width
from plot_ablations_50pct import run_name as ablation_run_name
from plot_data_efficiency import SEEDS, VOLUMES_K
from plot_data_efficiency import run_name as volume_run_name

DATADIRS = {"karel": "karel16x16-100k-d6-len-15-20-v2", "list": "list_l20v20-100k-d6-u100-len-15-20-v2",
            "rasp": "rasp_v18s16-sweep32k"}
# Redundancy-injected copies (Finding 2): karel_env/rewrite.py, rasp_env/rewrite.py
INJECTED = {"karel": DATADIRS["karel"] + "-rw50", "list": DATADIRS["list"] + "-rw50",
            "rasp": DATADIRS["rasp"] + "-nested-rw50"}
FULL_K = {"karel": 100, "list": 100, "rasp": 32}
HALF_K = {"karel": 50, "list": 50, "rasp": 16}
# RASP's token vocabulary is built at import from these; they must match the dataset (v18, s16).
RASP_ENV = {"RASP_VOCAB_SIZE": "18", "RASP_MAX_SEQ_LEN": "16", "RASP_FAMILY": "base"}

SHARED = ["--solver=trace2code", "--configfile=base", "--hidden_dim=256", "--batch_size=32", "--n_epochs=100",
          "--lr_schedule=constant(1e-4)", "--device=cuda", "--no_wandb"]
DOMAIN_FLAGS = {
    "karel": ["--num_demos=5", "--num_demos_test=6", "--defer_state_expansion", "--overwrite_logs"],
    "list": ["--num_demos=5", "--num_demos_test=6", "--defer_state_expansion", "--overwrite_logs"],
    "rasp": ["--num_demos=4", "--num_demos_test=8", "--state_encoder_type=linear", "--save_last",
             "--pre_enc_layers=3", "--pre_enc_hidden=512"],
}


def encoder_flags(domain, enc, depth):
    """Flags of encoder `enc` ('Logsig', 'Sig', 'LSTM', 'Transformer', 'Fourier') in its paper config."""
    sig = [f"--trace_encoder={enc.lower()}", "--use_set_trans", "--sig_step_norm=unit"]
    if depth != 2:
        sig.append(f"--sig_depth={depth}")
    if domain == "rasp":
        if enc in ("Logsig", "Sig"):
            return sig + ["--include_time", "--time_aug_mode=concat_onehot"]
        flags = {"LSTM": ["--trace_encoder=lstm", "--use_set_trans", "--weight_decay=1.0"],
                 "Transformer": ["--trace_encoder=transformer", "--use_set_trans"],
                 "Fourier": ["--trace_encoder=fourier", "--fourier_modes=9"]}[enc]
        return flags + ["--include_time", "--time_aug_mode=add_sin"]
    if enc in ("Logsig", "Sig"):
        return sig + ["--include_time"]
    return {"LSTM": ["--trace_encoder=lstm", "--use_set_trans"],
            "Transformer": ["--trace_encoder=transformer", "--use_set_trans"],
            "Fourier": ["--trace_encoder=fourier", "--fourier_modes=9"]}[enc] + ["--include_time"]


def run(domain, enc, k, run_name, seed, d=None, depth=2):
    """(env, argv) of one training run on k thousand samples at width d (default: the paper config's)."""
    d = paper_width(domain, enc) if d is None else d
    argv = (["python", "-m", "train", f"--datadir={DATADIRS[domain]}", f"--size_cap={k * 1000}"] + SHARED
            + DOMAIN_FLAGS[domain] + encoder_flags(domain, enc, depth))
    if domain == "rasp" or d != 256:  # Karel / List: one Linear to the hidden size (256) unless d is set
        argv.append(f"--pre_enc_dim={d}")
    argv += [f"--exp_name={run_name}", f"--seed={seed}"]
    return (RASP_ENV if domain == "rasp" else {}), argv


def finding1():
    for domain, volumes in VOLUMES_K.items():
        for enc in ENCODERS:
            for k in volumes:
                for seed in SEEDS:
                    yield run(domain, enc, k, volume_run_name(domain, enc, k), seed)


def ablations():
    """Seed-0 runs of the appendix figures that finding1 does not already make (same names are skipped)."""
    seen = {(argv[-2], argv[-1]) for _, argv in finding1()}
    wanted = []
    for domain in DATADIRS:
        for enc in ENCODERS:  # width_vs_baselines, 50% volume
            wanted += [(domain, enc, HALF_K[domain], d, 2, False) for d in WIDTHS]
        wanted += [(domain, "Logsig", HALF_K[domain], d, m, False) for m in (1, 2, 3) for d in DEPTH_WIDTHS]
        wanted.append((domain, "Sig", FULL_K[domain], paper_width(domain, "Sig"), 2, True))  # sig_vs_logsig
    for domain, enc, k, d, depth, full in wanted:
        name = ablation_run_name(domain, enc, d, depth=depth, full=full)
        if name is None:
            continue
        env, argv = run(domain, enc, k, name, 0, d=d, depth=depth)
        if (argv[-2], argv[-1]) not in seen:
            seen.add((argv[-2], argv[-1]))
            yield env, argv


def finding2():
    """Full-volume paper configs on the injected sets, with a bootstrap pass every 5 epochs from epoch 5 (19 passes)."""
    for domain in DATADIRS:
        for enc in ENCODERS:
            env, argv = run(domain, enc, FULL_K[domain], bootstrap_run_name(domain, enc), bootstrap_seed(domain, enc))
            argv = [f"--datadir={INJECTED[domain]}" if a.startswith("--datadir=")
                    else "--solver=trace2code_bs2" if a == "--solver=trace2code" else a for a in argv]
            argv[-2:-2] = ["--bs2_freq=5", "--bs2_warmup=5", f"--bs2_model_tag={enc.lower()}"]
            yield ({**env, "RASP_FAMILY": "nested"} if domain == "rasp" else env), argv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["finding1", "finding2", "ablations"])
    args = ap.parse_args()
    for env, argv in {"finding1": finding1, "finding2": finding2, "ablations": ablations}[args.which]():
        print(" ".join([f"{k}={v}" for k, v in env.items()] + [shlex.quote(a) for a in argv]))


if __name__ == "__main__":
    main()
