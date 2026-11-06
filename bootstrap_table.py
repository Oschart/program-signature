"""Finding 2 table: redundancy in the ground-truth labels surfaced by bootstrapping (--solver=trace2code_bs2) on the
redundancy-injected datasets Karel+ / List+ / RASP+, one seed per cell (runs: `python reproduce.py finding2`).

    dTokens   cumulative % reduction of the total training-label token count  (CSV column bs-bs_cum_pct_tokens_reduced)
    dSamples  % of training labels shortened at least once                    (CSV column bs-bs_cum_pct_samples_reduced)

both read at the last bootstrap pass of run_logs/<run>-r<seed>.csv.

    python bootstrap_table.py
"""

import argparse
import csv
from pathlib import Path

from plot_ablations_50pct import ENCODERS
from plot_data_efficiency import run_name as volume_run_name

DOMAINS = {"karel": "Karel+", "list": "List+", "rasp": "RASP+"}
FULL_K = {"karel": 100, "list": 100, "rasp": 32}
# Seed 0 everywhere except Karel+ Transformer: its seed-0 run had not converged after the 100 epochs every cell gets.
SEED = {("karel", "Transformer"): 1}


def seed(domain, enc):
    return SEED.get((domain, enc), 0)


def run_name(domain, enc):
    """Run name (without -r<seed>) of `enc`'s bootstrapping run: its full-volume paper config on the injected set."""
    name = volume_run_name(domain, enc, FULL_K[domain])
    if domain == "rasp":
        return name.replace("_sweep32k_cap32k", "_bs2_sweep32k-nested-rw50_cap32k")
    return name.replace("-v2-", "-v2rw50-") + "-bs2"


def last_pass(run_logs, run, seed=0):
    """(dTokens, dSamples, number of passes) at the last bootstrap pass of the log's last run. Fails loud on a
    missing log or one without a pass."""
    path = Path(run_logs) / f"{run}-r{seed}.csv"
    rows = list(csv.DictReader(open(path)))
    starts = [i for i, r in enumerate(rows) if r.get("epoch") == "0"]
    rows = rows[starts[-1]:] if starts else rows
    passes = [r for r in rows if r.get("bs-bs_cum_pct_tokens_reduced") not in ("", None)]
    if not passes:
        raise ValueError(f"{path}: no bootstrap pass logged")
    last = passes[-1]
    return float(last["bs-bs_cum_pct_tokens_reduced"]), float(last["bs-bs_cum_pct_samples_reduced"]), len(passes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_logs", default="run_logs")
    args = ap.parse_args()

    print(f"{'':12s}" + "".join(f"{name:>24s}" for name in DOMAINS.values()))
    print(f"{'Encoder':12s}" + f"{'dTokens %':>13s}{'dSamples %':>11s}" * len(DOMAINS))
    for enc in ENCODERS:
        cells = [last_pass(args.run_logs, run_name(dom, enc), seed(dom, enc)) for dom in DOMAINS]
        print(f"{enc:12s}" + "".join(f"{tok:13.2f}{smp:11.2f}" for tok, smp, _ in cells)
              + f"   ({', '.join(str(n) for _, _, n in cells)} passes)")


if __name__ == "__main__":
    main()
