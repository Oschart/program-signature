# The Shape of a Program: Path Signatures for Trace-to-Program Induction

Code for the NeurIPS 2026 paper by Mohamed Ghanem and Bernd Finkbeiner (CISPA Helmholtz Center for Information
Security, Technical University of Munich).

**[Project page](https://oschart.github.io/program-signature/)** · **[Paper](https://arxiv.org/pdf/2611.13005)**

Given a few execution traces of a hidden program, the task is to write a program that reproduces every trace. Each
state of a trace is embedded as a point, so a trace becomes a path, and the path is encoded by its truncated
**log-signature**, a fixed-size summary of its shape from rough path theory. A Transformer decoder reads the trace
encodings and writes the program. The repository contains the log-signature encoder and the LSTM, Transformer and
Fourier baselines, the three domains of the paper (Karel, List, RASP) with their dataset generators, and the scripts
behind the paper's figures.

## Setup

```bash
git clone https://github.com/Oschart/program-signature.git
cd program-signature
pip install -r requirements.txt
```

The paper's runs used Python 3.10 and PyTorch 2.2 on A100 GPUs; the code also runs on Python 3.12 with
PyTorch 2.12. Signatures and log-signatures are computed by a pure-PyTorch implementation (`common/signatures.py`),
so no extra signature library is needed.

## Data

```bash
scripts/generate_data.sh            # all three datasets into data/; or name some of: karel list rasp
```

| domain | dataset | train / test | traces per program (train / test) |
|---|---|---|---|
| Karel | `karel16x16-100k-d6-len-15-20-v2` | 100,000 / 2,500 programs | 5 / 6 |
| List | `list_l20v20-100k-d6-u100-len-15-20-v2` | 100,000 / 2,500 programs | 5 / 6 |
| RASP | `rasp_v18s16-sweep32k` | 32,000 samples / 2,500 programs | 4 / 8 |

Each dataset has a fixed seed and a fixed train/test split. Karel and List programs are sampled from their DSLs and
run from random initial states. Their stored traces cover every branch outcome of the program. Karel and List
generation runs many worker processes for several hours (`WORKERS=<n>` sets the count). In RASP, a trace is a
*sweep*: one input token is raised through every value and each frame pairs the input sequence with the program's
output. Every RASP test program has a behaviour that no training program has.

Finding 2 uses redundancy-injected copies, Karel⁺ / List⁺ / RASP⁺ (`scripts/generate_data.sh karel+ list+ rasp+`,
after the base sets). In each split, half of the programs are rewritten with extra structure that changes no trace,
e.g. `S` → `REPEAT R=1 (S)` or `S` → `IFELSE c (S) ELSE (S)` (Karel, List: `karel_env/rewrite.py`), or an operation
into an equivalent `COMBINE` of two (RASP: `rasp_env/rewrite.py`). This adds about 20% of the tokens. The copies go to
`data/<base>-rw50` and `data/rasp_v18s16-sweep32k-nested-rw50`, and keep each rewritten program's original label.

## Training

One run, the log-signature encoder on Karel with the full training set:

```bash
python -m train --datadir=karel16x16-100k-d6-len-15-20-v2 --size_cap=100000 --solver=trace2code --configfile=base \
    --hidden_dim=256 --batch_size=32 --n_epochs=100 '--lr_schedule=constant(1e-4)' --device=cuda --no_wandb \
    --num_demos=5 --num_demos_test=6 --defer_state_expansion --overwrite_logs --trace_encoder=logsig \
    --use_set_trans --sig_step_norm=unit --include_time --exp_name=KAREL-v2-100k-logsig-unitstep-grid --seed=0
```

`python reproduce.py finding1` prints the command of every run in the paper's main experiment, so it is the place to
look up any configuration. The flags that differ between encoders and domains:

| flag | meaning |
|---|---|
| `--trace_encoder` | `logsig` (ours), `sig`, `lstm`, `transformer`, `fourier` |
| `--sig_depth` | truncation level of the (log-)signature (paper: 2) |
| `--sig_step_norm=unit` | rescale every step of the embedded path to unit length before the signature |
| `--include_time`, `--time_aug_mode` | positional code added to the path (`add_sin`) or concatenated as one-hot channels (`concat_onehot`, RASP) |
| `--use_set_trans` | attention across the traces of one program (all encoders except Fourier) |
| `--pre_enc_dim` | width of the embedded path (default: `--hidden_dim`) |
| `--size_cap` | number of training samples used |

Each run writes `run_logs/<exp_name>-r<seed>.csv` (one row per epoch; test semantic accuracy in the
`ID[...]-program_total_match` column) and `checkpoints/<exp_name>-r<seed>/` (`best_model.pth`, `config.json`). A
decoded program counts as correct if it reproduces every given trace when run from that trace's initial state. To
evaluate a checkpoint only, add `--test_only --checkpoint=checkpoints/<run>/best_model.pth` to its training command:
it prints the test metrics and leaves the run's log and checkpoints untouched.

RASP runs need `RASP_VOCAB_SIZE=18 RASP_MAX_SEQ_LEN=16` in the environment (the commands from `reproduce.py` set
them). On macOS, add `--num_workers=0`.

## Reproducing the paper

**Finding 1, accuracy against training volume.** 144 runs: 3 domains × 4 encoders × 4 training volumes × 3 seeds,
100 epochs each.

```bash
python reproduce.py finding1 > finding1.sh   # one command per line: run in sequence, or one job per line
bash finding1.sh
python plot_data_efficiency.py               # table + plots/data_efficiency.png
```

The plotted value is each run's best test accuracy over its 100 epochs (the epoch of `best_model.pth`), averaged
over the 3 seeds. The datasets have no validation split, so that epoch is chosen on the test set.

**Finding 2, bootstrapping.** `--solver=trace2code_bs2` trains as above and, every `--bs2_freq` epochs, decodes
every training program greedily, runs the decoded program on all stored traces of that program, and replaces the
label when the decoded program is strictly shorter and reproduces every trace. The pass counts (programs shortened,
tokens removed) are printed and logged in the `bs-bs_*` columns of the run's CSV, and the shortened training set is
written to `data/<datadir>-bs-<bs2_model_tag>/`. The paper's table: 12 runs (3 injected datasets × 4 encoders, full
volume, a pass every 5 epochs from epoch 5), seed 0 except the Karel⁺ Transformer, whose seed-0 run had not converged
after 100 epochs (seed 1).

```bash
python reproduce.py finding2 > finding2.sh && bash finding2.sh
python bootstrap_table.py                    # tokens removed and labels shortened at each run's last pass
```

**Finding 3, latent geometry.** After Finding 1, plot the embedded traces of the seed-0, full-volume models listed
in `model_specs/latent_paper_configs.json` (t-SNE, five test programs per domain):

```bash
python latent_view.py --device=cuda --models=Karel-Logsig,Karel-LSTM,Karel-Transformer,Karel-Fourier \
    --plot_dir=plots/latent_view/final/karel
```

Drop `--models` for all three domains (the appendix figure). Every encoder's embedded states get the log-signature
encoder's unit-step normalization, so the four panels are comparable. The script also prints, per model, the share
of each state's five nearest states (over 20 test programs) that lie on its own trace.

**Appendix ablations.** Signature vs log-signature, truncation depth × width, and width for every encoder, seed 0:

```bash
python reproduce.py ablations > ablations.sh && bash ablations.sh   # 69 runs on top of Finding 1's
python plot_ablations_50pct.py --out_dir plots/ablations_50pct
```

**Appendix runtime.** Forward-pass time of each trace encoder (its Karel configuration) on one trace of up to
100,000 states; the paper's timings are from one NVIDIA A100:

```bash
python plot_encoder_runtime.py --device=cuda   # timings -> cache/encoder_runtime.json, figure -> plots/encoder_runtime.png
```

## Repository layout

| path | contents |
|---|---|
| `train.py` | training entry point (`--solver=trace2code` or `trace2code_bs2`) |
| `reproduce.py`, `plot_*.py`, `bootstrap_table.py`, `latent_view.py` | paper experiments, figures and table |
| `models/tracers.py` | state encoders: CNN for Karel grids, MLP / linear for List and RASP states |
| `models/encoders.py` | trace encoders: log-signature and signature (`LogsigEncoder`, `SigEncoder`), LSTM, Transformer, Fourier |
| `models/prog_decoder.py`, `models/decoders.py` | the full model with attention across traces; grammar-masked program decoder |
| `trainers/` | training loop, evaluation, bootstrapping |
| `karel_env/`, `list_env/`, `rasp_env/` | DSLs, interpreters, dataset generators and label rewriters (`rewrite.py`) |
| `common/signatures.py` | pure-PyTorch signatures and log-signatures |

## Citation

```bibtex
@inproceedings{ghanem2026shape,
  title     = {The Shape of a Program: Path Signatures for Trace-to-Program Induction},
  author    = {Ghanem, Mohamed and Finkbeiner, Bernd},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
