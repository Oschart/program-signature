"""RASP dataset generator (sweep view). Native evaluation only (no Tracr).

Walks the shuffled family (RASP_FAMILY) and keeps one program per behaviour (native outputs on --n_probe random
inputs) whose output is not constant; each kept program gets K sweeps
(rasp_env/sweep.py). Behaviours are distinct across the whole dataset, so every test program is a new behaviour;
the train/test split is random over the kept behaviours. --num_train_samples repeats each training program with
freshly drawn sweeps (the family has only 8,316 behaviours at V=18, L=16).
Writes:
  processed_dataset_{train,valid}.pkl  s_h (K, V, 2L) int8 sweep frames [input sequence, output sequence], dummy a_h
  rasp_io_{train,valid}.pkl            every sweep input (K x V) with its output: PTM = the decoded program
                                       reproduces every frame of every sweep
  rasp_data_info.json                  d_model = 2L, T = V, view = sweep

  RASP_VOCAB_SIZE=18 RASP_MAX_SEQ_LEN=16 python -m rasp_env.sweep_generator --dir_name=rasp_v18s16-sweep32k \
      --num_train_samples=32000
"""
import argparse, json, os, pickle, random
from collections import Counter

import numpy as np
import torch as th

from rasp_env import rasp_grammar as G
from rasp_env import sampler as S
from rasp_env.sweep import NONE_VALUE, sample_sweeps


def _program_dict(spec, pid, sweeps, dsl):
    prog_ids = th.tensor([dsl.token2int[t] for t in G.program_to_tokens(spec)], dtype=th.int64)
    K, T, _ = sweeps.shape
    return {"program_id": pid, "program": prog_ids, "program_len": int(prog_ids.shape[0]),
            "s_h": th.from_numpy(sweeps.astype(np.int8)),                    # (K, V, 2L)
            "a_h": th.zeros((K, T - 1), dtype=th.int8),                     # dummy (STracer ignores a_h)
            "s_h_len": th.full((K,), T, dtype=th.int64)}


def _io_entry(spec, sweeps, L):
    """Every sweep input with its output (the frames' second half; -1 back to None)."""
    frames = sweeps.reshape(-1, 2 * L).astype(int)
    return {"inputs": frames[:, :L].tolist(),
            "gt_outputs": [[None if v == NONE_VALUE else v for v in y] for y in frames[:, L:].tolist()],
            "gt_token_str": G.canonical_str(spec)}


def _add_train_copies(train, n, rng, k, L, V):
    """Fill the training split to n samples by repeating every program with freshly drawn sweeps (new contexts and
    positions): n // len(train) samples each, one more for a random n % len(train); shuffled, so a --size_cap
    prefix is a random subset."""
    reps = np.full(len(train), n // len(train))
    reps[rng.choice(len(train), n % len(train), replace=False)] += 1
    out = []
    for (spec, sweeps), r in zip(train, reps):
        out.append((spec, sweeps))
        out += [(spec, sample_sweeps(spec, rng, k, L, V)) for _ in range(r - 1)]
    return [out[i] for i in rng.permutation(len(out))]


def generate(config):
    V, L = config.vocab_size, config.max_seq_len
    dsl = G.RaspDSL(vocab_size=V, max_seq_len=L)
    probe = S.make_probe_inputs(V, L, config.n_probe, seed=config.probe_seed)
    specs = list(G.iter_all_specs(V, include_binary=True, family=G.RASP_FAMILY))
    random.Random(config.seed).shuffle(specs)
    rng = np.random.default_rng(config.seed)
    want = config.num_train + config.num_val
    kept, seen, reasons = [], set(), Counter()
    for spec in specs:
        if len(kept) == want:
            break
        key = S.behavior_key(spec, probe)
        if not S.is_nonconstant(key):
            reasons[S.REASON_CONSTANT] += 1
        elif key in seen:
            reasons[S.REASON_DUP_BEHAVIOR] += 1
        else:
            seen.add(key)
            kept.append((spec, sample_sweeps(spec, rng, config.num_demo_per_program, L, V)))
    if len(kept) < want:
        print(f"[WARN] the family has only {len(kept)} distinct non-constant behaviours (< {want} asked); "
              f"valid keeps {config.num_val}, train gets the rest")
    # the walk meets behaviours with many spellings (x < 7 = x <= 6, ...) first, so split at random, not in order
    random.Random(config.seed).shuffle(kept)
    splits = {"valid": kept[:config.num_val], "train": kept[config.num_val:]}
    n_train_programs = len(splits["train"])
    if config.num_train_samples > n_train_programs:     # own stream: the valid split stays the same
        splits["train"] = _add_train_copies(splits["train"], config.num_train_samples,
                                            np.random.default_rng(config.seed + 1), config.num_demo_per_program, L, V)

    out_dir = os.path.join("data", config.dir_name)
    os.makedirs(out_dir, exist_ok=True)
    max_prog_len = max(len(G.program_to_tokens(s)) for s, _ in kept)
    for split, items in splits.items():
        program_data, io_side = [], {}
        for j, (spec, sweeps) in enumerate(items):
            pid = f"{split}_{j}_{G.canonical_str(spec).replace(' ', '_')}"
            program_data.append(_program_dict(spec, pid, sweeps, dsl))
            io_side[pid] = _io_entry(spec, sweeps, L)
        with open(os.path.join(out_dir, f"processed_dataset_{split}.pkl"), "wb") as f:
            pickle.dump({"program_data": program_data, "max_demo_length": V, "max_program_len": max_prog_len,
                         "num_program_tokens": len(dsl.int2token)}, f)
        with open(os.path.join(out_dir, f"rasp_io_{split}.pkl"), "wb") as f:
            pickle.dump(io_side, f)
    meta = {"d_model": 2 * L, "max_seq_len": L, "vocab_size": V, "T": V, "family": G.RASP_FAMILY, "view": "sweep",
            "num_demo_per_program": config.num_demo_per_program, "num_program_tokens": len(dsl.int2token),
            "num_train": len(splits["train"]), "num_train_programs": n_train_programs, "num_val": len(splits["valid"])}
    with open(os.path.join(out_dir, "rasp_data_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "gen_config.json"), "w") as f:
        json.dump(vars(config), f, indent=2, sort_keys=True)
    print(f"{G.RASP_FAMILY} family, {len(specs)} specs walked until {len(kept)} kept "
          f"(train {n_train_programs} programs in {len(splits['train'])} samples / valid {len(splits['valid'])}); "
          f"rejected {dict(reasons)}; "
          f"traces (K, T, D) = {kept[0][1].shape} -> {out_dir}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dir_name", type=str, required=True, help="data/<dir_name>/ (rasp_v<V>s<L>-...sweep)")
    p.add_argument("--vocab_size", type=int, default=G.RASP_VOCAB_SIZE)
    p.add_argument("--max_seq_len", type=int, default=G.RASP_MAX_SEQ_LEN)
    p.add_argument("--num_demo_per_program", type=int, default=8, help="K sweeps, at K distinct positions")
    p.add_argument("--num_train", type=int, default=32000)
    p.add_argument("--num_val", type=int, default=2500)
    p.add_argument("--num_train_samples", type=int, default=0,
                   help="fill the training split to this many samples, each program again with fresh sweeps "
                        "(0 = one sample per program)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_probe", type=int, default=48, help="probe inputs for the behaviour key")
    p.add_argument("--probe_seed", type=int, default=1234)
    config = p.parse_args()
    if config.vocab_size != G.RASP_VOCAB_SIZE or config.max_seq_len != G.RASP_MAX_SEQ_LEN:
        raise SystemExit(f"[FATAL] --vocab_size/--max_seq_len ({config.vocab_size}/{config.max_seq_len}) != "
                         f"RASP_VOCAB_SIZE/RASP_MAX_SEQ_LEN env ({G.RASP_VOCAB_SIZE}/{G.RASP_MAX_SEQ_LEN}); "
                         f"the combine modulus is read from the env at import")
    generate(config)


if __name__ == "__main__":
    main()
