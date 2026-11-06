"""Redundant labels for RASP bootstrapping (RASP_FAMILY=nested): re-spell ops of a base-family dataset as equivalent
COMBINEs, so trace2code_bs2 has labels to shorten. The sweep sets keep one spelling per behaviour, so on their own
only 2.3% of programs have a shorter family spelling.

One rewrite rule: a categorical op X of a program becomes COMBINE(cf, A, B) over two categorical ops that equals X on
every input, e.g. CMPC EQ c -> COMBINE min (CMPC GEQ c) (CMPC LEQ c), or X -> COMBINE max X (CMPC LT C1). The table
holds every such (cf, A, B): candidates that match X on probe inputs, kept only if they also pass an exact pointwise
check. A --rate share of the records of each split gets one rewrite (a random op slot, a random expansion of its op).
Outputs are unchanged on every input, so sweep traces (frames = input + output) are too; each rewritten label is
re-checked against every stored sweep frame anyway.

Writes data/<src>-nested-rw<rate %>/ with the source's files: labels and max_program_len updated, rasp_io_*.pkl
entries get the new gt_token_str plus orig_token_str (the source label), rasp_data_info.json gets family=nested and
the rewrite stats.

  RASP_VOCAB_SIZE=18 RASP_MAX_SEQ_LEN=16 python -m rasp_env.rewrite --src=rasp_v18s16-sweep32k --rate=0.5
"""
import argparse, dataclasses, itertools, json, os, pickle, random, shutil
from collections import Counter, defaultdict

import numpy as np
import torch as th

from rasp_env import rasp_grammar as G


def _read(op):
    """(offset of the position op reads, value function). Categorical ops read one position: MAP / CMPC / TOK their
    own, SHIFT the one `offset` away (None past the ends)."""
    if op.op == G.OP_SHIFT:
        return op.offset, lambda x: x
    return 0, lambda x: G._op_eval_py(op, [x])[0]


def _equal_everywhere(X, cf, A, B, V, modulus):
    """COMBINE(cf, A, B) == X for every value of each position they read: values in 1..V, plus None at a nonzero
    offset (past the end). Offsets get independent values, a superset of what inputs reach, so passing is a proof."""
    reads = [_read(o) for o in (X, A, B)]
    offs = sorted({o for o, _ in reads})
    domains = [list(range(1, V + 1)) + ([None] if o != 0 else []) for o in offs]
    fn = G._combine_fn(cf, modulus)
    for vals in itertools.product(*domains):
        at = dict(zip(offs, vals))
        x, a, b = (None if at[o] is None else f(at[o]) for o, f in reads)
        if fn(a, b) != x:
            return False
    return True


def expansion_table(V, L, n_probe=8, seed=0):
    """{op: [(cf, A, B), ...]} for every categorical op: all COMBINEs over categorical ops equal to it everywhere."""
    modulus = V + 1
    _, cfs, ops, _ = G.family_parts(V, "base")
    rng = random.Random(seed)
    probes = [[rng.randint(1, V) for _ in range(L)] for _ in range(n_probe)]
    out = {o: [tuple(G._op_eval_py(o, p)) for p in probes] for o in ops}
    by_key = defaultdict(list)
    for cf in cfs:
        fn = G._combine_fn(cf, modulus)
        for A, B in itertools.product(ops, ops):
            key = tuple(tuple(fn(a, b) for a, b in zip(ya, yb)) for ya, yb in zip(out[A], out[B]))
            by_key[key].append((cf, A, B))
    return {X: [e for e in by_key.get(tuple(out[X]), []) if _equal_everywhere(X, *e, V, modulus)] for X in ops}


def _slots(spec, table):
    """Op slots a rewrite can re-spell: 'op' (unary), 'left' / 'right' (binary), when the op has an expansion."""
    names = ["op"] if spec.kind == "unary" else ["left", "right"]
    return [s for s in names if table.get(getattr(spec, s))]


def rewrite_spec(spec, table, rng):
    slot = rng.choice(_slots(spec, table))
    cf, A, B = rng.choice(table[getattr(spec, slot)])
    expr = G.RaspSpec("binary", combine=cf, left=A, right=B)
    return (expr if slot == "op" else dataclasses.replace(spec, **{slot: expr})), slot


def rewrite_split(program_data, io_side, table, rate, rng, dsl):
    """Rewrite round(rate * N) random records in place; return stats."""
    specs = [G.tokens_to_spec(io_side[p["program_id"]]["gt_token_str"].split()) for p in program_data]
    eligible = [i for i, s in enumerate(specs) if _slots(s, table)]
    n = round(rate * len(program_data))
    if n > len(eligible):
        raise SystemExit(f"[FATAL] --rate={rate} asks for {n} rewrites but only {len(eligible)} records have an op "
                         f"with an expansion")
    tokens_before = sum(int(p["program_len"]) for p in program_data)
    kinds, examples = Counter(), []
    for i in rng.sample(eligible, n):
        p, entry = program_data[i], io_side[program_data[i]["program_id"]]
        new, slot = rewrite_spec(specs[i], table, rng)
        bad = sum(G.evaluate_spec_py(new, x) != y for x, y in zip(entry["inputs"], entry["gt_outputs"]))
        if bad:
            raise SystemExit(f"[FATAL] rewrite of {entry['gt_token_str']!r} to {G.canonical_str(new)!r} changes "
                             f"{bad} stored sweep outputs")
        toks = G.program_to_tokens(new)
        p["program"] = th.tensor([dsl.token2int[t] for t in toks], dtype=th.int64)
        p["program_len"] = len(toks)
        entry["gt_token_str"] = " ".join(toks)
        old = specs[i].op if slot == "op" else getattr(specs[i], slot)
        kinds[old.op] += 1
        if len(examples) < 5:
            examples.append((G.canonical_str(specs[i]), entry["gt_token_str"]))
    tokens_after = sum(int(p["program_len"]) for p in program_data)
    return {"records": len(program_data), "rewritten": n, "eligible": len(eligible),
            "tokens_before": tokens_before, "tokens_after": tokens_after,
            "rewritten_op_kinds": dict(kinds), "examples": examples}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--src", required=True, help="base-family RASP dataset under data/")
    ap.add_argument("--rate", type=float, required=True, help="share of records per split that get one rewrite")
    ap.add_argument("--dst", default=None, help="output dir under data/ (default <src>-nested-rw<rate %%>)")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()
    src = os.path.join("data", cfg.src)
    with open(os.path.join(src, "rasp_data_info.json")) as f:
        meta = json.load(f)
    V, L = meta["vocab_size"], meta["max_seq_len"]
    if meta.get("family", "base") != "base" or (V, L) != (G.RASP_VOCAB_SIZE, G.RASP_MAX_SEQ_LEN):
        raise SystemExit(f"[FATAL] {cfg.src} is family {meta.get('family', 'base')!r} at V={V}, L={L}; needs a base "
                         f"set with RASP_VOCAB_SIZE/RASP_MAX_SEQ_LEN = {G.RASP_VOCAB_SIZE}/{G.RASP_MAX_SEQ_LEN}")
    dst = os.path.join("data", cfg.dst or f"{cfg.src}-nested-rw{round(cfg.rate * 100)}")
    os.makedirs(dst, exist_ok=True)

    table = expansion_table(V, L)
    sizes = [len(v) for v in table.values()]
    print(f"expansion table: {sum(sizes)} COMBINEs for {sum(s > 0 for s in sizes)}/{len(table)} categorical ops "
          f"(per op min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)})")
    dsl, rng, stats = G.RaspDSL(vocab_size=V, max_seq_len=L), random.Random(cfg.seed), {}
    data = {}
    for split in ("train", "valid"):
        with open(os.path.join(src, f"processed_dataset_{split}.pkl"), "rb") as f:
            ds = pickle.load(f)
        with open(os.path.join(src, f"rasp_io_{split}.pkl"), "rb") as f:
            io_side = pickle.load(f)
        for e in io_side.values():
            e["orig_token_str"] = e["gt_token_str"]
        stats[split] = st = rewrite_split(ds["program_data"], io_side, table, cfg.rate, rng, dsl)
        data[split] = (ds, io_side)
        b, a, n = st["tokens_before"], st["tokens_after"], st["records"]
        print(f"{split}: {st['rewritten']}/{n} records rewritten ({st['rewritten'] / n:.1%}; {st['eligible']} "
              f"eligible), tokens {b} -> {a} (mean length {b / n:.2f} -> {a / n:.2f}; undoing every rewrite removes "
              f"{1 - b / a:.1%} of tokens), rewritten ops {st['rewritten_op_kinds']}")
        for old, new in st["examples"]:
            print(f"    {old}  ->  {new}")
    max_len = max(int(p["program_len"]) for ds, _ in data.values() for p in ds["program_data"])
    for split, (ds, io_side) in data.items():
        ds["max_program_len"] = max_len
        with open(os.path.join(dst, f"processed_dataset_{split}.pkl"), "wb") as f:
            pickle.dump(ds, f)
        with open(os.path.join(dst, f"rasp_io_{split}.pkl"), "wb") as f:
            pickle.dump(io_side, f)
    meta.update(family="nested", rewrite={"src": cfg.src, "rate": cfg.rate, "seed": cfg.seed,
                                          **{s: {k: v for k, v in st.items() if k != "examples"}
                                             for s, st in stats.items()}})
    with open(os.path.join(dst, "rasp_data_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if os.path.exists(os.path.join(src, "gen_config.json")):
        shutil.copyfile(os.path.join(src, "gen_config.json"), os.path.join(dst, "gen_config.json"))
    print(f"-> {dst} (train with RASP_FAMILY=nested)")


if __name__ == "__main__":
    main()
