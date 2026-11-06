"""Redundant labels for bootstrapping (Karel / List, the shared imperative grammar): re-spell part of a --rate share
of the programs with structure that changes no action, so trace2code_bs2 has labels to shorten. The RASP
counterpart is rasp_env/rewrite.py.

Every rule is a rule of karel_env/dsl/canonicalize.py run backwards, and the exactness check is that canonicalizing
the rewritten label gives back the original (v2 labels are canonical). Rules, each at a random place:
  ifelse_dup   S                       -> IFELSE c (S) ELSE (S)        S = 1-2 statements, c random, sometimes `not c`
  repeat1      S                       -> REPEAT R=1 (S)
  push_tail    IFELSE c (A) ELSE (B) X -> IFELSE c (A X) ELSE (B X);   IF c (A) X -> IFELSE c (A X) ELSE (X)
  while_wrap   WHILE c X               -> IF c (WHILE c X) | WHILE c (WHILE c X) | REPEAT R=n (WHILE c X)
  repeat_nest  REPEAT R=a*b X          -> REPEAT R=a (REPEAT R=b X)
  not_flip     IFELSE c (A) ELSE (B)   -> IFELSE not c (B) ELSE (A)
A chosen program gets random rules (a random applicable rule, then a random place) until it has grown by --grow x its
length, never past --max_len tokens. The extra structure costs interpreter calls and the interpreter stops at
MAX_FUNC_CALL, so each rewritten label is also replayed on its stored demos (up to 3 tries, else left unchanged).

Writes data/<src>-rw<rate %>/: the split pickles with the new labels, dataset_info.json and the split id files, and
rewrite_info.json (arguments, per-split stats, the original label of every rewritten program).

  python -m karel_env.rewrite --src=karel16x16-100k-d6-len-15-20-v2
  python -m karel_env.rewrite --src=list_l20v20-100k-d6-u100-len-15-20-v2
"""
import argparse, json, os, pickle, random, shutil, time
from collections import Counter
from multiprocessing import get_context

import numpy as np
import torch as th
import yaml

from common.env_spec import get_env_spec
from karel_env.program_dataset import build_split_pickles, read_marked_split
from karel_env.dsl import get_DSL, dsl_version_for
from karel_env.dsl.canonicalize import parse_program, program_code, body_tokens, canonicalize
from trainers.trace2code_bs2 import _exec_program, _get_dsl

RULES = ["ifelse_dup", "repeat1", "push_tail", "while_wrap", "repeat_nest", "not_flip"]
_W = {}  # worker globals: env, env_kwargs, spec, conds


def _body_rewrites(b, rule, rng):
    """Every single application of `rule` at this body or inside it, as new bodies."""
    out = []
    for i, n in enumerate(b):
        k = n[0]
        if rule in ("ifelse_dup", "repeat1"):
            for p in (1, 2):
                S = b[i:i + p]
                if len(S) < p:
                    break
                if rule == "repeat1":
                    new = ("REPEAT", 1, S)
                else:
                    c = rng.choice(_W["conds"])
                    new = ("IFELSE", (c,) if rng.random() < 0.7 else ("not", c), S, S)
                out.append(b[:i] + [new] + b[i + p:])
        if rule == "push_tail" and k in ("IF", "IFELSE"):
            for q in (1, 2):
                X = b[i + 1:i + 1 + q]
                if len(X) < q:
                    break
                new = ("IFELSE", n[1], n[2] + X, n[3] + X) if k == "IFELSE" else ("IFELSE", n[1], n[2] + X, X)
                out.append(b[:i] + [new] + b[i + 1 + q:])
        if rule == "while_wrap" and k == "WHILE":
            for new in (("IF", n[1], [n]), ("WHILE", n[1], [n]), ("REPEAT", rng.randint(2, 5), [n])):
                out.append(b[:i] + [new] + b[i + 1:])
        if rule == "repeat_nest" and k == "REPEAT":
            out += [b[:i] + [("REPEAT", a, [("REPEAT", n[1] // a, n[2])])] + b[i + 1:]
                    for a in range(2, n[1]) if n[1] % a == 0]
        if rule == "not_flip" and k == "IFELSE" and n[1][0] != "not":
            out.append(b[:i] + [("IFELSE", ("not", n[1][0]), n[3], n[2])] + b[i + 1:])
        if k == "IFELSE":
            out += [b[:i] + [(k, n[1], nb, n[3])] + b[i + 1:] for nb in _body_rewrites(n[2], rule, rng)]
            out += [b[:i] + [(k, n[1], n[2], nb)] + b[i + 1:] for nb in _body_rewrites(n[3], rule, rng)]
        elif k != "A":
            out += [b[:i] + [(k, n[1], nb)] + b[i + 1:] for nb in _body_rewrites(n[2], rule, rng)]
    return out


def rewrite_program(code, target_len, max_len, rng):
    """Grow `code` towards target_len tokens (past it by at most 2, never past max_len); returns (new code, rules)."""
    body, rules = parse_program(code), []
    size = lambda bd: len(body_tokens(bd)) + 4                           # + DEF run m( m)
    while size(body) < target_len:
        options = []
        for r in RULES:
            cands = [nb for nb in _body_rewrites(body, r, rng) if size(body) < size(nb) <= min(target_len + 2, max_len)
                     and canonicalize(program_code(nb)) == code]
            if cands:
                options.append((r, cands))
        if not options:
            break
        r, cands = rng.choice(options)
        body = rng.choice(cands)
        rules.append(r)
    return program_code(body), rules


def _replays(code, s_0, a_h, s_h_len):
    """The label reproduces every stored demo's actions (as trace2code_bs2._verify_one); s_0: (K, state) starts."""
    for k in range(s_0.shape[0]):
        if _W["env"] == "karel":
            s0 = _W["spec"].expand_state(th.from_numpy(s_0[k]).view(1, 1, -1))[0, 0].permute(1, 2, 0).numpy().astype(bool)
        else:
            s0 = s_0[k].astype(np.int8)
        n = int(s_h_len[k])
        res = _exec_program(_W["env"], _W["env_kwargs"], code, s0, monitor_a=a_h[k, :n - 1].tolist())
        if res is None or res[2] != n:
            return False
    return True


def _init(env, version, datadir):
    spec = get_env_spec(env)
    spec.parse_data_dims(datadir)
    dsl = get_DSL(environment=env, version=version)
    _get_dsl(env)  # the DSL _exec_program replays with: build it too, so forked workers inherit it (no table race)
    _W.update(env=env, spec=spec, conds=list(dsl.conditional_functions),
              env_kwargs={"L_max": spec.L_max, "max_val": spec.max_val} if env == "list" else {})


def _work(job):
    i, code, target_len, max_len, s_0, a_h, s_h_len, seed = job
    rng = random.Random(seed)
    for _ in range(3):
        new, rules = rewrite_program(code, target_len, max_len, rng)
        if new != code and _replays(new, s_0, a_h, s_h_len):
            return i, new, rules
    return i, None, []


def rewrite_split(program_data, env, version, datadir, cfg, rng, i2t, t2i, workers):
    codes = [" ".join(i2t[int(t)] for t in p["program"][:int(p["program_len"])]) for p in program_data]
    # room for the smallest rewrite (+4 tokens), and canonical (the exactness check compares canonical forms)
    eligible = [i for i, c in enumerate(codes) if len(c.split()) + 4 <= cfg.max_len and canonicalize(c) == c]
    n = round(cfg.rate * len(codes))
    if n > len(eligible):
        raise SystemExit(f"[FATAL] --rate={cfg.rate} asks for {n} rewrites but only {len(eligible)} programs have room")
    chosen = rng.sample(eligible, n)
    # numpy, not tensors: torch sends tensors to workers through shared memory, which 50K jobs exhaust in a
    # container (small /dev/shm); start states copied out of s_h so a job does not carry the whole trace
    jobs = [(i, codes[i], min(len(codes[i].split()) + round(cfg.grow * len(codes[i].split())), cfg.max_len),
             cfg.max_len, program_data[i]["s_h"][:, 0].numpy().copy(), program_data[i]["a_h"].numpy(),
             program_data[i]["s_h_len"].numpy(), rng.randrange(2 ** 31)) for i in chosen]
    with get_context("fork").Pool(workers, initializer=_init, initargs=(env, version, datadir)) as pool:
        results = list(pool.imap_unordered(_work, jobs, chunksize=16))
    orig, rules, failed = {}, Counter(), []
    for i, new, used in results:
        if new is None:
            failed.append(codes[i])
            continue
        p = program_data[i]
        p["program"] = th.tensor([t2i[t] for t in new.split()], dtype=p["program"].dtype)
        p["program_len"] = len(new.split())
        orig[p["program_id"]] = codes[i]
        rules.update(used)
    before = sum(len(c.split()) for c in codes)
    after = sum(int(p["program_len"]) for p in program_data)
    examples = [(codes[i], new) for i, new, _ in results if new][:4]
    return {"programs": len(codes), "eligible": len(eligible), "chosen": n, "rewritten": len(orig),
            "failed": len(failed), "tokens_before": before, "tokens_after": after,
            "max_len_after": max(int(p["program_len"]) for p in program_data),
            "rules": dict(rules.most_common()), "failed_examples": failed[:5], "examples": examples}, orig


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--src", required=True, help="Karel / List dataset under data/ with marked train/test splits")
    ap.add_argument("--rate", type=float, default=0.5, help="share of programs per split that get rewritten")
    ap.add_argument("--grow", type=float, default=0.45, help="a rewritten program grows by this x its length")
    ap.add_argument("--max_len", type=int, default=45, help="token cap on rewritten labels (configs/base.yml dsl)")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--dst", default=None, help="output dir under data/ (default <src>-rw<rate %%>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    cfg = ap.parse_args()
    src = os.path.join("data", cfg.src)
    with open(os.path.join(src, "dataset_info.json")) as f:
        env = json.load(f)["environment"]
    if env not in ("karel", "list"):
        raise SystemExit(f"[FATAL] {cfg.src} is a {env!r} dataset; this rewriter covers karel and list")
    version = dsl_version_for(cfg.src)
    dsl = get_DSL(environment=env, version=version)
    i2t, t2i = dsl.int2token, {t: i for i, t in enumerate(dsl.int2token)}
    dst = os.path.join("data", cfg.dst or f"{cfg.src}-rw{round(cfg.rate * 100)}")
    os.makedirs(dst, exist_ok=True)
    _init(env, version, cfg.src)  # builds the DSL parser tables here, before the workers fork
    rng, info = random.Random(cfg.seed), {"args": vars(cfg), "environment": env, "dsl_version": version}
    if not all(os.path.exists(os.path.join(src, f"processed_dataset_{split}.pkl")) for split in cfg.splits.split(",")):
        # A fresh dataset holds only data.hdf5: build the split pickles as training does on its first load.
        with open(os.path.join("configs", "base.yml")) as f:
            dsl_max_len = yaml.safe_load(f)["dsl"]["max_program_len"]
        build_split_pickles(cfg.src, read_marked_split(cfg.src), dsl_max_len)
    for split in cfg.splits.split(","):
        t = time.time()
        with open(os.path.join(src, f"processed_dataset_{split}.pkl"), "rb") as f:
            ds = pickle.load(f)
        print(f"{split}: loaded {len(ds['program_data'])} programs in {time.time() - t:.0f} s", flush=True)
        st, orig = rewrite_split(ds["program_data"], env, version, cfg.src, cfg, rng, i2t, t2i, cfg.workers)
        b, a, n = st["tokens_before"], st["tokens_after"], st["programs"]
        print(f"{split}: {st['rewritten']}/{n} programs rewritten ({st['rewritten'] / n:.1%}; {st['failed']} "
              f"failed replay 3 times, left unchanged), tokens {b} -> {a} (mean {b / n:.2f} -> {a / n:.2f}; "
              f"undoing every rewrite removes {1 - b / a:.1%}), max length {st['max_len_after']}, rules {st['rules']}")
        for o, r in st["examples"]:
            print(f"    {o}\n -> {r}")
        ds["max_program_len"] = max(ds["max_program_len"], st["max_len_after"])
        with open(os.path.join(dst, f"processed_dataset_{split}.pkl"), "wb") as f:
            pickle.dump(ds, f)
        info[split] = {**{k: v for k, v in st.items() if k != "examples"}, "orig": orig}
        print(f"{split}: written in {time.time() - t:.0f} s", flush=True)
        del ds
    for f in ("dataset_info.json", "train_ids.txt", "test_ids.txt"):
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    with open(os.path.join(dst, "rewrite_info.json"), "w") as f:
        json.dump(info, f)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
