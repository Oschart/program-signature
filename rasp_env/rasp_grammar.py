"""RASP grammar core.

Self-contained functional-DSL definition + a fixed token serialization that is
the decoder's target grammar, and a native evaluator of its programs.

Program family (bounded, nesting depth <= 2, categorical + numerical):

  Level-1 SOps (the building blocks), each a small parameterized op on the input:
    MAP(fn)      elementwise Map on tokens         (categorical)
    SHIFT(off)   Aggregate(Select(idx, idx+off))   (categorical)  -- look-ahead/back
    CMPC(cmp,c)  Map(x -> int(x cmp c))            (categorical, boolean)
    TOK          rasp.tokens (identity)            (categorical)  -- branch-only
    HIST         SelectorWidth(Select(tok,tok,EQ)) (numerical)    -- per-token count
    COUNT(cmp)   SelectorWidth(Select(tok,tok,cmp))(numerical)    -- rank/count

  Programs:
    unary  : a single level-1 op is the whole program.
    binary : SequenceMap(cf, left, right) over two *categorical* level-1 ops.
             (binary combine is categorical-only: Tracr does not compile
             numerical arithmetic combines.)

Serialization (flat token stream, roundtrip-parseable):
    unary  : DEF <op-tokens> m)
    binary : DEF COMBINE CF_<cf> <left op-tokens> <right op-tokens> m)
  where op-tokens are:
    MAP -> [MAP, FN_<fn>]      SHIFT -> [SHIFT, OFF_<off>]
    CMPC-> [CMPC, <cmp>, C<c>] COUNT -> [COUNT, <cmp>]   HIST/TOK -> [HIST]/[TOK]

DEF / m) double as the decoder's hardcoded SOS / EOS (models/decoders.py reads
T2I["DEF"] and T2I["m)"]), so every program is [DEF] + body + [m)].

Nested family (RASP_FAMILY=nested; datadirs tagged `-nested`): the base family, except
that a COMBINE branch may itself be COMBINE(cf, op, op) over categorical ops (one level
deeper, same vocabulary). Too large to enumerate; its datasets are base datasets whose
ops rasp_env/rewrite.py re-spells as equivalent COMBINEs (redundant labels for
trace2code_bs2, Finding 2).
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Generation constants. These MUST match the generated dataset (the datadir tag
# `rasp_v<vocab_size>s<max_seq_len>` documents them). RaspDSL builds its token
# vocab from vocab_size, and the tokenizer/decoder inherit that vocab, so a
# mismatch between these and the dataset surfaces as a vocab-size error.
#
# Read once at import from the environment, so set them before the process starts
# (the paper's datasets: RASP_VOCAB_SIZE=18 RASP_MAX_SEQ_LEN=16; reproduce.py's commands set them).
RASP_VOCAB_SIZE = int(os.environ.get("RASP_VOCAB_SIZE", "5"))   # input token values 1..V
RASP_MAX_SEQ_LEN = int(os.environ.get("RASP_MAX_SEQ_LEN", "8"))  # real tokens/input (residual T = this + 1 BOS)
# Program family: "base", or "nested" for the redundancy-injected RASP+ set (see the module docstring). Same
# contract as the two above: read once at import.
RASP_FAMILY = os.environ.get("RASP_FAMILY", "base")
if RASP_FAMILY not in ("base", "nested"):
    raise ValueError(f"RASP_FAMILY={RASP_FAMILY!r}: expected 'base' or 'nested'")

# ---- serialization alphabet (op / param token names) -----------------------
SOS_TOKEN = "DEF"            # decoder SOS (models/decoders.py hardcodes T2I["DEF"])
EOS_TOKEN = "m)"             # decoder EOS (models/decoders.py hardcodes T2I["m)"])
COMBINE_TOKEN = "COMBINE"

OP_MAP, OP_SHIFT, OP_CMPC, OP_HIST, OP_COUNT, OP_TOK = (
    "MAP", "SHIFT", "CMPC", "HIST", "COUNT", "TOK")
OP_TOKENS = [OP_MAP, OP_SHIFT, OP_CMPC, OP_HIST, OP_COUNT, OP_TOK]

MAP_FNS = ["inc", "dec", "dbl", "mod2", "mod3"]          # FN_<name>
OFFSETS = [-3, -2, -1, 1, 2, 3]                          # OFF_<v>  (m<abs> for negatives)
COMPARISONS = ["EQ", "LT", "LEQ", "GT", "GEQ", "NEQ"]    # bare tokens (shared)
COUNT_CMPS = ["LT", "LEQ", "GT", "GEQ", "NEQ"]           # EQ == HIST, excluded here
COMBINE_FNS = ["add", "sub", "max", "min", "gt", "eq"]   # CF_<name>


def _off_token(off: int) -> str:
    return f"OFF_{'m' if off < 0 else ''}{abs(off)}"


def _parse_off_token(tok: str) -> int:
    body = tok[len("OFF_"):]
    return -int(body[1:]) if body.startswith("m") else int(body)


def build_vocab(vocab_size: int = RASP_VOCAB_SIZE) -> List[str]:
    """The fixed decoder token vocabulary (order stable; DEF first, m) second), the same for both families.

    `<pad>` is appended later by the Tokenizer, not here.
    """
    toks: List[str] = [SOS_TOKEN, EOS_TOKEN, COMBINE_TOKEN]
    toks += OP_TOKENS
    toks += [f"FN_{f}" for f in MAP_FNS]
    toks += [_off_token(o) for o in OFFSETS]
    toks += list(COMPARISONS)
    toks += [f"C{c}" for c in range(1, vocab_size + 1)]      # CMPC constants
    toks += [f"CF_{c}" for c in COMBINE_FNS]
    # sanity: no duplicates
    assert len(toks) == len(set(toks)), "duplicate token in RASP vocab"
    return toks


# ----------------------------------------------------------------------------
# Spec representation
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OpSpec:
    """A level-1 SOp: op name + its params (only the relevant fields are set)."""
    op: str
    fn: Optional[str] = None        # MAP
    offset: Optional[int] = None    # SHIFT
    cmp: Optional[str] = None       # CMPC / COUNT
    const: Optional[int] = None     # CMPC


@dataclass(frozen=True)
class RaspSpec:
    """A program: a single unary OpSpec, or a binary SequenceMap of two branches.
    A branch is an OpSpec, or (nested family) a binary RaspSpec over two OpSpecs."""
    kind: str                       # "unary" | "binary"
    op: Optional[OpSpec] = None                 # unary
    combine: Optional[str] = None               # binary combine fn name
    left: Optional["OpSpec | RaspSpec"] = None  # binary
    right: Optional["OpSpec | RaspSpec"] = None


# ----------------------------------------------------------------------------
# Serialization: spec <-> flat token list
# ----------------------------------------------------------------------------
def _op_to_tokens(op: OpSpec) -> List[str]:
    if op.op == OP_MAP:
        return [OP_MAP, f"FN_{op.fn}"]
    if op.op == OP_SHIFT:
        return [OP_SHIFT, _off_token(op.offset)]
    if op.op == OP_CMPC:
        return [OP_CMPC, op.cmp, f"C{op.const}"]
    if op.op == OP_COUNT:
        return [OP_COUNT, op.cmp]
    if op.op == OP_HIST:
        return [OP_HIST]
    if op.op == OP_TOK:
        return [OP_TOK]
    raise ValueError(f"unknown op {op.op!r}")


def _expr_to_tokens(x) -> List[str]:
    """An OpSpec, or a binary RaspSpec as COMBINE CF_<cf> <left> <right> (branches may nest)."""
    if isinstance(x, OpSpec):
        return _op_to_tokens(x)
    return [COMBINE_TOKEN, f"CF_{x.combine}"] + _expr_to_tokens(x.left) + _expr_to_tokens(x.right)


def _tokens_to_op(tokens: List[str], i: int) -> Tuple[OpSpec, int]:
    """Parse one OpSpec starting at index i; return (op, next_index)."""
    op = tokens[i]
    if op == OP_MAP:
        return OpSpec(OP_MAP, fn=tokens[i + 1][len("FN_"):]), i + 2
    if op == OP_SHIFT:
        return OpSpec(OP_SHIFT, offset=_parse_off_token(tokens[i + 1])), i + 2
    if op == OP_CMPC:
        return OpSpec(OP_CMPC, cmp=tokens[i + 1], const=int(tokens[i + 2][1:])), i + 3
    if op == OP_COUNT:
        return OpSpec(OP_COUNT, cmp=tokens[i + 1]), i + 2
    if op == OP_HIST:
        return OpSpec(OP_HIST), i + 1
    if op == OP_TOK:
        return OpSpec(OP_TOK), i + 1
    raise ValueError(f"unknown op token {op!r} at {i}")


def _tokens_to_expr(tokens: List[str], i: int):
    """Parse an OpSpec or a (possibly nested) COMBINE starting at index i; return (expr, next_index)."""
    if tokens[i] != COMBINE_TOKEN:
        return _tokens_to_op(tokens, i)
    left, j = _tokens_to_expr(tokens, i + 2)
    right, k = _tokens_to_expr(tokens, j)
    return RaspSpec("binary", combine=tokens[i + 1][len("CF_"):], left=left, right=right), k


def program_to_tokens(spec: RaspSpec) -> List[str]:
    """Serialize a spec to [DEF] + body + [m)]."""
    body: List[str]
    if spec.kind == "unary":
        body = _op_to_tokens(spec.op)
    elif spec.kind == "binary":
        body = _expr_to_tokens(spec)
    else:
        raise ValueError(f"unknown spec kind {spec.kind!r}")
    return [SOS_TOKEN] + body + [EOS_TOKEN]


def tokens_to_spec(tokens: List[str]) -> RaspSpec:
    """Inverse of program_to_tokens. Raises on malformed streams (used to count
    parse failures in behavioral eval)."""
    if not tokens or tokens[0] != SOS_TOKEN or tokens[-1] != EOS_TOKEN:
        raise ValueError(f"program must be wrapped in {SOS_TOKEN}..{EOS_TOKEN}")
    body = tokens[1:-1]
    expr, j = _tokens_to_expr(body, 0)
    if j != len(body):
        raise ValueError(f"trailing tokens after parse: {body[j:]}")
    return expr if isinstance(expr, RaspSpec) else RaspSpec("unary", op=expr)


def canonical_str(spec: RaspSpec) -> str:
    """Whitespace-joined token string; used for de-dup keying."""
    return " ".join(program_to_tokens(spec))


# ----------------------------------------------------------------------------
# Enumeration of specs
# ----------------------------------------------------------------------------
def _iter_op_specs(vocab_size: int, categorical_only: bool) -> List[OpSpec]:
    ops: List[OpSpec] = []
    for f in MAP_FNS:
        ops.append(OpSpec(OP_MAP, fn=f))
    for o in OFFSETS:
        ops.append(OpSpec(OP_SHIFT, offset=o))
    for c in COMPARISONS:
        for k in range(1, vocab_size + 1):
            ops.append(OpSpec(OP_CMPC, cmp=c, const=k))
    ops.append(OpSpec(OP_TOK))
    if not categorical_only:
        ops.append(OpSpec(OP_HIST))
        for c in COUNT_CMPS:
            ops.append(OpSpec(OP_COUNT, cmp=c))
    return ops


def family_parts(vocab_size: int = RASP_VOCAB_SIZE, family: str = RASP_FAMILY):
    """(unary ops, combine fns, left ops, right ops). The family is every unary op
    alone plus COMBINE(cf, left, right) for every cf x left x right (nested: the base
    parts; a branch may also be COMBINE(cf, op, op), see RaspGrammarChecker).

    unary = all ops except the trivial standalone TOK (identity); left = right =
    the categorical ops (TOK allowed as a branch).
    """
    if family not in ("base", "nested"):
        raise ValueError(f"unknown RASP family {family!r}")
    cat_ops = _iter_op_specs(vocab_size, categorical_only=True)
    unary_ops = [o for o in _iter_op_specs(vocab_size, categorical_only=False)
                 if o.op != OP_TOK]
    return unary_ops, COMBINE_FNS, cat_ops, cat_ops


def iter_all_specs(vocab_size: int = RASP_VOCAB_SIZE, include_binary: bool = True,
                   family: str = RASP_FAMILY):
    """Deterministically enumerate the whole program family (see `family_parts`)."""
    if family == "nested":
        raise ValueError("the nested family is too large to enumerate; nested datasets come from rasp_env.rewrite")
    unary_ops, combine_fns, left_ops, right_ops = family_parts(vocab_size, family)
    for o in unary_ops:
        yield RaspSpec("unary", op=o)
    if include_binary:
        for cf in combine_fns:
            for a, b in itertools.product(left_ops, right_ops):
                yield RaspSpec("binary", combine=cf, left=a, right=b)


# ----------------------------------------------------------------------------
# Native evaluation
# ----------------------------------------------------------------------------
def _map_fn(name: str):
    return {
        "inc": lambda x: x + 1,
        "dec": lambda x: x - 1,
        "dbl": lambda x: x * 2,
        "mod2": lambda x: x % 2,
        "mod3": lambda x: x % 3,
    }[name]


def _cmp_pred(name: str):
    return {
        "EQ": lambda x, c: x == c,
        "LT": lambda x, c: x < c,
        "LEQ": lambda x, c: x <= c,
        "GT": lambda x, c: x > c,
        "GEQ": lambda x, c: x >= c,
        "NEQ": lambda x, c: x != c,
    }[name]


def _combine_fn(name: str, modulus: int):
    """None-safe categorical combine fns with bounded output in {0..modulus-1}
    (or {0,1} for gt/eq). None can appear from a SHIFT branch at a boundary;
    native SequenceMap short-circuits None (fn not called), but we guard anyway."""
    def guard(f):
        return lambda a, b: (None if (a is None or b is None) else f(a, b))
    return {
        "add": guard(lambda a, b: (a + b) % modulus),
        "sub": guard(lambda a, b: (a - b) % modulus),
        "max": guard(lambda a, b: max(a, b)),
        "min": guard(lambda a, b: min(a, b)),
        "gt": guard(lambda a, b: int(a > b)),
        "eq": guard(lambda a, b: int(a == b)),
    }[name]


# Tracr's native RASP semantics, reimplemented (verified against Tracr's evaluator across the whole family):
# rasp.Select(keys, queries, cmp) selects (query i, key j) where cmp(key_j,
# query_i); Aggregate over a one-hot selector returns the selected value;
# SelectorWidth counts selected keys per query; SequenceMap/Map propagate None.
def _op_eval_py(op: OpSpec, inp: List[int]) -> List:
    n = len(inp)
    if op.op == OP_TOK:
        return list(inp)
    if op.op == OP_MAP:
        f = _map_fn(op.fn)
        return [f(x) for x in inp]
    if op.op == OP_SHIFT:
        off = op.offset
        return [inp[i + off] if 0 <= i + off < n else None for i in range(n)]
    if op.op == OP_CMPC:
        pred = _cmp_pred(op.cmp)
        c = op.const
        return [int(pred(x, c)) for x in inp]
    if op.op == OP_HIST:
        return [sum(1 for kj in inp if kj == xi) for xi in inp]
    if op.op == OP_COUNT:
        pred = _cmp_pred(op.cmp)
        return [sum(1 for kj in inp if pred(kj, xi)) for xi in inp]
    raise ValueError(f"unknown op {op.op!r}")


def _expr_eval_py(x, inp: List[int], modulus: int) -> List:
    if isinstance(x, OpSpec):
        return _op_eval_py(x, inp)
    cf = _combine_fn(x.combine, modulus)
    return [cf(a, b) for a, b in zip(_expr_eval_py(x.left, inp, modulus), _expr_eval_py(x.right, inp, modulus))]


def evaluate_spec_py(spec: RaspSpec, input_seq: List[int],
                     modulus: int = RASP_VOCAB_SIZE + 1) -> List:
    """Per-position outputs of `spec` on `input_seq` (None where a SHIFT runs past the ends)."""
    inp = list(input_seq)
    if spec.kind == "unary":
        return _op_eval_py(spec.op, inp)
    if spec.kind == "binary":
        return _expr_eval_py(spec, inp, modulus)
    raise ValueError(f"unknown spec kind {spec.kind!r}")


def trim_program_ids(ids, dsl) -> "List[str] | None":
    """List[int] -> token list [DEF, ..., m)] up to the first EOS, or None if
    the stream is malformed (no leading DEF / SOS, or no EOS before it ends).

    Used by the behavioral-eq metric (trainers/trace2code.py)."""
    sos, eos = dsl.token2int[SOS_TOKEN], dsl.token2int[EOS_TOKEN]
    if len(ids) == 0 or ids[0] != sos:
        return None
    out = [ids[0]]
    for t in ids[1:]:
        out.append(t)
        if t == eos:
            return [dsl.int2token[i] for i in out]
    return None


# ----------------------------------------------------------------------------
# DSL shim (duck-typed for Tokenizer / config_utils; no yacc / DSLData)
# ----------------------------------------------------------------------------
class RaspDSL:
    """Minimal DSL interface the trace2code pipeline expects.

    Provides the token vocab (`int2token`/`token2int`) that config_utils turns
    into `dsl_tokens`, plus `action_functions=[]` (the tracer ignores actions,
    so num_agent_actions == 1). No yacc grammar -- decoding is masked to the
    generator family by `RaspGrammarChecker`.
    """
    def __init__(self, seed=None, environment: str = "rasp",
                 vocab_size: int = RASP_VOCAB_SIZE,
                 max_seq_len: int = RASP_MAX_SEQ_LEN):
        self.environment = "rasp"
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.int2token = build_vocab(vocab_size)
        self.token2int = {t: i for i, t in enumerate(self.int2token)}
        self.action_functions: List = []

    # convenience (mirrors the other DSLs' surface used by a few call sites)
    def str2intseq(self, s: str) -> List[int]:
        return [self.token2int[t] for t in s.split()]

    def intseq2str(self, ids) -> str:
        return " ".join(self.int2token[int(i)] for i in ids)

    def spec_to_intseq(self, spec: RaspSpec) -> List[int]:
        return [self.token2int[t] for t in program_to_tokens(spec)]

    def intseq_to_spec(self, ids) -> RaspSpec:
        return tokens_to_spec([self.int2token[int(i)] for i in ids])


def _op_trie(ops: List[OpSpec], T2I: Dict[str, int], then: Dict) -> Dict:
    """Prefix tree over the ops' token sequences whose leaves continue into `then`.
    Ops of one kind have a fixed arity, so no op's tokens are a prefix of another's."""
    root: Dict = {}
    for op in ops:
        toks = [T2I[t] for t in _op_to_tokens(op)]
        node = root
        for t in toks[:-1]:
            node = node.setdefault(t, {})
        node[toks[-1]] = then
    return root


class RaspGrammarChecker:
    """Grammar mask restricting RASP decoding to the generator family.

    A prefix graph over the token sequences of every `iter_all_specs` program: the
    feasible next tokens are the children of the current node, and after EOS only
    `<pad>`. Built from `family_parts` with shared sub-trees (unary op | COMBINE cf
    -> left op -> right op, then EOS), so its size grows with the number of ops, not
    of programs. The decoder can therefore only emit family programs -- the ones the
    generator samples from -- never parseable non-family programs (bare TOK, COMBINE
    over HIST/COUNT branches).

    Duck-types the interface models/decoders.py calls (get_initial_checker_state
    / forward / allowed_tokens). The state is a 1-element list holding the current
    node; `forward` advances it in place.
    `allowed_tokens` returns a (1, 1, V) bool mask, True == masked out, matching
    PySyntaxChecker's contract. Nested family: each branch may also be COMBINE cf op op."""
    def __init__(self, T2I: Dict[str, int], vocab_size: int = RASP_VOCAB_SIZE,
                 family: str = RASP_FAMILY):
        self.num_tokens = len(T2I)
        self.pad = T2I["<pad>"]
        self._done: Dict = {}                       # after EOS: only <pad>
        unary_ops, combine_fns, left_ops, right_ops = family_parts(vocab_size, family)
        end = {T2I[EOS_TOKEN]: self._done}

        def branch(ops, then):
            node = _op_trie(ops, T2I, then)
            if family == "nested":
                inner = _op_trie(ops, T2I, _op_trie(ops, T2I, then))
                node[T2I[COMBINE_TOKEN]] = {T2I[f"CF_{cf}"]: inner for cf in combine_fns}
            return node

        left = branch(left_ops, branch(right_ops, end))
        root = _op_trie(unary_ops, T2I, end)
        root[T2I[COMBINE_TOKEN]] = {T2I[f"CF_{cf}"]: left for cf in combine_fns}
        self._start = {T2I[SOS_TOKEN]: root}         # before SOS
        self._masks: Dict[Tuple[int, ...], "torch.Tensor"] = {}

    def get_initial_checker_state(self):
        return [self._start]

    def forward(self, state, token):
        node = state[0]
        if node is self._done:
            if token != self.pad:
                raise ValueError(f"token {token} after EOS; only <pad> is feasible")
            return state
        if token not in node:
            raise ValueError(f"token {token} does not continue any RASP family program")
        state[0] = node[token]
        return state

    def allowed_tokens(self, state):
        node = state[0]
        key = (self.pad,) if node is self._done else tuple(sorted(node))
        mask = self._masks.get(key)
        if mask is None:
            import torch
            mask = torch.ones((1, 1, self.num_tokens), dtype=torch.bool)
            mask[0, 0, list(key)] = False
            self._masks[key] = mask
        return mask


def _nested_self_check(n: int = 3000, seed: int = 0):
    """RASP_FAMILY=nested cannot be enumerated, so check a random sample: nested programs roundtrip through
    tokens and pass the mask, every base program passes, deeper nesting and numerical branches are blocked."""
    import random
    rng = random.Random(seed)
    T2I = {t: i for i, t in enumerate(build_vocab())}
    T2I["<pad>"] = len(T2I)
    chk = RaspGrammarChecker(T2I)
    _, cfs, cat_ops, _ = family_parts()

    def branch():
        if rng.random() < 0.5:
            return rng.choice(cat_ops)
        return RaspSpec("binary", combine=rng.choice(cfs), left=rng.choice(cat_ops), right=rng.choice(cat_ops))

    def accepts(tokens):
        st = chk.get_initial_checker_state()
        for t in tokens + ["<pad>"]:
            if chk.allowed_tokens(st).view(-1)[T2I[t]]:
                return False
            chk.forward(st, T2I[t])
        return True

    for _ in range(n):
        spec = RaspSpec("binary", combine=rng.choice(cfs), left=branch(), right=branch())
        toks = program_to_tokens(spec)
        assert tokens_to_spec(toks) == spec and accepts(toks), toks
    base = [program_to_tokens(s) for s in iter_all_specs(family="base")]
    assert all(accepts(t) for t in base)
    for bad in ["DEF COMBINE CF_max COMBINE CF_min COMBINE CF_add TOK TOK TOK TOK m)",
                "DEF COMBINE CF_max COMBINE CF_min HIST TOK TOK m)"]:
        tokens_to_spec(bad.split())                  # parses fine ...
        assert not accepts(bad.split()), bad         # ... but the mask rejects it
    print(f"nested: {n} random nested programs roundtrip and pass the mask, so do all {len(base)} base programs; "
          f"deeper nesting and numerical branches are blocked")


if __name__ == "__main__":
    if RASP_FAMILY == "nested":
        _nested_self_check()
        raise SystemExit
    # Quick self-check of the serialization roundtrip (Tracr-independent).
    n = 0
    for spec in iter_all_specs():
        toks = program_to_tokens(spec)
        assert tokens_to_spec(toks) == spec, (spec, toks)
        n += 1
    print(f"roundtrip OK on all {n} enumerated specs")
    print("vocab size:", len(build_vocab()), "tokens")

    # Grammar mask accepts EXACTLY the family: DFS through the decoder-facing
    # interface (copy.copy + forward) collects every program it lets through,
    # which must equal the enumerated family.
    import copy
    T2I = {t: i for i, t in enumerate(build_vocab())}
    T2I["<pad>"] = len(T2I)
    I2T = {i: t for t, i in T2I.items()}
    chk = RaspGrammarChecker(T2I)
    accepted = set()

    def _dfs(state, prefix):
        feasible = (~chk.allowed_tokens(state).view(-1)).nonzero().view(-1).tolist()
        if feasible == [T2I["<pad>"]]:
            accepted.add(" ".join(prefix))
            return
        for tok in feasible:
            child = copy.copy(state)
            chk.forward(child, tok)
            _dfs(child, prefix + [I2T[tok]])

    s0 = chk.get_initial_checker_state()
    chk.forward(s0, T2I[SOS_TOKEN])
    _dfs(s0, [SOS_TOKEN])
    family = {" ".join(program_to_tokens(s)) for s in iter_all_specs()}
    assert accepted == family, (len(accepted), len(family))
    # Parseable-but-non-family programs must be blocked at some step.
    blocked_examples = ["DEF TOK m)", "DEF COMBINE CF_add HIST TOK m)", "DEF COMBINE CF_min TOK COUNT LT m)"]
    for bad in blocked_examples:
        tokens_to_spec(bad.split())                  # parses fine ...
        st = chk.get_initial_checker_state()
        blocked = False
        for t in bad.split():
            if chk.allowed_tokens(st).view(-1)[T2I[t]]:
                blocked = True
                break
            chk.forward(st, T2I[t])
        assert blocked, bad                          # ... but the mask rejects it
    print(f"grammar mask accepts exactly the {len(family)} family programs; non-family programs blocked")
