"""Per-program filters of the RASP generator (rasp_env/sweep_generator.py).

  - non-constant  (native output varies across a fixed probe set -> K-diverse)
  - behavior key  (hashable native outputs over the probe set; one program is kept per behaviour)

The probe set is deterministic given (vocab_size, max_seq_len, probe_seed) so
behavior keys are comparable across processes.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from rasp_env import rasp_grammar as G

# Rejection reason codes (kept as constants so counters are consistent).
REASON_CONSTANT = "constant_output"
REASON_DUP_BEHAVIOR = "dup_behavior"


def make_probe_inputs(vocab_size: int, max_seq_len: int, n_probe: int,
                      seed: int = 1234) -> List[List[int]]:
    """Deterministic set of distinct fixed-length probe inputs (values 1..V)."""
    rng = random.Random(seed)
    out: List[List[int]] = []
    seen = set()
    max_distinct = vocab_size ** max_seq_len
    n_probe = min(n_probe, max_distinct)
    guard = 0
    while len(out) < n_probe and guard < n_probe * 100:
        guard += 1
        x = tuple(rng.randint(1, vocab_size) for _ in range(max_seq_len))
        if x in seen:
            continue
        seen.add(x)
        out.append(list(x))
    return out


def behavior_key(spec: G.RaspSpec, probe_inputs: List[List[int]]) -> Optional[Tuple]:
    """Tuple of native outputs over the probe set (hashable), or None if native
    evaluation raises on any probe input."""
    outs = []
    for x in probe_inputs:
        try:
            o = G.evaluate_spec_py(spec, x)
        except Exception:
            return None
        outs.append(tuple(o))
    return tuple(outs)


def is_nonconstant(behavior: Optional[Tuple]) -> bool:
    """True iff the output varies across the probe inputs (needed so the K demos
    of a program give diverse evidence)."""
    return behavior is not None and len(set(behavior)) > 1
