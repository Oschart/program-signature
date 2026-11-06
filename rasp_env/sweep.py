"""Sweep traces of RASP programs: the program's response to raising one input
token through every value.

A sweep fixes a random input (the context), picks a position p and sets its token to 1, 2, ..., V in turn. Each
frame is [input sequence, output sequence] as scalars; an output with no value (SHIFT past the end) is -1.
"""
import numpy as np

from rasp_env import rasp_grammar as G

NONE_VALUE = -1


def sweep_trace(spec, context, p, vocab_size=G.RASP_VOCAB_SIZE):
    """(V, 2L) frames of one sweep of position p through tokens 1..V."""
    x, frames = list(context), []
    for v in range(1, vocab_size + 1):
        x[p] = v
        y = G.evaluate_spec_py(spec, x)
        frames.append(list(x) + [NONE_VALUE if u is None else u for u in y])
    return np.array(frames, dtype=np.float32)


def sample_sweeps(spec, rng, k, seq_len=G.RASP_MAX_SEQ_LEN, vocab_size=G.RASP_VOCAB_SIZE):
    """(k, V, 2L): k sweeps at k distinct positions, each in its own random context (tokens 1..V)."""
    positions = rng.choice(seq_len, size=k, replace=False)
    return np.stack([sweep_trace(spec, rng.integers(1, vocab_size + 1, seq_len), int(p), vocab_size)
                     for p in positions])
