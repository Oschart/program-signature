"""
Pure-PyTorch path signatures and log-signatures, truncated at depth 1, 2 or 3 (the paper uses 1-3).

Computes signatures over the piecewise-linear interpolation of the path (the convention of the `signatory`
library), vectorized with bmm/einsum. All operations are differentiable.

Usage:
    from common import signatures as sig

    sig_fn = sig.LogSignature(depth=2)
    out = sig_fn(path)  # path: (B, T, D) -> out: (B, logsignature_channels(D, 2))

    n = sig.signature_channels(D, depth)
    n = sig.logsignature_channels(D, depth)
"""

import torch
import torch.nn as nn


def _mobius(n: int) -> int:
    """Möbius function μ(n)."""
    if n == 1:
        return 1
    factors = []
    d, temp = 2, n
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    for p in factors:
        if n % (p * p) == 0:
            return 0
    return (-1) ** len(factors)


def _witt_dim(D: int, k: int) -> int:
    """Dimension of degree-k free Lie algebra on D generators (Witt's formula)."""
    total = 0
    for d in range(1, k + 1):
        if k % d == 0:
            total += _mobius(k // d) * (D ** d)
    return total // k


def signature_channels(in_channels: int, depth: int) -> int:
    """Number of output channels for the signature transform.

    Equal to D + D^2 + ... + D^depth (geometric series).
    """
    if depth < 1:
        raise ValueError(f"Depth must be >= 1, got {depth}")
    D = in_channels
    if D == 1:
        return depth
    return D * (D ** depth - 1) // (D - 1)


def logsignature_channels(in_channels: int, depth: int) -> int:
    """Number of output channels for the log-signature transform.

    Uses Witt's formula (free Lie algebra dimensions) at each level.
    """
    if depth < 1:
        raise ValueError(f"Depth must be >= 1, got {depth}")
    D = in_channels
    return sum(_witt_dim(D, k) for k in range(1, depth + 1))


# ─── Shared helpers ───────────────────────────────────────────────────

def _compute_increments(path: torch.Tensor):
    """Compute increments and shifted cumulative sum.

    Returns:
        dX: (B, T-1, D) increments
        shifted_cumsum: (B, T-1, D) where shifted_cumsum[t] = sum_{s<t} dX[s]
    """
    dX = path[:, 1:] - path[:, :-1]  # (B, T-1, D)
    cumsum = dX.cumsum(dim=1)  # (B, T-1, D)
    shifted_cumsum = cumsum - dX  # shifted_cumsum[t] = sum_{s=0..t-1} dX[s]
    return dX, shifted_cumsum


def _extract_logsig_level2(S2: torch.Tensor) -> torch.Tensor:
    """Extract log-signature level 2 from signature level 2.

    The log-signature at level 2 is the antisymmetric part of S2,
    represented by the strictly upper-triangular entries of (S2 - S2^T) / 2.
    Note: the within-segment correction (0.5 * Gram) is symmetric and cancels here.

    Args:
        S2: (B, D, D) signature level 2
    Returns:
        (B, D*(D-1)/2) upper-triangular antisymmetric entries
    """
    A = (S2 - S2.transpose(-1, -2)) / 2.0
    D = S2.shape[1]
    idx = torch.triu_indices(D, D, offset=1, device=S2.device)
    return A[:, idx[0], idx[1]]


def _extract_logsig_level3(S1: torch.Tensor, S2: torch.Tensor, S3: torch.Tensor) -> torch.Tensor:
    """Extract log-signature level 3 from signature levels.

    Truncated tensor logarithm at level 3:
        log3 = S3 - 0.5*(S1⊗S2 + S2⊗S1) + (1/3)*(S1⊗S1⊗S1)

    Then Dynkin projection onto the free Lie algebra:
        pi(L) = (2L - L.perm(1,2,0) - L.perm(2,0,1)) / 3

    Finally extract independent Lie basis elements (Lyndon words).
    """
    S1_S2 = torch.einsum('bi,bjk->bijk', S1, S2)
    S2_S1 = torch.einsum('bij,bk->bijk', S2, S1)
    S1_S1_S1 = torch.einsum('bi,bj,bk->bijk', S1, S1, S1)
    log3 = S3 - 0.5 * (S1_S2 + S2_S1) + (1.0 / 3.0) * S1_S1_S1

    L = (2 * log3 - log3.permute(0, 2, 3, 1) - log3.permute(0, 3, 1, 2)) / 3.0

    indices = _lyndon3_indices(D=S1.shape[1], device=S1.device)
    return L[:, indices[:, 0], indices[:, 1], indices[:, 2]]


# Cache for Lyndon word indices
_lyndon3_cache = {}


def _lyndon3_indices(D: int, device: torch.device) -> torch.Tensor:
    """Get indices of Lyndon words of length 3 over alphabet {0, ..., D-1}.

    A Lyndon word is a string that is strictly smaller than all of its rotations.
    Total count = (D^3 - D) / 3.
    """
    cache_key = (D, device)
    if cache_key in _lyndon3_cache:
        return _lyndon3_cache[cache_key]

    indices = []
    for i in range(D):
        for j in range(D):
            for k in range(D):
                word = (i, j, k)
                rot1 = (j, k, i)
                rot2 = (k, i, j)
                if word < rot1 and word < rot2:
                    indices.append(word)

    result = torch.tensor(indices, dtype=torch.long, device=device)
    _lyndon3_cache[cache_key] = result
    return result


# ─── Vectorized backend ──────────────────────────────────────────────

def _sig_level1(dX: torch.Tensor) -> torch.Tensor:
    """Signature level 1: sum of increments. Returns (B, D)."""
    return dX.sum(dim=1)


def _sig_level2(dX: torch.Tensor, shifted_cumsum: torch.Tensor) -> torch.Tensor:
    """Signature level 2 over piecewise-linear path (vectorized).

    Combines:
    - Cross-segment: sum_{s<t} dX[s] ⊗ dX[t]
    - Within-segment: 0.5 * sum_t dX[t] ⊗ dX[t]

    Returns (B, D, D).
    """
    cross = torch.bmm(shifted_cumsum.transpose(1, 2), dX)
    within = 0.5 * torch.bmm(dX.transpose(1, 2), dX)
    return cross + within


def _sig_level3(dX: torch.Tensor, shifted_cumsum: torch.Tensor) -> torch.Tensor:
    """Signature level 3 over piecewise-linear path (vectorized).

    Returns (B, D, D, D).
    """
    dX_outer = torch.einsum('btj,btk->btjk', dX, dX)

    # Term 1: sum_t S2_continuous_before[t] ⊗ dX[t]
    A2_terms = torch.einsum('bti,btj->btij', shifted_cumsum, dX)
    A2_shifted = A2_terms.cumsum(dim=1) - A2_terms
    term1_cross = torch.einsum('btij,btk->bijk', A2_shifted, dX)

    gram_shifted = dX_outer.cumsum(dim=1) - dX_outer
    term1_within = 0.5 * torch.einsum('btij,btk->bijk', gram_shifted, dX)

    # Term 2: 0.5 * sum_t shifted_cumsum[t] ⊗ dX_outer[t]
    term2 = 0.5 * torch.einsum('bti,btjk->bijk', shifted_cumsum, dX_outer)

    # Term 3: (1/6) * sum_t dX[t]^⊗3
    term3 = (1.0 / 6.0) * torch.einsum('bti,btjk->bijk', dX, dX_outer)

    return term1_cross + term1_within + term2 + term3


def _compute_signature_vectorized(path: torch.Tensor, depth: int) -> torch.Tensor:
    """Compute signature using vectorized bmm/einsum operations."""
    B, T, D = path.shape

    if T <= 1:
        return torch.zeros(B, signature_channels(D, depth), device=path.device, dtype=path.dtype)

    dX, shifted_cumsum = _compute_increments(path)
    parts = [_sig_level1(dX)]

    if depth >= 2:
        S2 = _sig_level2(dX, shifted_cumsum)
        parts.append(S2.reshape(B, D * D))

    if depth >= 3:
        S3 = _sig_level3(dX, shifted_cumsum)
        parts.append(S3.reshape(B, D * D * D))

    return torch.cat(parts, dim=1)


def _compute_logsignature_vectorized(path: torch.Tensor, depth: int) -> torch.Tensor:
    """Compute log-signature using vectorized bmm/einsum operations."""
    B, T, D = path.shape

    if T <= 1:
        return torch.zeros(B, logsignature_channels(D, depth), device=path.device, dtype=path.dtype)

    dX, shifted_cumsum = _compute_increments(path)
    S1 = _sig_level1(dX)
    parts = [S1]

    if depth >= 2:
        S2 = _sig_level2(dX, shifted_cumsum)
        parts.append(_extract_logsig_level2(S2))

    if depth >= 3:
        S3 = _sig_level3(dX, shifted_cumsum)
        parts.append(_extract_logsig_level3(S1, S2, S3))

    return torch.cat(parts, dim=1)


# ─── Public API ───────────────────────────────────────────────────────

def _check_depth(depth: int):
    if depth not in (1, 2, 3):
        raise ValueError(f"Depth must be 1, 2 or 3, got {depth}")


class Signature(nn.Module):
    """Compute the truncated signature of a batch of paths.

    Input: (B, T, D) path tensor
    Output: (B, signature_channels(D, depth)) signature tensor
    """

    def __init__(self, depth: int):
        super().__init__()
        _check_depth(depth)
        self.depth = depth

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        if path.ndim != 3:
            raise ValueError(f"Expected 3D input (B, T, D), got {path.ndim}D")
        return _compute_signature_vectorized(path, self.depth)

    def extra_repr(self) -> str:
        return f"depth={self.depth}"


class LogSignature(nn.Module):
    """Compute the truncated log-signature of a batch of paths.

    Input: (B, T, D) path tensor
    Output: (B, logsignature_channels(D, depth)) log-signature tensor
    """

    def __init__(self, depth: int):
        super().__init__()
        _check_depth(depth)
        self.depth = depth

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        if path.ndim != 3:
            raise ValueError(f"Expected 3D input (B, T, D), got {path.ndim}D")
        return _compute_logsignature_vectorized(path, self.depth)

    def extra_repr(self) -> str:
        return f"depth={self.depth}"
