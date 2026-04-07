"""Transformer Key cache Gram matrix for KV eviction certification.

H = K K^T + eta I where K is the Key tensor (seq_len x d_head).
The eviction mask partitions tokens into retained and omitted.
chi < 1 certifies that the eviction is licensed: the reduced
attention matrix faithfully represents the retained context.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np


def build_gram(
    K: Any,
    *,
    eta: float,
) -> np.ndarray:
    """H = K K^T + eta * I (symmetric positive definite)."""
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.ndim != 2:
        raise ValueError("K must be 2D (seq_len, d_head)")
    n = int(K_arr.shape[0])
    h = K_arr @ K_arr.T + eta * np.eye(n)
    return 0.5 * (h + h.T)


def partition_from_mask(
    keep_mask: Sequence[int] | np.ndarray,
) -> tuple[list[int], list[int]]:
    """Retained (mask==1) and omitted (mask==0) indices from eviction mask."""
    m = np.asarray(keep_mask, dtype=int).reshape(-1)
    retained = sorted(int(i) for i, v in enumerate(m) if v == 1)
    omitted = sorted(int(i) for i, v in enumerate(m) if v == 0)
    if not retained or not omitted:
        raise ValueError("need at least one retained and one omitted token")
    return retained, omitted


def hash_key_and_mask(
    K: np.ndarray,
    keep_mask: np.ndarray,
    *,
    key_dtype: str = "float32",
) -> str:
    """SHA-256 of K (as key_dtype bytes) || mask (as uint8 bytes)."""
    K_arr = np.asarray(K)
    n = int(K_arr.shape[0])
    m = np.asarray(keep_mask).reshape(-1)
    if m.shape[0] != n:
        raise ValueError("mask length must match K.shape[0]")
    raw = np.asarray(K_arr, dtype=np.dtype(key_dtype)).tobytes("C")
    raw += np.asarray(m, dtype=np.uint8).tobytes("C")
    return hashlib.sha256(raw).hexdigest()
