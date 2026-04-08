"""Shared numeric conversion helpers for device-agnostic evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _maybe_to_cpu(value: Any) -> Any:
    """Detach and move tensor-like values to CPU if supported."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value


def to_numpy(value: Any, dtype: Any = None) -> np.ndarray:
    """Convert scalar/tensor/array-like values into a NumPy array on host memory."""
    if isinstance(value, np.ndarray):
        arr = value
    else:
        value = _maybe_to_cpu(value)

        if hasattr(value, "to_numpy"):
            arr = value.to_numpy()
        elif hasattr(value, "__array__"):
            arr = np.asarray(value)
        else:
            arr = np.asarray(value)

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def to_float(value: Any) -> float:
    """Convert metric-like values (including tensors/dicts) to a scalar float."""
    if isinstance(value, (int, float, np.floating)):
        return float(value)

    if isinstance(value, dict):
        if not value:
            return float("nan")
        vals = [to_float(v) for v in value.values()]
        return float(np.mean(vals))

    value = _maybe_to_cpu(value)

    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass

    arr = to_numpy(value)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))
