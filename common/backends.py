"""A minimal array interface with two backends: numpy and torch.

The estimators and the rejection-curve driver are written once against the
``Backend`` interface. Sample-scale tasks (classification, QA) pass numpy
arrays and get the :class:`NumpyBackend`; pixel-scale segmentation passes torch
tensors and gets the :class:`TorchBackend`, keeping GPU execution.

Only the handful of operations actually used downstream are exposed -- this is
deliberately not a general array-abstraction layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Backend:
    """Interface implemented by :class:`NumpyBackend` and :class:`TorchBackend`."""

    name: str

    @staticmethod
    def as_float(x: Any) -> Any: ...
    @staticmethod
    def argsort_ascending(x: Any, axis: int) -> Any: ...
    @staticmethod
    def take_along_axis(x: Any, idx: Any, axis: int) -> Any: ...
    @staticmethod
    def select(x: Any, index: int, axis: int) -> Any: ...
    @staticmethod
    def var(x: Any, axis: int) -> Any: ...
    @staticmethod
    def mean(x: Any, axis: int) -> Any: ...
    @staticmethod
    def sum(x: Any, axis: int) -> Any: ...
    @staticmethod
    def stack(xs: list, axis: int = 0) -> Any: ...
    @staticmethod
    def zeros_like(x: Any) -> Any: ...
    @staticmethod
    def clip(x: Any, lo: float, hi: float) -> Any: ...
    @staticmethod
    def log(x: Any) -> Any: ...
    @staticmethod
    def softplus(x: Any) -> Any: ...
    @staticmethod
    def to_scalar(x: Any) -> float: ...


class NumpyBackend(Backend):
    name = "numpy"

    @staticmethod
    def as_float(x):
        return np.asarray(x, dtype=np.float64)

    @staticmethod
    def argsort_ascending(x, axis):
        return np.argsort(x, axis=axis)

    @staticmethod
    def take_along_axis(x, idx, axis):
        return np.take_along_axis(x, idx, axis=axis)

    @staticmethod
    def select(x, index, axis):
        return np.take(x, index, axis=axis)

    @staticmethod
    def var(x, axis):
        return np.var(x, axis=axis)  # population variance (ddof=0)

    @staticmethod
    def mean(x, axis):
        return np.mean(x, axis=axis)

    @staticmethod
    def sum(x, axis):
        return np.sum(x, axis=axis)

    @staticmethod
    def stack(xs, axis=0):
        return np.stack(xs, axis=axis)

    @staticmethod
    def zeros_like(x):
        return np.zeros_like(x)

    @staticmethod
    def clip(x, lo, hi):
        return np.clip(x, lo, hi)

    @staticmethod
    def log(x):
        return np.log(x)

    @staticmethod
    def softplus(x):
        return np.logaddexp(0.0, x)  # numerically stable log(1 + exp(x))

    @staticmethod
    def to_scalar(x):
        return float(x)


class TorchBackend(Backend):
    name = "torch"

    @staticmethod
    def as_float(x):
        return x.float()

    @staticmethod
    def argsort_ascending(x, axis):
        import torch
        return torch.argsort(x, dim=axis)

    @staticmethod
    def take_along_axis(x, idx, axis):
        import torch
        return torch.take_along_dim(x, idx, dim=axis)

    @staticmethod
    def select(x, index, axis):
        return x.select(axis, index)

    @staticmethod
    def var(x, axis):
        return x.var(dim=axis, correction=0)  # population variance

    @staticmethod
    def mean(x, axis):
        return x.mean(dim=axis)

    @staticmethod
    def sum(x, axis):
        return x.sum(dim=axis)

    @staticmethod
    def stack(xs, axis=0):
        import torch
        return torch.stack(xs, dim=axis)

    @staticmethod
    def zeros_like(x):
        import torch
        return torch.zeros_like(x)

    @staticmethod
    def clip(x, lo, hi):
        return x.clamp(lo, hi)

    @staticmethod
    def log(x):
        return x.log()

    @staticmethod
    def softplus(x):
        import torch.nn.functional as F
        return F.softplus(x)

    @staticmethod
    def to_scalar(x):
        return float(x.item())


def get_backend(x: Any) -> Backend:
    """Return the backend matching ``x`` (a numpy array or a torch tensor)."""
    if isinstance(x, np.ndarray):
        return NumpyBackend()
    # Avoid importing torch unless needed (numpy-only installs stay light).
    type_name = f"{type(x).__module__}.{type(x).__name__}"
    if type_name.startswith("torch."):
        return TorchBackend()
    raise TypeError(f"Unsupported array type for backend dispatch: {type_name}")
