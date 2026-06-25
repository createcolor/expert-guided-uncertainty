"""Uncertainty estimators over arrays shaped ``[members, units, ...classes]``.

These are pure functions over the array interface in :mod:`common.backends`, so
the same code runs on numpy arrays (sample-scale) and torch tensors
(pixel-scale). "members" is the ensemble axis; "units" is samples or pixels.

The total-variance decomposition follows the law of total variance: an
epistemic component from a model ensemble plus an aleatoric component from an
expert ensemble (or expert annotations). Either component may be omitted by
passing ``method=None`` for it.
"""

from __future__ import annotations

from typing import Any

from common.backends import Backend, get_backend


def _normalise_method(method: str | None) -> str | None:
    if method in ("none", "None"):
        return None
    return method


def reduce_members(preds: Any, method: str | None, member_axis: int = 0,
                   backend: Backend | None = None) -> Any:
    """Collapse the member axis into a per-unit uncertainty map.

    ``"var"``       population variance across members (epistemic signal).
    ``"bern_var"``  mean of ``p(1 - p)`` across members (aleatoric signal).
    ``None``        a zero map shaped like the per-unit mean (no contribution).
    """
    method = _normalise_method(method)
    b = backend or get_backend(preds)

    if method == "var":
        return b.var(preds, member_axis)
    if method == "bern_var":
        return b.mean(preds * (1.0 - preds), member_axis)
    if method is None:
        return b.zeros_like(b.mean(preds, member_axis))
    raise ValueError(f"Uncertainty method {method!r} is not supported.")


def total_variance(epist_preds: Any | None, aleat_preds: Any | None = None,
                   epist_method: str | None = "var", aleat_method: str | None = None,
                   member_axis: int = 0) -> Any:
    """Epistemic + aleatoric uncertainty for a single class.

    At least one of ``epist_preds`` / ``aleat_preds`` must be provided, and a
    method must be accompanied by its predictions.
    """
    epist_method = _normalise_method(epist_method)
    aleat_method = _normalise_method(aleat_method)

    assert (epist_preds is not None) or (aleat_preds is not None), \
        "Either epist_preds or aleat_preds must be provided."
    assert (epist_method is None) or (epist_preds is not None), \
        "epist_method was given without epist_preds."
    assert (aleat_method is None) or (aleat_preds is not None), \
        "aleat_method was given without aleat_preds."

    b = get_backend(epist_preds if epist_preds is not None else aleat_preds)

    if epist_preds is not None:
        epist = reduce_members(epist_preds, epist_method, member_axis, b)
    else:
        epist = reduce_members(aleat_preds, None, member_axis, b)

    if (aleat_preds is not None) and (aleat_method is not None):
        aleat = reduce_members(aleat_preds, aleat_method, member_axis, b)
    else:
        aleat = reduce_members(epist_preds, None, member_axis, b)

    return epist + aleat


def multiclass_total_variance(n_classes: int,
                              epist_preds: Any | None, aleat_preds: Any | None = None,
                              epist_method: str | None = "var",
                              aleat_method: str | None = None,
                              member_axis: int = 0, class_axis: int = -2) -> Any:
    """Total variance summed over classes.

    Each class is reduced independently and the per-class maps are summed, which
    matches the single-channel ``n_classes == 1`` case as a degenerate form.
    """
    if n_classes == 1:
        ep = None if epist_preds is None else as_float(epist_preds)
        al = None if aleat_preds is None else as_float(aleat_preds)
        return total_variance(ep, al, epist_method, aleat_method, member_axis)

    b = get_backend(epist_preds if epist_preds is not None else aleat_preds)
    per_class = []
    for i in range(n_classes):
        ep = None if epist_preds is None else as_float(b.select(epist_preds, i, class_axis), b)
        al = None if aleat_preds is None else as_float(b.select(aleat_preds, i, class_axis), b)
        per_class.append(total_variance(ep, al, epist_method, aleat_method, member_axis))

    return b.sum(b.stack(per_class, axis=0), axis=0)


def entropy(probs: Any, class_axis: int = -1, backend: Backend | None = None) -> Any:
    """Shannon entropy ``-sum(p log p)`` over the class axis (clamped for log)."""
    b = backend or get_backend(probs)
    safe = b.clip(probs, 1e-12, 1.0)
    return -b.sum(probs * b.log(safe), class_axis)


def as_float(x: Any, backend: Backend | None = None) -> Any:
    b = backend or get_backend(x)
    return b.as_float(x)
