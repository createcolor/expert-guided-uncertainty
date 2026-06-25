"""The rejection-curve driver, shared across all tasks.

A rejection curve sorts units by uncertainty, discards the most uncertain ones
at a sequence of throwaway rates, and scores the retained units. That skeleton
is identical for every task; only the quality metric differs (accuracy over
retained samples, Dice over retained pixels). The metric is supplied as a
``member_curve`` callback so this module stays task- and backend-agnostic.

Contract for ``member_curve(member_idx, order, keep_counts) -> sequence[float]``:
  - ``order`` is the ascending-uncertainty ordering of this member's units
    (least uncertain first), as returned by the backend's argsort.
  - The callback sorts its own arrays by ``order`` once, then returns one score
    per entry in ``keep_counts`` (the number of least-uncertain units to keep).
  - It must handle ``keep == 0`` (everything thrown away).

Keeping the inner keep-count loop inside the callback lets each task use its
efficient path (contiguous prefix slicing for Dice, a cumulative sum for
accuracy) instead of forcing a generic per-rate re-gather.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from common.backends import get_backend

MemberCurve = Callable[[int, Any, Sequence[int]], Sequence[float]]


def keep_counts_for(n_units: int, rates: Sequence[float]) -> list[int]:
    """Convert throwaway rates to the number of least-uncertain units to keep."""
    return [n_units - round(rate * n_units) for rate in rates]


def sweep_members(order: Any, n_units: int, member_curve: MemberCurve,
                  rates: Sequence[float], member_axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Run ``member_curve`` for every member over a precomputed sort order.

    Use this directly when several curves share one ranking (e.g. per-class
    segmentation curves that all sort by the same class-summed uncertainty), so
    the (expensive) argsort happens once.

    Returns ``(mean_scores, std_scores)``, each of length ``len(rates)``,
    averaged across members.
    """
    b = get_backend(order)
    n_members = order.shape[member_axis]
    counts = keep_counts_for(n_units, rates)

    per_member = []
    for m in range(n_members):
        order_m = b.select(order, m, member_axis)
        per_member.append(list(member_curve(m, order_m, counts)))

    per_member = np.asarray(per_member, dtype=float)  # [n_members, n_rates]
    return per_member.mean(axis=0), per_member.std(axis=0)


def rejection_curve(uncertainty: Any, member_curve: MemberCurve,
                    throwaway_rates: Sequence[float],
                    member_axis: int = 0, unit_axis: int = -1
                    ) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Compute one rejection curve from a per-member uncertainty map.

    Args:
        uncertainty: array shaped ``[members, ..., units]``; sorting happens
            along ``unit_axis`` independently per member.
        member_curve: scoring callback (see module docstring).
        throwaway_rates: rates in ``(0, 1]``; ``0.0`` is prepended automatically.

    Returns ``(rates, mean_scores, std_scores)`` where ``rates`` includes the
    leading ``0.0``.
    """
    b = get_backend(uncertainty)
    n_units = uncertainty.shape[unit_axis]
    order = b.argsort_ascending(uncertainty, unit_axis)  # least -> most uncertain

    rates = [0.0] + list(throwaway_rates)
    mean_scores, std_scores = sweep_members(order, n_units, member_curve, rates, member_axis)
    return rates, mean_scores, std_scores
