"""Accuracy-rejection curves and AAC for item-level tasks (PubMedQA, BloodyWell).

A vectorised re-implementation of ``get_accs_throwaway_rates`` and
``get_rc_info_averaging_mode`` from the old
``uncertainty_estimation/rejection_curves.py``:

  - items with equal uncertainty are discarded together, most uncertain first;
  - accuracy is computed on the remaining items, for each base-ensemble net;
  - each net's curve is linearly interpolated onto the ``i / n_bins`` grid;
  - curves are averaged over nets.

It reproduces the old output (``tests/test_paired_stats.py`` checks this against
a verbatim copy of the old functions) but is fast enough to run inside a
bootstrap. The old loop's ``break`` once accuracy reaches 1.0 is not needed:
every later point also has accuracy 1.0, so the interpolated curve is the same.
"""

from __future__ import annotations

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 compatibility


def make_grid(n_bins: int = 1000) -> np.ndarray:
    """The old x-axis grid, built exactly as ``x_list`` in the old code."""
    return np.array([i * (1.0 / n_bins) for i in range(n_bins + 1)])


def _curves_for_ranking(uncertainty: np.ndarray, wrong: np.ndarray,
                        grid: np.ndarray) -> np.ndarray:
    """Rejection curves of several nets that share one uncertainty ranking.

    Args:
        uncertainty: ``[N]`` uncertainty per item.
        wrong: ``[nets, N]`` whether each net misclassified each item.
        grid: x-axis points to interpolate onto.

    Returns:
        ``[nets, len(grid)]`` accuracy curves.
    """
    n_items = uncertainty.shape[0]
    _, inverse = np.unique(uncertainty, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    group = n_groups - 1 - inverse.ravel()          # 0 = most uncertain group

    counts = np.bincount(group, minlength=n_groups)
    wrong_per_group = np.zeros((n_groups, wrong.shape[0]))
    np.add.at(wrong_per_group, group, wrong.T.astype(float))

    left = n_items - np.cumsum(counts)               # items remaining after each group
    wrong_left = wrong.sum(axis=1)[:, None] - np.cumsum(wrong_per_group, axis=0).T
    keep = left > 0

    rates = np.concatenate(([0.0], 1.0 - left[keep] / n_items, [1.0]))
    initial = 1.0 - wrong.sum(axis=1) / n_items
    curves = np.empty((wrong.shape[0], grid.shape[0]))
    for net in range(wrong.shape[0]):
        accs = np.concatenate(([initial[net]], 1.0 - wrong_left[net, keep] / left[keep], [1.0]))
        curves[net] = np.interp(grid, rates, accs)
    return curves


def net_curves(uncertainty: np.ndarray, wrong: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Per-net accuracy-rejection curves, ``[nets, len(grid)]``.

    ``uncertainty`` is ``[N]`` for methods that rank all nets' items the same
    way (e.g. total variance) or ``[nets, N]`` for methods that produce one
    ranking per base net (e.g. MCMC, HUQ).
    """
    if uncertainty.ndim == 1:
        return _curves_for_ranking(uncertainty, wrong, grid)
    return np.vstack([
        _curves_for_ranking(uncertainty[net], wrong[net:net + 1], grid)
        for net in range(wrong.shape[0])
    ])


def area_above_curve(curves: np.ndarray, grid: np.ndarray, r_max: float = 0.8) -> np.ndarray:
    """AAC = integral of ``1 - accuracy`` over ``[0, r_max]``, per curve."""
    curves = np.atleast_2d(curves)
    mask = grid <= r_max
    x = np.append(grid[mask], r_max)
    end = np.array([np.interp(r_max, grid, curve) for curve in curves])[:, None]
    y = np.concatenate((curves[:, mask], end), axis=1)
    return _trapezoid(1.0 - y, x, axis=1)


def method_aac(uncertainty: np.ndarray, wrong: np.ndarray, grid: np.ndarray,
               r_max: float = 0.8) -> float:
    """AAC of the net-averaged curve (equal to the mean of per-net AACs)."""
    return float(area_above_curve(net_curves(uncertainty, wrong, grid), grid, r_max).mean())
