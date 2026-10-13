"""Paired bootstrap, Wilcoxon signed-rank and Holm correction.

All methods in a dataset are scored on the *same* bootstrap resample of test
items, so the bootstrap distribution of ``AAC(ours) - AAC(other)`` keeps the
pairing between methods. Only items are resampled: rejection-curve points and
ensemble members are not independent observations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Mapping

import numpy as np
from scipy import stats as sp_stats

StatFn = Callable[[np.ndarray], Mapping[str, float]]


def bootstrap_method_stats(stat_fn: StatFn, n_units: int, n_boot: int = 10_000,
                           seed: int = 0) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate ``stat_fn`` on the full test set and on ``n_boot`` item resamples.

    ``stat_fn(idx)`` receives an index array into the test items (with
    repetitions for a resample) and returns ``{method: AAC}`` for every method,
    so all methods share each resample.

    Returns ``(observed, boot)`` where ``boot[method]`` has shape ``[n_boot]``.
    """
    rng = np.random.default_rng(seed)
    observed = dict(stat_fn(np.arange(n_units)))
    boot = {method: np.empty(n_boot) for method in observed}
    for b in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        for method, value in stat_fn(idx).items():
            boot[method][b] = value
    return observed, boot


def paired_difference(observed: Mapping[str, float], boot: Mapping[str, np.ndarray],
                      ours: str, other: str, ci: float = 0.95) -> tuple[float, float, float, float]:
    """Delta = stat(ours) - stat(other) with a percentile CI and a two-sided p-value.

    The p-value tests ``H0: Delta = 0`` by centring the bootstrap distribution
    on zero: ``p = (1 + #{|d* - d| >= |d|}) / (B + 1)``.
    """
    delta = observed[ours] - observed[other]
    d_star = np.asarray(boot[ours]) - np.asarray(boot[other])
    tail = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(d_star, [tail, 1.0 - tail])
    extreme = np.sum(np.abs(d_star - delta) >= abs(delta))
    p_value = (1.0 + extreme) / (len(d_star) + 1.0)
    return float(delta), float(ci_low), float(ci_high), float(min(p_value, 1.0))


def wilcoxon_p(differences: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired per-unit differences."""
    differences = np.asarray(differences, dtype=float)
    if np.allclose(differences, 0.0):
        return 1.0
    return float(sp_stats.wilcoxon(differences, zero_method="wilcox").pvalue)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p[i])
        adjusted[i] = min(running_max, 1.0)
    return adjusted.tolist()


@dataclass
class PairedResult:
    """One row of the results table."""
    dataset: str
    ours: str
    other: str
    n_units: int
    aac_ours: float
    aac_other: float
    delta: float
    ci_low: float
    ci_high: float
    p_bootstrap: float
    p_wilcoxon: float | None = None
    p_holm: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)
