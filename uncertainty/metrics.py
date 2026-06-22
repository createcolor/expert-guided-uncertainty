"""
Task-agnostic metrics: AAC (Area Above Curve) and AUROC.
Each task's rejection_curves.py calls these after computing its own (accuracy, throwaway_rate) pairs.
"""
import numpy as np
from sklearn import metrics as sk_metrics


def compute_aac(accuracies: list, throwaway_rates: list, r_max: float = 0.8) -> float:
    """
    Area Above Curve: integral of (oracle_curve - method_curve) over [0, r_max].
    Lower is better. AAC of the oracle curve itself is 0.

    Args:
        accuracies: accuracy values of the method's rejection curve.
        throwaway_rates: corresponding throwaway rates (x-axis), must start at 0.
        r_max: upper integration limit (paper uses 0.8).

    Returns:
        AAC value (non-negative float).
    """
    rates = np.array(throwaway_rates)
    accs = np.array(accuracies)

    mask = rates <= r_max
    rates_clipped = np.append(rates[mask], r_max)
    accs_clipped = np.append(accs[mask], np.interp(r_max, rates, accs))

    oracle_auc = np.trapezoid(np.ones_like(rates_clipped), rates_clipped)
    method_auc = np.trapezoid(accs_clipped, rates_clipped)
    return float(oracle_auc - method_auc)


def compute_auroc(y_true: list, uncertainty_scores: list) -> float:
    """
    AUROC for error detection: treats uncertainty as a score for predicting
    whether the model made a mistake (higher uncertainty → higher chance of error).

    Args:
        y_true: binary labels where 1 = model was wrong, 0 = model was correct.
        uncertainty_scores: uncertainty value per sample.

    Returns:
        AUROC score (higher is better, 0.5 = random).
    """
    return float(sk_metrics.roc_auc_score(y_true, uncertainty_scores))
