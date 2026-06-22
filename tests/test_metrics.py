"""
Tests for uncertainty/metrics.py — task-agnostic AAC and AUROC.
Run with: pytest tests/test_metrics.py
"""
import pytest
import numpy as np
from uncertainty.metrics import compute_aac, compute_auroc


class TestComputeAAC:
    def test_perfect_oracle_is_zero(self):
        # A curve that immediately reaches 1.0 accuracy should have AAC ~0
        rates = [0.0, 0.01, 1.0]
        accs = [1.0, 1.0, 1.0]
        aac = compute_aac(accs, rates, r_max=0.8)
        assert aac == pytest.approx(0.0, abs=1e-6)

    def test_random_baseline_positive(self):
        # A flat, below-1.0 curve should have positive AAC
        rates = [0.0, 0.5, 1.0]
        accs = [0.8, 0.8, 0.8]
        aac = compute_aac(accs, rates, r_max=0.8)
        assert aac > 0

    def test_better_method_has_lower_aac(self):
        rates = [0.0, 0.4, 0.8, 1.0]
        good_accs = [0.95, 0.98, 1.0, 1.0]
        bad_accs  = [0.80, 0.85, 0.90, 1.0]
        assert compute_aac(good_accs, rates) < compute_aac(bad_accs, rates)

    def test_r_max_clipping(self):
        # Only rates up to r_max=0.5 should affect the result
        rates = [0.0, 0.5, 0.6, 1.0]
        accs_a = [0.9, 0.95, 0.5, 1.0]  # drops after 0.5
        accs_b = [0.9, 0.95, 1.0, 1.0]  # rises after 0.5
        # Both should give the same AAC when r_max=0.5
        assert compute_aac(accs_a, rates, r_max=0.5) == pytest.approx(
            compute_aac(accs_b, rates, r_max=0.5), abs=1e-6
        )

    def test_monotone_curve_decreasing_aac(self):
        # Curve that rises faster should have lower AAC
        rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        fast_rise = [0.85, 0.95, 0.98, 1.0, 1.0, 1.0]
        slow_rise = [0.85, 0.87, 0.90, 0.93, 0.97, 1.0]
        assert compute_aac(fast_rise, rates) < compute_aac(slow_rise, rates)


class TestComputeAUROC:
    def test_perfect_classifier(self):
        y_true = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        assert compute_auroc(y_true, scores) == pytest.approx(1.0)

    def test_random_classifier_near_half(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200).tolist()
        scores = rng.random(200).tolist()
        auroc = compute_auroc(y_true, scores)
        assert 0.3 < auroc < 0.7

    def test_inverted_classifier(self):
        y_true = [1, 1, 0, 0]
        scores = [0.1, 0.2, 0.8, 0.9]  # low uncertainty on errors
        assert compute_auroc(y_true, scores) == pytest.approx(0.0)

    def test_single_positive(self):
        y_true = [1, 0, 0, 0]
        scores = [0.9, 0.5, 0.3, 0.1]
        assert compute_auroc(y_true, scores) == pytest.approx(1.0)
