"""
Tests for tasks/classification/rejection_curves.py — sample-level rejection.
Run with: pytest tests/test_classification_rejection_curves.py
"""
import pytest
from tasks.classification.rejection_curves import (
    calculate_accuracy,
    get_accs_throwaway_rates,
)


def _make_markup(names):
    return {n: {"gt_result": 0, "reagent": "A"} for n in names}


class TestCalculateAccuracy:
    def test_no_mistakes(self):
        assert calculate_accuracy([], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_all_mistakes(self):
        assert calculate_accuracy(["a", "b"], ["a", "b"]) == pytest.approx(0.0)

    def test_half_mistakes(self):
        assert calculate_accuracy(["a"], ["a", "b"]) == pytest.approx(0.5)

    def test_empty_test_set_raises(self):
        with pytest.raises((ZeroDivisionError, AssertionError)):
            calculate_accuracy([], [])


class TestGetAccsThrowawayRates:
    def _build_unc_dict(self, mapping):
        """mapping: {uncertainty_value: [alv_name, ...]}"""
        return {str(k): v for k, v in mapping.items()}

    def test_starts_at_zero_throwaway(self):
        markup = _make_markup(["a", "b", "c"])
        unc = self._build_unc_dict({0.9: ["a"], 0.5: ["b"], 0.1: ["c"]})
        accs, rates = get_accs_throwaway_rates(unc, markup, [])
        assert rates[0] == pytest.approx(0.0)

    def test_ends_at_full_throwaway(self):
        markup = _make_markup(["a", "b"])
        unc = self._build_unc_dict({0.8: ["a"], 0.2: ["b"]})
        accs, rates = get_accs_throwaway_rates(unc, markup, [])
        assert rates[-1] == pytest.approx(1.0)

    def test_accuracy_monotone_nondecreasing(self):
        markup = _make_markup(["a", "b", "c", "d"])
        # "a" and "b" are mistakes; throw away highest-uncertainty first
        unc = self._build_unc_dict({0.9: ["a"], 0.7: ["b"], 0.3: ["c"], 0.1: ["d"]})
        accs, rates = get_accs_throwaway_rates(unc, markup, ["a", "b"])
        for i in range(len(accs) - 1):
            assert accs[i] <= accs[i + 1] + 1e-9

    def test_perfect_uncertainty_reaches_100pct(self):
        # If the most uncertain samples are exactly the wrong ones, accuracy should reach 1.0
        markup = _make_markup(["a", "b", "c"])
        unc = self._build_unc_dict({0.9: ["a"], 0.5: ["b"], 0.1: ["c"]})
        accs, _ = get_accs_throwaway_rates(unc, markup, ["a"])
        assert 1.0 in accs
