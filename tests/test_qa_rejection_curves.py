"""
Tests for tasks/qa rejection curve logic — sample-level rejection for QA (uses nlp-2 format).
The QA rejection curves use a different data format than classification:
  uncertainties_dict: {uncertainty_str: [[is_wrong_net0, is_wrong_net1, ...], ...]}
Run with: pytest tests/test_qa_rejection_curves.py
"""
import pytest
import numpy as np
from tasks.qa.rejection_curves import get_accs_throwaway_rates, get_rc_info_averaging_mode


class TestGetAccsThrowawayRatesQA:
    def _make_unc(self, mapping):
        """mapping: {float_unc: [(is_wrong_net0, is_wrong_net1), ...]}
        Keys use 7 decimal places to match the sign_figs_num=7 default in get_accs_throwaway_rates."""
        return {f"{k:.7f}": v for k, v in mapping.items()}

    def test_starts_at_zero(self):
        unc = self._make_unc({0.9: [[True, False]], 0.1: [[False, True]]})
        accs, rates = get_accs_throwaway_rates(unc, net_id=0)
        assert rates[0] == pytest.approx(0.0)

    def test_ends_at_one(self):
        unc = self._make_unc({0.9: [[True]], 0.1: [[False]]})
        accs, rates = get_accs_throwaway_rates(unc, net_id=0)
        assert rates[-1] == pytest.approx(1.0)

    def test_accuracy_nondecreasing_when_worst_rejected_first(self):
        # net_id=0: sample 0 is wrong (high unc), sample 1 correct (low unc)
        unc = self._make_unc({0.9: [[True, False]], 0.2: [[False, True]]})
        accs, rates = get_accs_throwaway_rates(unc, net_id=0)
        for i in range(len(accs) - 1):
            assert accs[i] <= accs[i + 1] + 1e-9


class TestGetRCInfoAveragingModeQA:
    def test_returns_dict_with_x_keys(self):
        unc = {"0.900": [[True, False], [False, True]], "0.100": [[False, True], [True, False]]}
        result = get_rc_info_averaging_mode(unc, n_bins=5, net_ids=range(2))
        assert len(result) == 6  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

    def test_values_are_mean_std_tuples(self):
        unc = {"0.900": [[True]], "0.100": [[False]]}
        result = get_rc_info_averaging_mode(unc, n_bins=4, net_ids=range(1))
        for key, val in result.items():
            mean, std = val
            assert 0.0 <= mean <= 1.0
            assert std >= 0.0
