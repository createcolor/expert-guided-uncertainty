"""Tests for the paired-test tooling in stats/.

The curve engines are checked against verbatim copies of the old functions they
replace (``uncertainty_estimation/rejection_curves.py`` and the Dice functions in
``segmentation/utils/metrics/dice.py``), so the new statistics are computed on
exactly the curves the paper reports.
"""

import numpy as np
import pytest

from stats.paired_stats import bootstrap_method_stats, paired_difference, holm, wilcoxon_p
from stats.sample_curves import make_grid, net_curves, area_above_curve, method_aac, oracle_curve
from stats.segmentation_curves import (
    old_throwaway_rates, old_keep_counts, per_image_dice_curves, area_under_curves,
)


# ── Verbatim copies of the old item-level rejection-curve code ────────────────

def _old_calc_accuracy4bin(x, throwaway_rates, accuracies):
    if x in throwaway_rates:
        accuracy = accuracies[throwaway_rates.index(x)]
    else:
        throwaway_rates_less_x = list(filter(lambda rate: rate < x, throwaway_rates))
        left_closest_rate = min(throwaway_rates_less_x, key=lambda rate: x - rate)
        right_closest_rate = throwaway_rates[throwaway_rates.index(left_closest_rate) + 1]
        alpha = (x - left_closest_rate) / (right_closest_rate - left_closest_rate)
        accuracy1 = accuracies[throwaway_rates.index(left_closest_rate)]
        accuracy2 = accuracies[throwaway_rates.index(right_closest_rate)]
        accuracy = accuracy1 + alpha * (accuracy2 - accuracy1)
    return accuracy


def _old_get_accs_throwaway_rates(uncertainties_dict, net_id, sign_figs_num=7):
    total_samples_num = 0
    mistakes_num_left = 0
    uncertainties = {}
    for unc, samples_info in uncertainties_dict.items():
        mistakes_num = 0
        correct_num = 0
        for sample_info in samples_info:
            if sample_info[net_id]:
                mistakes_num += 1
            else:
                correct_num += 1
        uncertainties[unc] = (mistakes_num, correct_num)
        total_samples_num += (mistakes_num + correct_num)
        mistakes_num_left += mistakes_num
    initial_acc = 1.0 - mistakes_num_left / total_samples_num
    accuracies, throwaway_rates = [initial_acc], [0.0]
    sorted_uncertainties = sorted(list(map(float, uncertainties.keys())), reverse=True)
    samples_num_left = total_samples_num
    for uncertainty in sorted_uncertainties:
        uncertainty = f'{uncertainty:.{sign_figs_num}f}'
        mistakes_num_left -= uncertainties[uncertainty][0]
        samples_num_left -= (uncertainties[uncertainty][0] + uncertainties[uncertainty][1])
        if samples_num_left > 0:
            throwaway_rate = 1. - (samples_num_left / total_samples_num)
            throwaway_rates.append(throwaway_rate)
            accuracy = 1.0 - mistakes_num_left / samples_num_left
            accuracies.append(accuracy)
            if accuracy == 1.0:
                break
    accuracies.append(1.0)
    throwaway_rates.append(1.0)
    return accuracies, throwaway_rates


def _old_get_rc_info_averaging_mode(uncertainties, n_bins, net_ids, sign_figs_num):
    x_list = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ensemble_accuracies_dict = {x: [] for x in x_list}
    for net_id in net_ids:
        if type(uncertainties) == list:
            accuracies, throwaway_rates = _old_get_accs_throwaway_rates(
                uncertainties[net_id], net_id, sign_figs_num)
        else:
            accuracies, throwaway_rates = _old_get_accs_throwaway_rates(
                uncertainties, net_id, sign_figs_num)
        for x in x_list:
            ensemble_accuracies_dict[x].append(
                _old_calc_accuracy4bin(x, throwaway_rates, accuracies))
    return np.array([np.mean(ensemble_accuracies_dict[x]) for x in x_list])


def _to_old_format(unc_values, wrong, sign_figs):
    """{unc_str: [[is_wrong_net0, ...], ...]} as the old calculate_uncertainty saves it."""
    out = {}
    for i, u in enumerate(unc_values):
        out.setdefault(f"{u:.{sign_figs}f}", []).append([bool(w) for w in wrong[:, i]])
    return out


# ── Item-level curves ─────────────────────────────────────────────────────────

class TestItemLevelCurves:
    SIGN_FIGS = 3          # coarse rounding -> many tied uncertainties, the hard case
    N_BINS = 50

    def _data(self, seed, n_items=80, n_nets=4):
        rng = np.random.default_rng(seed)
        unc = rng.random(n_items)
        unc_str = [f"{u:.{self.SIGN_FIGS}f}" for u in np.round(unc, 2)]   # ties
        wrong = rng.random((n_nets, n_items)) < 0.3
        return np.array([float(s) for s in unc_str]), wrong

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_shared_ranking_matches_old_code(self, seed):
        unc, wrong = self._data(seed)
        grid = make_grid(self.N_BINS)
        old = _old_get_rc_info_averaging_mode(
            _to_old_format(unc, wrong, self.SIGN_FIGS), self.N_BINS,
            range(wrong.shape[0]), self.SIGN_FIGS)
        new = net_curves(unc, wrong, grid).mean(axis=0)
        np.testing.assert_allclose(new, old, atol=1e-12)

    @pytest.mark.parametrize("seed", [3, 4])
    def test_per_net_ranking_matches_old_code(self, seed):
        _, wrong = self._data(seed)
        rng = np.random.default_rng(seed + 100)
        per_net = np.round(rng.random(wrong.shape), 2)
        grid = make_grid(self.N_BINS)
        old_dicts = [_to_old_format(per_net[k], wrong, self.SIGN_FIGS)
                     for k in range(wrong.shape[0])]
        old = _old_get_rc_info_averaging_mode(
            old_dicts, self.N_BINS, range(wrong.shape[0]), self.SIGN_FIGS)
        new = net_curves(per_net, wrong, grid).mean(axis=0)
        np.testing.assert_allclose(new, old, atol=1e-12)

    def test_aac_is_area_between_oracle_and_mean_curve(self):
        unc, wrong = self._data(5)
        grid = make_grid(self.N_BINS)
        curves = net_curves(unc, wrong, grid)
        oracle = oracle_curve(wrong.mean(), grid)
        assert method_aac(unc, wrong, grid) == pytest.approx(
            area_above_curve(curves.mean(axis=0), grid)[0] - area_above_curve(oracle, grid)[0])

    def test_oracle_ranking_has_zero_aac(self):
        wrong = np.array([[True] * 10 + [False] * 90])
        assert method_aac(wrong[0].astype(float), wrong, make_grid(100)) == pytest.approx(0, abs=1e-15)

    def test_perfect_ranking_beats_reversed_ranking(self):
        wrong = np.array([[True] * 10 + [False] * 90])
        perfect = np.r_[np.ones(10), np.zeros(90)] + np.linspace(0, 0.01, 100)
        grid = make_grid(100)
        assert method_aac(perfect, wrong, grid) < method_aac(-perfect, wrong, grid)


# ── Segmentation curves vs the old Dice functions ─────────────────────────────

torch = pytest.importorskip("torch")


def _old_stacked_Dice(preds, labels, binarize=True, smooth=1.):
    if binarize:
        preds = preds > 0.5
    intersection = (preds * labels).sum(dim=1)
    return (2. * intersection + smooth) / (preds.sum(dim=1) + labels.sum(dim=1) + smooth)


def _old_stacked_micro_Dice(preds, labels, binarize=True, smooth=1.):
    if binarize:
        preds = preds > 0.5
    labels[labels == 0] = -1
    intersection = (preds == labels).sum(dim=1)
    return (2. * intersection + smooth) / ((preds > 0).sum(dim=1) + (labels > 0).sum(dim=1) + smooth)


def _old_per_image_loop(pred, lab, order, rates, dice_fn):
    """The old loop, keeping per-image scores instead of their mean."""
    ps = torch.from_numpy(np.take_along_axis(pred, order, axis=1))
    ls = torch.from_numpy(np.take_along_axis(lab, order, axis=1))
    out = [dice_fn(ps, ls.clone(), binarize=False).double().numpy()]
    for rate in rates[1:]:
        throwaway_num = round(rate * ps.shape[-1])
        out.append(dice_fn(ps[:, :-throwaway_num], ls[:, :-throwaway_num].clone(),
                           binarize=False).double().numpy())
    return np.stack(out, axis=1)


class TestSegmentationCurves:
    def _data(self, seed, n_classes, n_images=5, n_pixels=200):
        rng = np.random.default_rng(seed)
        pred = rng.integers(0, n_classes, (n_images, n_pixels))
        lab = rng.integers(0, n_classes, (n_images, n_pixels))
        unc = np.round(rng.random((n_images, n_pixels)), 2)            # ties, as in real maps
        return pred, lab, np.argsort(unc, axis=1)

    # 0.001 * 200 rounds to 0 -> exercises the old empty-slice quirk
    RATES = np.array([0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.8, 0.999, 1.0])

    def test_multiclass_matches_stacked_micro_dice(self):
        pred, lab, order = self._data(0, n_classes=3)
        keep = old_keep_counts(self.RATES, pred.shape[1])
        new = per_image_dice_curves(pred, lab, order, keep)
        old = _old_per_image_loop(pred, lab, order, self.RATES, _old_stacked_micro_Dice)
        np.testing.assert_allclose(new, old, rtol=1e-6)

    def test_binary_matches_stacked_dice(self):
        pred, lab, order = self._data(1, n_classes=2)
        keep = old_keep_counts(self.RATES, pred.shape[1])
        new = per_image_dice_curves(pred, lab, order, keep)
        old = _old_per_image_loop(pred.astype(bool), lab, order, self.RATES, _old_stacked_Dice)
        np.testing.assert_allclose(new, old, rtol=1e-6)

    def test_keep_counts_reproduce_empty_slice(self):
        keep = old_keep_counts(np.array([0.0, 0.001, 0.5]), 200)
        assert keep.tolist() == [200, 0, 100]       # round(0.2) == 0 -> old slice [:-0] is empty

    def test_area_uses_old_interval(self):
        rates = old_throwaway_rates()
        ones = np.ones((2, rates.size))
        np.testing.assert_allclose(area_under_curves(ones, rates, 0.8), 0.8, atol=1e-9)


# ── Statistics ────────────────────────────────────────────────────────────────

class TestStatistics:
    def test_identical_methods_give_zero_delta_and_p_one(self):
        values = np.random.default_rng(0).random(50)
        stat = lambda idx: {"a": values[idx].mean(), "b": values[idx].mean()}
        observed, boot = bootstrap_method_stats(stat, 50, n_boot=200, seed=1)
        delta, low, high, p = paired_difference(observed, boot, "a", "b")
        assert delta == 0.0 and low == 0.0 and high == 0.0 and p == 1.0

    def test_consistent_paired_shift_is_significant(self):
        rng = np.random.default_rng(0)
        base = rng.random(100)
        better = base - 0.05 + rng.normal(0, 0.01, 100)   # small but consistent paired gain
        stat = lambda idx: {"ours": better[idx].mean(), "other": base[idx].mean()}
        observed, boot = bootstrap_method_stats(stat, 100, n_boot=2000, seed=1)
        delta, low, high, p = paired_difference(observed, boot, "ours", "other")
        assert delta < 0 and high < 0
        assert p < 0.001
        assert wilcoxon_p(better - base) < 0.001

    def test_no_true_difference_is_not_significant(self):
        rng = np.random.default_rng(3)
        a, b = rng.random(200), rng.random(200)
        stat = lambda idx: {"a": a[idx].mean(), "b": b[idx].mean()}
        observed, boot = bootstrap_method_stats(stat, 200, n_boot=2000, seed=1)
        *_, p = paired_difference(observed, boot, "a", "b")
        assert p > 0.05

    def test_holm(self):
        assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
        assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
