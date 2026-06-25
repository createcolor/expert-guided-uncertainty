"""Tests for the shared common/ layer: backends, estimators, rejection driver.

These exercise the task-agnostic core that all three tasks build on. They cover
both backends (numpy and torch) and the cross-backend consistency that the
"two backends, one interface" design promises.
"""

import numpy as np
import pytest

from common.backends import get_backend, NumpyBackend, TorchBackend
from common.estimators import (
    reduce_members, total_variance, multiclass_total_variance, entropy,
)
from common.rejection import keep_counts_for, rejection_curve

torch = pytest.importorskip("torch")


# ── Backend dispatch ──────────────────────────────────────────────────────────

class TestBackendDispatch:
    def test_numpy_array_gets_numpy_backend(self):
        assert isinstance(get_backend(np.zeros(3)), NumpyBackend)

    def test_torch_tensor_gets_torch_backend(self):
        assert isinstance(get_backend(torch.zeros(3)), TorchBackend)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            get_backend([1, 2, 3])


# ── Estimators ────────────────────────────────────────────────────────────────

class TestEstimators:
    def test_var_matches_population_variance(self):
        x = np.random.rand(5, 4)  # [members, units]
        got = reduce_members(x, "var", member_axis=0)
        np.testing.assert_allclose(got, np.var(x, axis=0), rtol=1e-6)

    def test_bern_var_matches_formula(self):
        x = np.random.rand(5, 4)
        got = reduce_members(x, "bern_var", member_axis=0)
        np.testing.assert_allclose(got, np.mean(x * (1 - x), axis=0), rtol=1e-6)

    def test_none_method_is_zero_map(self):
        x = np.random.rand(5, 4)
        got = reduce_members(x, None, member_axis=0)
        assert got.shape == (4,)
        np.testing.assert_array_equal(got, np.zeros(4))

    def test_total_variance_epistemic_only_equals_variance(self):
        preds = np.random.rand(5, 4)
        tv = total_variance(preds, None, "var", None)
        np.testing.assert_allclose(tv, np.var(preds, axis=0), rtol=1e-6)

    def test_total_variance_combines_two_sources(self):
        epist = np.random.rand(5, 4)
        aleat = np.random.rand(5, 4)
        tv = total_variance(epist, aleat, "var", "bern_var")
        expected = np.var(epist, axis=0) + np.mean(aleat * (1 - aleat), axis=0)
        np.testing.assert_allclose(tv, expected, rtol=1e-6)

    def test_total_variance_requires_some_input(self):
        with pytest.raises(AssertionError):
            total_variance(None, None)

    def test_multiclass_sums_over_classes(self):
        # [members, units, classes] with class_axis=-1
        preds = np.random.rand(5, 4, 3)
        mc = multiclass_total_variance(3, preds, None, "var", None,
                                       member_axis=0, class_axis=-1)
        per_class = np.var(preds, axis=0)              # [units, classes]
        np.testing.assert_allclose(mc, per_class.sum(axis=-1), rtol=1e-6)

    def test_entropy_uniform_is_log_k(self):
        p = np.full((4, 3), 1 / 3)
        np.testing.assert_allclose(entropy(p, class_axis=-1), np.log(3), rtol=1e-6)

    def test_numpy_torch_consistency(self):
        x = np.random.rand(5, 4, 3)
        np_res = multiclass_total_variance(3, x, None, "var", None,
                                           member_axis=0, class_axis=-1)
        torch_res = multiclass_total_variance(3, torch.from_numpy(x), None, "var", None,
                                              member_axis=0, class_axis=-1)
        np.testing.assert_allclose(np_res, torch_res.numpy(), rtol=1e-5)


# ── Rejection driver ──────────────────────────────────────────────────────────

def _accuracy_member_curve(correctness):
    """member_curve computing accuracy over the retained (least-uncertain) prefix."""
    def member_curve(member_idx, order, keep_counts):
        c_sorted = correctness[member_idx][np.asarray(order)]
        scores = []
        for keep in keep_counts:
            scores.append(float(np.mean(c_sorted[:keep])) if keep > 0 else 1.0)
        return scores
    return member_curve


class TestRejectionDriver:
    def test_keep_counts_for(self):
        assert keep_counts_for(4, [0.0, 0.25, 0.5, 1.0]) == [4, 3, 2, 0]

    def test_known_accuracy_curve(self):
        # 1 member, 4 units. Most uncertain unit (0.4) is the only wrong one.
        uncertainty = np.array([[0.4, 0.1, 0.3, 0.2]])
        correctness = np.array([[0, 1, 1, 1]])  # 1 = correct
        rates, mean_scores, _ = rejection_curve(
            uncertainty, _accuracy_member_curve(correctness), [0.25, 0.5],
            member_axis=0, unit_axis=-1,
        )
        # rates -> [0, 0.25, 0.5]; keep -> [4, 3, 2]; ascending-uncert order keeps the wrong one last
        assert rates == [0.0, 0.25, 0.5]
        np.testing.assert_allclose(mean_scores, [0.75, 1.0, 1.0])

    def test_perfect_ranking_reaches_one(self):
        # uncertainty perfectly ranks the single error highest -> accuracy hits 1 after one drop
        uncertainty = np.array([[0.9, 0.1, 0.2]])
        correctness = np.array([[0, 1, 1]])
        _, mean_scores, _ = rejection_curve(
            uncertainty, _accuracy_member_curve(correctness), [1 / 3],
            member_axis=0, unit_axis=-1,
        )
        assert mean_scores[0] == pytest.approx(2 / 3)  # keep all
        assert mean_scores[1] == pytest.approx(1.0)    # drop the error

    def test_averages_across_members(self):
        uncertainty = np.array([[0.2, 0.1], [0.1, 0.2]])
        correctness = np.array([[1, 0], [0, 1]])
        _, mean_scores, std_scores = rejection_curve(
            uncertainty, _accuracy_member_curve(correctness), [0.5],
            member_axis=0, unit_axis=-1,
        )
        # Both members have one correct, one wrong -> accuracy 0.5 keeping all, identical curves
        assert mean_scores[0] == pytest.approx(0.5)
        assert std_scores[0] == pytest.approx(0.0)
