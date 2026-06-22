"""
Tests for uncertainty/variance_uncertainty.py — TV decomposition for image tasks (segmentation).
Run with: pytest tests/test_variance_uncertainty.py
"""
import pytest
import torch
from uncertainty.variance_uncertainty import (
    calculate_uncertainty,
    tv_uncertainty,
    multiclass_uncertainty,
)


class TestCalculateUncertainty:
    def test_var_identical_preds_is_zero(self):
        preds = torch.ones(5, 10) * 0.6  # 5 models, 10 pixels, same prediction
        unc = calculate_uncertainty(preds, method="var")
        assert unc.max().item() == pytest.approx(0.0, abs=1e-6)

    def test_var_different_preds_is_positive(self):
        preds = torch.stack([torch.zeros(10), torch.ones(10)])
        unc = calculate_uncertainty(preds, method="var")
        assert unc.min().item() > 0

    def test_bern_var_at_half_is_max(self):
        preds = torch.ones(3, 4) * 0.5
        unc = calculate_uncertainty(preds, method="bern_var")
        assert unc.mean().item() == pytest.approx(0.25, abs=1e-5)

    def test_bern_var_at_zero_or_one_is_zero(self):
        preds_zero = torch.zeros(3, 4)
        preds_one = torch.ones(3, 4)
        assert calculate_uncertainty(preds_zero, "bern_var").max().item() == pytest.approx(0.0, abs=1e-6)
        assert calculate_uncertainty(preds_one, "bern_var").max().item() == pytest.approx(0.0, abs=1e-6)

    def test_none_method_returns_zeros(self):
        preds = torch.rand(3, 4)
        unc = calculate_uncertainty(preds, method=None)
        assert unc.max().item() == pytest.approx(0.0, abs=1e-6)

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError):
            calculate_uncertainty(torch.rand(2, 4), method="unknown")


class TestTVUncertainty:
    def test_additive_epist_aleat(self):
        preds = torch.stack([torch.zeros(4), torch.ones(4)])
        expert_preds = torch.ones(3, 4) * 0.5

        epist_only = calculate_uncertainty(preds, "var")
        aleat_only = calculate_uncertainty(expert_preds, "bern_var")
        total = tv_uncertainty(preds, expert_preds, method="var", expert_method="bern_var")

        assert torch.allclose(total, epist_only + aleat_only)

    def test_no_expert_uses_zero_aleat(self):
        preds = torch.rand(3, 6)
        result = tv_uncertainty(preds, expert_preds=None, method="var", expert_method=None)
        expected = calculate_uncertainty(preds, "var")
        assert torch.allclose(result, expected)

    def test_requires_at_least_one_input(self):
        with pytest.raises(AssertionError):
            tv_uncertainty(None, None, method=None, expert_method=None)


class TestMulticlassUncertainty:
    def test_single_class_matches_tv(self):
        preds = torch.rand(3, 8)
        expert_preds = torch.rand(4, 8)
        mc = multiclass_uncertainty(1, preds, expert_preds, method="var", expert_method="bern_var")
        tv = tv_uncertainty(preds, expert_preds, method="var", expert_method="bern_var")
        assert torch.allclose(mc, tv)

    def test_multiclass_sums_per_class(self):
        n = 3  # classes
        preds = torch.rand(5, n, 10)        # 5 models, 3 classes, 10 pixels
        expert_preds = torch.rand(4, n, 10)
        result = multiclass_uncertainty(n, preds, expert_preds, method="var", expert_method="bern_var")
        assert result.shape == (10,)
