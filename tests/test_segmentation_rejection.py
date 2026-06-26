"""Integration tests for the unified segmentation rejection pipeline.

Covers the membership reduction, the per-pixel uncertainty step, and the curve
construction on tiny synthetic tensors — no real data or GPU required.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common.rejection import sweep_members
from tasks.segmentation.uncertainty import compute_uncertainty
from tasks.segmentation.rejection_curves import to_membership, dice_member_curve


# ── Membership reduction ──────────────────────────────────────────────────────

class TestToMembership:
    def test_multiclass_is_one_hot(self):
        preds = torch.randn(4, 2, 3, 16)        # [members, samples, classes, pixels]
        labels = torch.randint(0, 3, (2, 16))
        pred_member, label_member, eval_classes = to_membership(preds, labels)

        assert pred_member.shape == (4, 2, 3, 16)
        assert label_member.shape == (2, 3, 16)
        # exactly one class active per pixel
        assert torch.all(pred_member.sum(dim=2) == 1)
        assert torch.all(label_member.sum(dim=1) == 1)
        assert eval_classes == [1, 2]           # background (0) skipped

    def test_binary_single_channel(self):
        preds = torch.randn(4, 2, 1, 16)
        labels = torch.randint(0, 2, (2, 16))
        pred_member, label_member, eval_classes = to_membership(preds, labels)

        assert eval_classes == [0]
        assert set(torch.unique(pred_member).tolist()).issubset({0.0, 1.0})
        # matches sigmoid-threshold semantics used by DiceScore at inference
        expected = (torch.sigmoid(preds[:, :, 0]) >= 0.5).float()
        assert torch.equal(pred_member[:, :, 0], expected)

    def test_missing_class_axis_treated_as_binary(self):
        preds = torch.randn(4, 2, 16)           # no class axis
        labels = torch.randint(0, 2, (2, 16))
        pred_member, _, eval_classes = to_membership(preds, labels)
        assert pred_member.shape == (4, 2, 1, 16)
        assert eval_classes == [0]


# ── End-to-end curve ──────────────────────────────────────────────────────────

class TestRejectionPipeline:
    def test_multiclass_uncertainty_shape(self, tmp_path):
        M, S, C, H, W = 5, 2, 3, 4, 4
        raw = torch.randn(M, S, C, H, W)              # multiclass: 5D, class axis present
        preds_path = tmp_path / "predictions.pt"
        torch.save(raw, preds_path)
        uncertainty = compute_uncertainty(
            {"models_predictions": str(preds_path), "models_ids": M, "models_method": "var"},
            torch.device("cpu"),
        )
        assert uncertainty.shape == (M, S, H * W)

    def test_binary_uncertainty_shape(self, tmp_path):
        # Binary predictions have NO class axis: [members, samples, H, W] (4D).
        # Regression guard: this must still yield [members, samples, pixels].
        M, S, H, W = 5, 3, 4, 4
        raw = torch.randn(M, S, H, W)
        preds_path = tmp_path / "predictions.pt"
        torch.save(raw, preds_path)
        uncertainty = compute_uncertainty(
            {"models_predictions": str(preds_path), "models_ids": M, "models_method": "var"},
            torch.device("cpu"),
        )
        assert uncertainty.shape == (M, S, H * W)

    def test_binary_end_to_end_curve(self, tmp_path):
        # Full binary pipeline: predictions -> uncertainty -> rejection curve.
        M, S, H, W = 4, 2, 4, 4
        P = H * W
        raw = torch.randn(M, S, H, W)
        labels = torch.randint(0, 2, (S, P))
        preds_path = tmp_path / "predictions.pt"
        torch.save(raw, preds_path)

        uncertainty = compute_uncertainty(
            {"models_predictions": str(preds_path), "models_ids": M, "models_method": "var"},
            torch.device("cpu"),
        )
        raw_flat = torch.reshape(raw, raw.shape[:-2] + (-1,))
        pred_member, label_member, eval_classes = to_membership(raw_flat, labels)
        assert eval_classes == [0]
        order = torch.argsort(uncertainty, dim=-1)
        mc = dice_member_curve(pred_member, label_member, 0)
        rates = [0.0, 0.25, 0.5]
        mean_dice, _ = sweep_members(order, P, mc, rates, member_axis=0)
        assert mean_dice.shape == (len(rates),)
        assert np.all((mean_dice >= 0.0) & (mean_dice <= 1.0))

    def test_curve_shape_and_range(self, tmp_path):
        M, S, C, H, W = 5, 2, 3, 4, 4
        raw = torch.randn(M, S, C, H, W)
        labels = torch.randint(0, C, (S, H * W))

        preds_path = tmp_path / "predictions.pt"
        torch.save(raw, preds_path)
        uncertainty = compute_uncertainty(
            {"models_predictions": str(preds_path), "models_ids": M, "models_method": "var"},
            torch.device("cpu"),
        )
        assert uncertainty.shape == (M, S, H * W)

        raw_flat = torch.reshape(raw, raw.shape[:-2] + (-1,))
        pred_member, label_member, eval_classes = to_membership(raw_flat, labels)
        order = torch.argsort(uncertainty, dim=-1)
        rates = [0.0, 0.25, 0.5, 0.75]

        for cls in eval_classes:
            mc = dice_member_curve(pred_member, label_member, cls)
            mean_dice, std_dice = sweep_members(order, H * W, mc, rates, member_axis=0)
            assert mean_dice.shape == (len(rates),)
            assert np.all((mean_dice >= 0.0) & (mean_dice <= 1.0))
            assert np.all(std_dice >= 0.0)

    def test_curve_improves_when_uncertainty_tracks_errors(self):
        # One sample, one member, binary. Wrong pixels are made most uncertain;
        # discarding them must not decrease Dice.
        P = 20
        preds_logits = torch.full((1, 1, 1, P), 5.0)   # all predict foreground
        labels = torch.ones(1, P)
        labels[0, :5] = 0                               # first 5 are background -> 5 errors

        pred_member, label_member, _ = to_membership(preds_logits, labels)
        # uncertainty highest exactly on the 5 wrong pixels
        uncertainty = torch.zeros(1, 1, P)
        uncertainty[0, 0, :5] = 1.0

        order = torch.argsort(uncertainty, dim=-1)
        mc = dice_member_curve(pred_member, label_member, 0)
        mean_dice, _ = sweep_members(order, P, mc, [0.25], member_axis=0)
        assert mean_dice[-1] >= mean_dice[0]            # dropping the errors helps
