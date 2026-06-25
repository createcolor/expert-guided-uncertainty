"""
Smoke tests for segmentation training and inference (UNet, binary + multiclass).

Uses synthetic data — no real dataset or GPU required, but torch must be installed.
The UNet requires spatial dimensions divisible by 16 (4 max-pool layers).
Run with: pytest tests/test_train_segmentation.py -v
"""
import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tasks.segmentation.models.UNet import UNet
from tasks.segmentation.seg_utils.train import train
from tasks.segmentation.seg_utils.metrics.dice import DiceScore, stacked_Dice


# ── Tiny synthetic dataloaders ────────────────────────────────────────────────

IMG_H, IMG_W = 64, 64   # must be divisible by 16
N_SAMPLES   = 4
BATCH_SIZE  = 2


def _make_binary_loaders():
    torch.manual_seed(0)
    X = torch.rand(N_SAMPLES, 1, IMG_H, IMG_W)
    Y = torch.randint(0, 2, (N_SAMPLES, 1, IMG_H, IMG_W)).float()
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return loader


def _make_multiclass_loaders(n_classes=3):
    torch.manual_seed(1)
    X = torch.rand(N_SAMPLES, 1, IMG_H, IMG_W)
    # Long class indices for CrossEntropyLoss
    Y = torch.randint(0, n_classes, (N_SAMPLES, IMG_H, IMG_W))
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return loader


class _DummyMetric(nn.Module):
    """Always returns 0.5 — lets train() log a value without real computation."""
    def forward(self, pred, target):
        return torch.tensor(0.5)


# ── UNet forward-pass sanity check ───────────────────────────────────────────

class TestUNetForward:
    def test_binary_output_shape(self):
        model = UNet(n_channels=1, n_classes=1)
        x = torch.rand(2, 1, IMG_H, IMG_W)
        out = model(x)
        assert out.shape == (2, 1, IMG_H, IMG_W)

    def test_multiclass_output_shape(self):
        n_classes = 3
        model = UNet(n_channels=1, n_classes=n_classes)
        x = torch.rand(2, 1, IMG_H, IMG_W)
        out = model(x)
        assert out.shape == (2, n_classes, IMG_H, IMG_W)

    def test_output_spatial_size_preserved(self):
        model = UNet(n_channels=3, n_classes=1)
        x = torch.rand(1, 3, IMG_H, IMG_W)
        out = model(x)
        assert out.shape[-2:] == (IMG_H, IMG_W), "UNet must preserve spatial dimensions"


# ── Training smoke tests ──────────────────────────────────────────────────────

class TestSegmentationTraining:
    def _setup(self):
        model     = UNet(n_channels=1, n_classes=1)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
        loss_fn   = nn.BCEWithLogitsLoss()
        metric    = _DummyMetric()
        return model, optimizer, scheduler, loss_fn, metric

    def test_one_epoch_runs_without_error(self, tmp_path):
        model, optimizer, scheduler, loss_fn, metric = self._setup()
        loader = _make_binary_loaders()

        train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metric=metric,
            epochs=1,
            data_tr=loader,
            data_val=loader,
            device=torch.device("cpu"),
        )

    def test_model_weights_change_after_one_epoch(self, tmp_path):
        model, optimizer, scheduler, loss_fn, metric = self._setup()
        loader = _make_binary_loaders()

        weights_before = [p.clone().detach() for p in model.parameters()]

        train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metric=metric,
            epochs=1,
            data_tr=loader,
            data_val=loader,
            device=torch.device("cpu"),
        )

        weights_after = list(model.parameters())
        any_changed = any(
            not torch.equal(b, a.detach())
            for b, a in zip(weights_before, weights_after)
        )
        assert any_changed, "No model weights changed after one epoch"

    def test_model_saved_when_save_path_provided(self, tmp_path):
        model, optimizer, scheduler, loss_fn, metric = self._setup()
        loader = _make_binary_loaders()
        save_path = tmp_path / "unet.pt"

        train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metric=metric,
            epochs=1,
            data_tr=loader,
            data_val=loader,
            device=torch.device("cpu"),
            save_path=save_path,
        )

        assert save_path.exists(), "Model checkpoint not saved"
        loaded = torch.load(save_path, map_location="cpu", weights_only=True)
        assert isinstance(loaded, dict), "Saved file should be a state_dict"

    def test_loss_is_finite(self):
        model = UNet(n_channels=1, n_classes=1)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn   = nn.BCEWithLogitsLoss()
        loader    = _make_binary_loaders()

        model.train()
        losses = []
        for X_batch, Y_batch in loader:
            optimizer.zero_grad()
            out  = model(X_batch)
            loss = loss_fn(out, Y_batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert all(math.isfinite(l) for l in losses), "Training produced non-finite loss"


# ── Inference smoke tests ─────────────────────────────────────────────────────

class TestSegmentationInference:
    def test_eval_mode_no_grad(self):
        model = UNet(n_channels=1, n_classes=1)
        x = torch.rand(2, 1, IMG_H, IMG_W)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, IMG_H, IMG_W)

    def test_dice_score_perfect_prediction(self):
        """Dice of a mask predicted perfectly should be 1."""
        mask = torch.randint(0, 2, (2, 1, IMG_H, IMG_W)).float()
        logits = mask * 10 - (1 - mask) * 10  # push predictions to 0/1
        dice = DiceScore()
        score = dice(logits, mask)
        assert score.item() == pytest.approx(1.0, abs=0.01)

    def test_stacked_dice_shape(self):
        preds = torch.randint(0, 2, (N_SAMPLES, IMG_H * IMG_W)).bool()
        labels = torch.randint(0, 2, (N_SAMPLES, IMG_H * IMG_W)).bool()
        result = stacked_Dice(preds, labels, binarize=False)
        assert result.shape == (N_SAMPLES,)

    def test_checkpoint_roundtrip(self, tmp_path):
        """Save and reload a UNet state dict; outputs must be identical."""
        model = UNet(n_channels=1, n_classes=1)
        model.eval()
        x = torch.rand(1, 1, IMG_H, IMG_W)

        with torch.no_grad():
            out_before = model(x)

        path = tmp_path / "unet.pt"
        torch.save(model.state_dict(), str(path))

        model2 = UNet(n_channels=1, n_classes=1)
        model2.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
        model2.eval()

        with torch.no_grad():
            out_after = model2(x)

        assert torch.allclose(out_before, out_after)
