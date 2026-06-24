"""
Smoke tests for classification training and inference (BloodyWell / MobileNet).

Uses synthetic data — no real dataset or GPU required, but torch must be installed.
Run with: pytest tests/test_train_classification.py -v
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tasks.classification.models.MobileNet import mobilenet_v3_small
from tasks.classification.train_utils import train
from tasks.classification.inference import run_net, run_net_with_true_lebels


# ── Tiny synthetic dataset ────────────────────────────────────────────────────

IMG_SIZE = 32   # MobileNetV3Small works with any spatial size ≥ 32
N_SAMPLES = 8
BATCH_SIZE = 4


class _SyntheticBloodyWell(Dataset):
    """Mimics the dict-based batch format expected by the classification train loop."""

    def __init__(self, n=N_SAMPLES):
        self.n = n
        torch.manual_seed(0)
        self.images = torch.rand(n, 3, IMG_SIZE, IMG_SIZE)
        self.labels = torch.randint(0, 2, (n,)).float()
        self.names  = [f"alv_{i:04d}.png" for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image":    self.images[idx],
            "agg_type": self.labels[idx],
            "name":     self.names[idx],
            "reagent":  "A",
        }


def _make_loaders():
    ds = _SyntheticBloodyWell()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return loader


def _make_inference_loader():
    """batch_size=1 matches production use: run_net stores data['name'][0] per iteration."""
    ds = _SyntheticBloodyWell()
    return DataLoader(ds, batch_size=1, shuffle=False)


def _make_model_and_optimizer():
    net = mobilenet_v3_small()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    return net, optimizer, scheduler


# ── Training smoke tests ──────────────────────────────────────────────────────

class TestClassificationTraining:
    def test_one_epoch_runs_without_error(self, tmp_path):
        net, optimizer, scheduler = _make_model_and_optimizer()
        loader = _make_loaders()
        device = torch.device("cpu")
        loss_fn = nn.BCELoss()

        returned_net, _ = train(
            device=device,
            net=net,
            trainloader=loader,
            loss_fun=loss_fn,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            epochs=1,
            logdir=tmp_path / "logs",
        )
        assert returned_net is net

    def test_model_weights_change_after_training(self, tmp_path):
        net, optimizer, scheduler = _make_model_and_optimizer()
        loader = _make_loaders()
        device = torch.device("cpu")

        weights_before = [p.clone().detach() for p in net.parameters()]

        train(
            device=device,
            net=net,
            trainloader=loader,
            loss_fun=nn.BCELoss(),
            optimizer=optimizer,
            lr_scheduler=scheduler,
            epochs=1,
            logdir=tmp_path / "logs",
        )

        weights_after = list(net.parameters())
        any_changed = any(
            not torch.equal(b, a.detach())
            for b, a in zip(weights_before, weights_after)
        )
        assert any_changed, "No model weights changed during training"

    def test_loss_is_finite(self, tmp_path):
        """Wrap the training loop manually to capture the loss."""
        net, optimizer, scheduler = _make_model_and_optimizer()
        loader = _make_loaders()
        device = torch.device("cpu")
        loss_fn = nn.BCELoss()

        net.train()
        losses = []
        for data in loader:
            inputs = data["image"].to(device).float()
            labels = data["agg_type"].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs.squeeze().float(), labels.float())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert all(math.isfinite(l) for l in losses), "Training produced non-finite loss"


# ── Inference smoke tests ─────────────────────────────────────────────────────

class TestClassificationInference:
    def test_run_net_returns_dict_of_outputs(self):
        net, _, _ = _make_model_and_optimizer()
        loader = _make_inference_loader()
        device = torch.device("cpu")

        outputs = run_net(device, net, loader)

        assert isinstance(outputs, dict)
        assert len(outputs) == N_SAMPLES
        for name, val in outputs.items():
            assert isinstance(val, (float, int, np.floating)), f"Unexpected type {type(val)}"
            assert 0.0 <= float(val) <= 1.0, f"Output {val} out of [0,1] range"

    def test_run_net_with_true_labels_shapes(self):
        net, _, _ = _make_model_and_optimizer()
        loader = _make_inference_loader()
        device = torch.device("cpu")

        y_pred, y_true = run_net_with_true_lebels(device, net, loader, thr=0.5)

        assert len(y_pred) == N_SAMPLES
        assert len(y_true) == N_SAMPLES
        assert all(p in (0, 1) for p in y_pred), "Predictions should be binary after thresholding"

    def test_inference_output_changes_when_weights_change(self):
        """Zeroing all weights must change the outputs, proving run_net actually uses the model."""
        net, _, _ = _make_model_and_optimizer()
        loader = _make_inference_loader()
        device = torch.device("cpu")

        out_before = run_net(device, net, loader)

        with torch.no_grad():
            for p in net.parameters():
                p.fill_(1.0)

        out_after = run_net(device, net, loader)

        assert out_before != out_after
