"""Collect per-image Dice-rejection areas from the old segmentation pipeline.

The paper's segmentation curves came from the old ``segmentation/`` scripts:
``rejection_curves_microavg.py`` for RIGA (multiclass, foreground classes
pooled) and ``rejection_curves.py`` for LIDC-IDRI (binary). Both compute Dice per
image and then average over images, which hides the per-image values a paired
test needs. This script reproduces those runs but keeps, for every method, net
and image, the area under that image's Dice curve on ``[0, r_max]``.

Fidelity to the old scripts:
  - the uncertainty dispatch below is copied verbatim from them (one function
    per script), and imports the old ``utils`` package for everything else;
  - pixels are ordered with the same ``np.argsort`` call on the same float32
    maps, so tied uncertainties break the same way;
  - Dice curves use the same rates, including the old slicing quirks (see
    ``segmentation_curves.old_keep_counts``);
  - with ``--validate-dir`` the image-averaged curves are compared with the
    ``dice_scores.npy`` files the original runs saved.

Spec file (one per dataset), e.g. for RIGA:
    {
      "variant": "microavg",                      # "microavg" (RIGA) or "binary" (LIDC-IDRI)
      "rates": {"start": 0.0001, "stop": 1.0, "num": 10000},
      "r_max": 0.8,
      "methods": {
        "tv_cae":  {"config": "configs/rejection_configs/main/tv_cae.json", "final_activation": true},
        "oracle":  {"config": "configs/rejection_configs/main/oracle.json"}
      }
    }
Relative config paths are resolved against ``--old-seg-dir``. ``final_activation``
mirrors the ``-f`` flag each method was originally run with (see ``plot_curves.sh``).

Usage:
    python -m stats.collect_segmentation --old-seg-dir /path/to/confidence-aware-uncertainty/segmentation \
        --spec stats/configs/segmentation_riga.json --out results/paired_tests/riga_images.npz \
        --validate-dir /path/to/segmentation/rcs/e3_10k_microavg_std_full_dice
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stats.data import ImageLevelData
from stats.segmentation_curves import (
    old_throwaway_rates, old_keep_counts, per_image_dice_curves, area_under_curves,
)


# ── Verbatim uncertainty dispatch from rejection_curves_microavg.py (RIGA) ────
# pylint: disable=too-many-arguments,too-many-branches

def slide_uncertainty_microavg(config, models_predictions, labels, models, expert_models,
                               n_classes, device, final_activation, old_unc):
    expert_path = config.get("expert_predictions", None)
    if expert_path is not None:
        expert_path = Path(expert_path)
        method = config.get("expert_method", None)

        if method == "mcmc":
            expert_predictions = old_unc.load_mcmc(expert_path, config["step_size"], config["total_epochs"])
            expert_predictions = torch.flatten(expert_predictions, start_dim=-2)

            net_uncs = []
            for model in expert_models:
                uncs = old_unc.multiclass_uncertainty(n_classes,
                                                      preds=None, expert_preds=expert_predictions[model],
                                                      method=None, expert_method="var")
                net_uncs.append(uncs)
            slide_uncs = torch.stack(net_uncs)

        elif method == "abnn":
            expert_predictions = F.softmax(torch.load(expert_path), dim=3).to(device)
            expert_predictions = expert_predictions[expert_models]
            expert_predictions = torch.movedim(torch.flatten(expert_predictions,
                                                             start_dim=-2), 2, 0)

            slide_uncs = old_unc.multiclass_uncertainty(n_classes,
                                                        preds=None, expert_preds=expert_predictions,
                                                        method=None, expert_method="var")

        elif method == "oracle":
            class_predictions = torch.argmax(models_predictions.detach(), dim=-2)
            slide_uncs = 1. - (class_predictions == torch.stack([labels] * len(models))).int()

        elif method == "test-aug":
            slide_uncs = torch.load(expert_path)[models]
            slide_uncs = torch.var(slide_uncs, dim=2)
            slide_uncs = torch.flatten(torch.sum(slide_uncs, dim=2), start_dim=-2).to(device)

        elif method == "edl":
            model_predictions = torch.load(expert_path)[models]
            evidence_scores = F.softplus(model_predictions)
            slide_uncs = n_classes / (torch.sum(evidence_scores + 1., dim=2))
            slide_uncs = torch.flatten(slide_uncs, start_dim=-2).to(device)

        else:
            expert_predictions = torch.load(config["expert_predictions"]).to(device)
            expert_predictions = expert_predictions[expert_models]
            if final_activation:
                expert_predictions = F.softmax(expert_predictions, dim=2)
            expert_predictions = torch.flatten(expert_predictions, start_dim=-2)

            slide_uncs = old_unc.multiclass_uncertainty(n_classes, preds=models_predictions,
                                                        expert_preds=expert_predictions,
                                                        method=config.get("models_method", None),
                                                        expert_method=method,
                                                        alpha=config.get("method_alpha", 0.5))
            slide_uncs = torch.stack([slide_uncs] * 10)
    else:
        method = config.get("models_method", None)
        if method == "single-net":
            slide_uncs = 1. - torch.max(models_predictions, dim=-2).values
        else:
            slide_uncs = old_unc.multiclass_uncertainty(n_classes=n_classes,
                                                        preds=models_predictions, expert_preds=None,
                                                        method=config.get("models_method", None),
                                                        expert_method=None)
            slide_uncs = torch.stack([slide_uncs] * 10)
    return slide_uncs


# ── Verbatim uncertainty dispatch from rejection_curves.py (LIDC-IDRI) ────────

def slide_uncertainty_binary(config, models_predictions, labels, models, expert_models,
                             device, final_activation, old_unc):
    expert_path = config.get("expert_predictions", None)
    if expert_path is not None:
        expert_path = Path(expert_path)
        method = config.get("expert_method", None)

        if method == "mcmc":
            expert_predictions = old_unc.load_mcmc_binary(expert_path, config["step_size"], config["total_epochs"])
            expert_predictions = torch.flatten(expert_predictions, start_dim=-2)

            net_uncs = []
            for model in expert_models:
                uncs = old_unc.multiclass_uncertainty(n_classes=1,
                                                      preds=None, expert_preds=expert_predictions[model],
                                                      method=None, expert_method="var")
                net_uncs.append(uncs)
            slide_uncs = torch.stack(net_uncs)

        elif method == "abnn":
            expert_predictions = torch.sigmoid(torch.load(expert_path)).to(device)
            expert_predictions = expert_predictions[expert_models]
            expert_predictions = torch.movedim(torch.flatten(expert_predictions,
                                                             start_dim=-2), 2, 0)

            slide_uncs = old_unc.multiclass_uncertainty(n_classes=1,
                                                        preds=None, expert_preds=expert_predictions,
                                                        method=None, expert_method="var")

        elif method == "edl":
            expert_predictions = torch.load(expert_path)[models]
            n_classes = 2
            evidence_scores = F.softplus(expert_predictions)
            slide_uncs = n_classes / (torch.sum(evidence_scores + 1., dim=2))
            slide_uncs = torch.flatten(slide_uncs, start_dim=-2).to(device)

        elif method == "oracle":
            class_predictions = (models_predictions > 0.5)
            slide_uncs = 1. - (class_predictions == torch.stack([labels] * len(models))).int()

        else:
            expert_predictions = torch.load(config["expert_predictions"]).to(device)
            expert_predictions = expert_predictions[expert_models]
            if final_activation:
                expert_predictions = torch.sigmoid(expert_predictions)
            expert_predictions = torch.flatten(expert_predictions, start_dim=-2)

            slide_uncs = old_unc.multiclass_uncertainty(n_classes=1, preds=models_predictions,
                                                        expert_preds=expert_predictions,
                                                        method=config.get("models_method", None),
                                                        expert_method=method)
            slide_uncs = torch.stack([slide_uncs] * 10)
    else:
        method = config.get("models_method", None)
        if method == "single-net":
            slide_uncs = 0.5 - torch.abs(models_predictions - 0.5)
        else:
            slide_uncs = old_unc.multiclass_uncertainty(n_classes=1,
                                                        preds=models_predictions, expert_preds=None,
                                                        method=config.get("models_method", None),
                                                        expert_method=None)
            slide_uncs = torch.stack([slide_uncs] * 10)
    return slide_uncs

# pylint: enable=too-many-arguments,too-many-branches


# ── Running one method ────────────────────────────────────────────────────────

def _ids(ids: int | list[int]) -> list[int]:
    return list(range(ids)) if isinstance(ids, int) else ids


def method_maps(variant: str, config: dict, final_activation: bool, device: torch.device,
                old_unc, flatten_labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted classes ``[nets, S, P]``, labels ``[S, P]`` and uncertainty ``[>=nets, S, P]``,
    prepared exactly as the old script for ``variant`` prepares them."""
    models, expert_models = _ids(config["models_ids"]), _ids(config["expert_ids"])

    if variant == "microavg":
        labels = flatten_labels(config["dataset"], config["data_path"], config["expert_id"]).to(device)
        preds = F.softmax(torch.load(config["models_predictions"]), dim=2).to(device)
        preds = torch.reshape(preds, preds.shape[:-2] + (-1,))
        n_classes = preds.shape[-2]
        preds = preds[models]
        slide_uncs = slide_uncertainty_microavg(config, preds, labels, models, expert_models,
                                                n_classes, device, final_activation, old_unc)
        pred_classes = torch.argmax(preds.detach(), dim=-2)
        labels = labels.squeeze()
    elif variant == "binary":
        labels = flatten_labels(config["dataset"], config["data_path"], config["expert_id"])
        labels = labels.squeeze().to(device)
        preds = torch.sigmoid(torch.load(config["models_predictions"])).to(device)
        preds = torch.reshape(preds, preds.shape[:-2] + (-1,))
        preds = preds[models]
        slide_uncs = slide_uncertainty_binary(config, preds, labels, models, expert_models,
                                              device, final_activation, old_unc)
        pred_classes = preds > 0.5
    else:
        raise ValueError(f"Unknown variant {variant!r}; expected 'microavg' or 'binary'.")

    return (pred_classes.cpu().numpy().astype(np.int64), labels.cpu().numpy().astype(np.int64),
            slide_uncs.cpu().numpy())


def image_areas(pred_classes: np.ndarray, labels: np.ndarray, slide_uncs: np.ndarray,
                rates: np.ndarray, r_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-image areas ``[nets, S]`` and image-averaged curves ``[nets, R]``."""
    keep = old_keep_counts(rates, pred_classes.shape[-1])
    areas, mean_curves = [], []
    for net in range(pred_classes.shape[0]):
        order = np.argsort(slide_uncs[net], axis=1)      # == np.argsort(slide_uncs, axis=2)[net]
        curves = per_image_dice_curves(pred_classes[net], labels, order, keep)
        areas.append(area_under_curves(curves, rates, r_max))
        mean_curves.append(curves.mean(axis=0))
    return np.stack(areas), np.stack(mean_curves)


def validate(method: str, mean_curve: np.ndarray, validate_dir: Path) -> None:
    """Compare image-averaged curves with the original run's dice_scores.npy."""
    saved = np.load(validate_dir / method / "dice_scores.npy")
    if saved.shape[-1] != mean_curve.shape[-1]:
        raise AssertionError(
            f"{method}: the saved run has {saved.shape[-1]} rates but the spec gives "
            f"{mean_curve.shape[-1]}; set spec['rates'] to the grid that run used.")
    saved = saved.reshape(-1, mean_curve.shape[-1])
    n = min(len(saved), len(mean_curve))
    worst = np.abs(saved[:n] - mean_curve[:n]).max()
    if worst > 1e-5:   # the old script computed Dice in float32
        raise AssertionError(f"{method}: curves differ from the saved run (max |diff| = {worst:.2e}).")
    print(f"  {method}: matches the saved run on {n} net(s) (max curve diff {worst:.1e})")


def collect(old_seg_dir: Path, spec: dict, device: torch.device,
            validate_dir: Path | None) -> ImageLevelData:
    sys.path.insert(0, str(old_seg_dir))
    os.chdir(old_seg_dir)
    old_unc = importlib.import_module("utils.uncertainty")
    flatten_labels = importlib.import_module("utils.object_loader").flatten_labels

    rates = old_throwaway_rates(**spec.get("rates", {}))
    r_max = float(spec.get("r_max", 0.8))
    data = ImageLevelData(rates=rates, r_max=r_max)

    for method, info in spec["methods"].items():
        with open(info["config"], "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"{method}: computing uncertainty and per-image curves...")
        with torch.no_grad():
            pred_classes, labels, slide_uncs = method_maps(
                spec["variant"], config, bool(info.get("final_activation", False)),
                device, old_unc, flatten_labels)
        data.image_auc[method], data.mean_curve[method] = image_areas(
            pred_classes, labels, slide_uncs, rates, r_max)
        if validate_dir is not None:
            validate(method, data.mean_curve[method], validate_dir)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old-seg-dir", type=Path, required=True,
                        help="The old segmentation/ directory (containing utils/).")
    parser.add_argument("--spec", type=Path, required=True, help="Dataset spec JSON (see module docstring).")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz file.")
    parser.add_argument("--validate-dir", type=Path, default=None,
                        help="rcs/<run> dir of the original run, with <method>/dice_scores.npy.")
    parser.add_argument("--device", default="cpu",
                        help="Device for the uncertainty maps (the old scripts used cpu).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    for info in spec["methods"].values():     # relative config paths are relative to --old-seg-dir
        info["config"] = str((args.old_seg_dir / info["config"]).resolve())
    validate_dir = args.validate_dir.resolve() if args.validate_dir else None
    data = collect(args.old_seg_dir.resolve(), spec, torch.device(args.device), validate_dir)
    data.save(out)
    print(f"Saved per-image areas for {len(data.image_auc)} method(s) to {out}")


if __name__ == "__main__":
    main()
