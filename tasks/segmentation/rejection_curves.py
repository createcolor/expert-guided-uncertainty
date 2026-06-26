"""Pixel-level rejection curves for segmentation (binary and multiclass).

This replaces the two near-duplicate ``rejection_curves.py`` /
``rejection_curves_multiclass.py`` scripts. Uncertainty is no longer computed
here -- it is read from ``uncertainty.pt`` produced by ``uncertainty.py`` -- and
the curve itself is built by the shared :func:`common.rejection.sweep_members`
driver with a Dice scoring callback.

Binary segmentation is handled as the single-foreground-class case of the same
logic. Predictions are reduced to per-class membership maps the same way the
Dice metric does at inference time: sigmoid + threshold for the single binary
channel, softmax + argmax one-hot for multiclass.

Pipeline:
    inference.py  ->  predictions.pt
    uncertainty.py -> uncertainty.pt
    rejection_curves.py (this) -> rejection_curve_<c>.json + plot
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common.rejection import sweep_members
from tasks.segmentation.seg_utils.metrics.dice import stacked_Dice
from tasks.segmentation.seg_utils.object_loader import flatten_labels

# Dense throwaway grid near 0 (where the curve moves fastest), coarser afterwards.
DEFAULT_THROWAWAY_RATES = np.concatenate(
    (np.linspace(0.0001, 0.1, 1000), np.linspace(0.101, 1.0, 900))
)


def to_membership(predictions: torch.Tensor, labels: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Reduce raw predictions and labels to per-class binary membership maps.

    Args:
        predictions: raw ensemble logits, ``[members, samples, classes, pixels]``
            (a missing class axis is treated as a single binary channel).
        labels: integer label map ``[samples, pixels]``.

    Returns:
        pred_member ``[members, samples, classes, pixels]`` (float 0/1),
        label_member ``[samples, classes, pixels]`` (float 0/1),
        and the list of class indices to evaluate (background skipped for
        multiclass).
    """
    # flatten_labels can carry a singleton channel axis (LIDC labels are stored as
    # [1, H, W], giving [samples, 1, pixels]); drop it to a clean [samples, pixels] map.
    if labels.ndim == 3 and labels.shape[1] == 1:
        labels = labels[:, 0]

    if predictions.ndim == 3:                      # [members, samples, pixels] -> add class axis
        predictions = predictions.unsqueeze(2)
    n_classes = predictions.shape[2]

    if n_classes == 1:
        pred_member = (torch.sigmoid(predictions) >= 0.5).float()        # [M, S, 1, P]
        label_member = labels.unsqueeze(1).float()                       # [S, 1, P]
        eval_classes = [0]
    else:
        probs = F.softmax(predictions, dim=2)
        argmax = torch.argmax(probs, dim=2)                              # [M, S, P]
        pred_member = F.one_hot(argmax, num_classes=n_classes).movedim(-1, 2).float()
        label_member = F.one_hot(labels.long(), num_classes=n_classes).movedim(-1, 1).float()
        eval_classes = list(range(1, n_classes))                         # skip background
    return pred_member, label_member, eval_classes


def dice_member_curve(pred_member: torch.Tensor, label_member: torch.Tensor, cls: int):
    """Build a Dice scoring callback for one class, for the shared driver.

    Returns ``member_curve(member_idx, order, keep_counts) -> [dice per keep]``:
    it sorts the member's pixels by the supplied uncertainty order once, then
    scores every retained prefix.
    """
    def member_curve(member_idx: int, order, keep_counts):
        preds_sorted = torch.take_along_dim(pred_member[member_idx, :, cls], order, dim=-1)
        labels_sorted = torch.take_along_dim(label_member[:, cls], order, dim=-1)
        scores = []
        for keep in keep_counts:
            score = stacked_Dice(preds_sorted[:, :keep], labels_sorted[:, :keep], binarize=False)
            scores.append(score.mean().item())
        return scores
    return member_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build pixel-level segmentation rejection curves.")
    parser.add_argument("-c", "--config", type=Path, required=True,
                        help="Path to the rejection config file.")
    parser.add_argument("-u", "--uncertainty", type=Path, required=True,
                        help="Path to uncertainty.pt produced by uncertainty.py.")
    parser.add_argument("-s", "--save_path", type=Path, required=True,
                        help="Directory to save the rejection curves.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading predictions, labels and uncertainty...")
    raw_preds = torch.load(config["models_predictions"]).to(device)
    raw_preds = torch.reshape(raw_preds, raw_preds.shape[:-2] + (-1,))   # flatten H, W
    model_ids = config["models_ids"]
    model_ids = list(range(model_ids)) if isinstance(model_ids, int) else model_ids
    raw_preds = raw_preds[model_ids]

    labels = flatten_labels(config["dataset"], config["data_path"], config["expert_id"]).to(device)
    uncertainty = torch.load(args.uncertainty).to(device)               # [members, samples, pixels]

    pred_member, label_member, eval_classes = to_membership(raw_preds, labels)
    n_pixels = pred_member.shape[-1]

    # One ranking shared across classes -> sort once, reuse for each class curve.
    order = torch.argsort(uncertainty, dim=-1)
    rates = [0.0] + list(DEFAULT_THROWAWAY_RATES)

    os.makedirs(args.save_path, exist_ok=True)
    print(f"Building rejection curves for classes {eval_classes}...")
    for cls in eval_classes:
        member_curve = dice_member_curve(pred_member, label_member, cls)
        mean_dice, _ = sweep_members(order, n_pixels, member_curve, rates, member_axis=0)
        print(f"  class {cls}: Dice {mean_dice[0]:.4f} -> {mean_dice[-1]:.4f}")

        plt.plot(rates, mean_dice, label=f"class {cls}")
        curve = {str(rate): str(score) for rate, score in zip(rates, mean_dice)}
        with open(args.save_path / f"rejection_curve_{cls}.json", "w", encoding="utf-8") as f:
            json.dump(curve, f, ensure_ascii=False, indent=4)

    plt.grid()
    plt.legend()
    plt.savefig(args.save_path / "rejection_curve.png", format="png", bbox_inches="tight")
    plt.close()
    print(f"Saved curves to {args.save_path}")


if __name__ == "__main__":
    main()
