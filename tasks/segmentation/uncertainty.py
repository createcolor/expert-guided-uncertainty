"""Compute the per-pixel uncertainty map for a segmentation ensemble.

This is the pipeline step between inference and rejection curves. It used to be
done inline inside the rejection-curve scripts; pulling it out gives
segmentation the same ``inference -> uncertainty -> rejection_curves`` shape as
the other tasks, and routes the actual math through :mod:`common.estimators`.

Total uncertainty follows the law of total variance: an epistemic component
from the model ensemble plus an aleatoric component from an expert source. The
expert source can be a plain ensemble, an MCMC chain, or an ABNN sampler.

Output: a tensor ``uncertainty.pt`` shaped ``[members, samples, pixels]`` where
the pixel axis is the flattened ``H * W`` map, ready to rank pixels in the
rejection step.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common.estimators import multiclass_total_variance
from tasks.segmentation.seg_utils.load_mcmc import load_mcmc


def _as_ids(ids: int | list[int]) -> list[int]:
    return list(range(ids)) if isinstance(ids, int) else ids


def _flatten_spatial(t: torch.Tensor) -> torch.Tensor:
    """Flatten the trailing ``H, W`` axes into a single pixel axis."""
    return torch.reshape(t, t.shape[:-2] + (-1,))


def _to_probabilities(t: torch.Tensor) -> torch.Tensor:
    """Map logits to probabilities by shape.

    Binary predictions have no class axis (``[..., samples, pixels]``) and use a
    sigmoid; multiclass predictions carry a class axis at dim 2
    (``[..., samples, classes, pixels]``) and use a softmax over it. This matches
    the activation the Dice metric applies at inference time.
    """
    return torch.sigmoid(t) if t.ndim == 3 else F.softmax(t, dim=2)


def load_model_predictions(path: str | Path, model_ids: int | list[int],
                           device: torch.device) -> torch.Tensor:
    """Load ensemble predictions, flatten spatial dims, map to probabilities.

    Returns ``[members, samples, pixels]`` for binary segmentation or
    ``[members, samples, classes, pixels]`` for multiclass.
    """
    preds = _to_probabilities(_flatten_spatial(torch.load(path).to(device)))
    return preds[_as_ids(model_ids)]


def expert_uncertainty(config: dict, n_classes: int, model_preds: torch.Tensor,
                       device: torch.device, final_activation: bool = False) -> torch.Tensor:
    """Per-pixel uncertainty when an expert (aleatoric) source is configured.

    Dispatches on ``expert_method``: ``"mcmc"`` and ``"abnn"`` consume their own
    sampler outputs as the aleatoric source; any other value treats
    ``expert_predictions`` as a plain expert ensemble combined with the model
    ensemble's epistemic variance.
    """
    expert_path = Path(config["expert_predictions"])
    expert_ids = _as_ids(config["expert_ids"])
    method = config.get("expert_method", None)
    n_members = model_preds.shape[0]

    if method == "mcmc":
        chains = load_mcmc(expert_path, config["step_size"], config["total_epochs"])
        chains = torch.flatten(chains, start_dim=-2)  # [nets, checkpoints, samples, classes, pixels]
        per_net = [
            multiclass_total_variance(n_classes, epist_preds=None, aleat_preds=chains[net],
                                      epist_method=None, aleat_method="var")
            for net in expert_ids
        ]
        return torch.stack(per_net)

    if method == "abnn":
        samples = F.softmax(torch.load(expert_path), dim=3).to(device)[expert_ids]
        samples = torch.movedim(torch.flatten(samples, start_dim=-2), 2, 0)
        return multiclass_total_variance(n_classes, epist_preds=None, aleat_preds=samples,
                                         epist_method=None, aleat_method="var")

    expert_preds = _flatten_spatial(torch.load(expert_path).to(device)[expert_ids])
    if final_activation:
        expert_preds = _to_probabilities(expert_preds)

    unc = multiclass_total_variance(n_classes, epist_preds=model_preds, aleat_preds=expert_preds,
                                    epist_method=config.get("models_method", None),
                                    aleat_method=method)
    return torch.stack([unc] * n_members)


def compute_uncertainty(config: dict, device: torch.device,
                        final_activation: bool = False) -> torch.Tensor:
    """Return the per-pixel uncertainty map shaped ``[members, samples, pixels]``."""
    model_preds = load_model_predictions(config["models_predictions"],
                                         config["models_ids"], device)
    # Binary predictions have no class axis (ndim 3); multiclass carries one at -2.
    n_classes = 1 if model_preds.ndim == 3 else model_preds.shape[-2]

    if config.get("expert_predictions") is not None:
        return expert_uncertainty(config, n_classes, model_preds, device, final_activation)

    unc = multiclass_total_variance(n_classes, epist_preds=model_preds, aleat_preds=None,
                                    epist_method=config.get("models_method", None),
                                    aleat_method=None)
    return torch.stack([unc] * model_preds.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute per-pixel segmentation uncertainty.")
    parser.add_argument("-c", "--config", type=Path, required=True,
                        help="Path to the rejection/uncertainty config file.")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Directory to write uncertainty.pt to.")
    parser.add_argument("-f", "--final_activation", action="store_true",
                        help="Map plain expert predictions to probabilities "
                             "(sigmoid for binary, softmax for multiclass) before use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Computing uncertainty...")
    uncertainty = compute_uncertainty(config, device, args.final_activation)
    print("Uncertainty map shape:", tuple(uncertainty.shape))

    os.makedirs(args.output, exist_ok=True)
    out_path = args.output / "uncertainty.pt"
    torch.save(uncertainty.cpu(), out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
