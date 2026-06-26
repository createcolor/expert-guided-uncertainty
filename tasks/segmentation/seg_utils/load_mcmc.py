"""Segmentation-specific loading of MCMC prediction chains.

The uncertainty math (variance / Bernoulli variance / total variance) now lives
in :mod:`common.estimators`; only this dataset-shaped loader remains here.
"""

from pathlib import Path

import torch


def load_mcmc(path: Path | str, step_size: int, total_epochs: int) -> torch.Tensor:
    """
    Loads predictions of an MCMC model taken with checkpoints from a directory.

    Args:
        path (Path | str): path to the directory containing the predictions.
        step_size (int): size of the step with which the checkpoints were taken.
        total_epochs (int): number of epochs the model was trained for.

    Returns:
        torch.Tensor: a tensor of shape (num. nets, num. checkpoints, num. samples,
                                         num. classes, height, width)
    """
    epoch_preds = []
    for step in range(step_size, total_epochs + 1, step_size):
        epoch_ens = torch.load(Path(path) / f"{step}_epochs" / "predictions.pt")
        epoch_preds.append(torch.softmax(epoch_ens, dim=-3))

    epoch_preds = torch.movedim(torch.stack(epoch_preds), 0, 1)
    return epoch_preds
