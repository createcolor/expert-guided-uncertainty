"""On-disk formats shared by the collectors and ``run_paired_tests`` (numpy only).

Item-level (PubMedQA, BloodyWell), ``ItemLevelData``:
    item_ids     [N]          test item identifiers
    wrong        [nets, N]    base-ensemble mistakes (correctness is always the base ensemble's)
    uncertainty  {method: [N] or [nets, N]}

Image-level (LIDC-IDRI, RIGA), ``ImageLevelData``:
    rates        [R]                    throwaway rates of the stored curves
    image_auc    {method: [nets, S]}    area under each image's Dice curve on [0, r_max]
    mean_curve   {method: [nets, R]}    curve averaged over images (equals the old dice_scores.npy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_UNC, _AUC, _CURVE = "unc__", "auc__", "curve__"


@dataclass
class ItemLevelData:
    item_ids: np.ndarray
    wrong: np.ndarray
    uncertainty: dict[str, np.ndarray] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        arrays = {_UNC + name: values for name, values in self.uncertainty.items()}
        np.savez_compressed(path, kind="item", item_ids=self.item_ids.astype(str),
                            wrong=self.wrong, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "ItemLevelData":
        with np.load(path, allow_pickle=False) as npz:
            _check_kind(npz, "item", path)
            unc = {key[len(_UNC):]: npz[key] for key in npz.files if key.startswith(_UNC)}
            return cls(item_ids=npz["item_ids"], wrong=npz["wrong"].astype(bool), uncertainty=unc)


@dataclass
class ImageLevelData:
    rates: np.ndarray
    r_max: float
    image_auc: dict[str, np.ndarray] = field(default_factory=dict)
    mean_curve: dict[str, np.ndarray] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        arrays = {_AUC + name: values for name, values in self.image_auc.items()}
        arrays.update({_CURVE + name: values for name, values in self.mean_curve.items()})
        np.savez_compressed(path, kind="image", rates=self.rates, r_max=self.r_max, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "ImageLevelData":
        with np.load(path, allow_pickle=False) as npz:
            _check_kind(npz, "image", path)
            auc = {key[len(_AUC):]: npz[key] for key in npz.files if key.startswith(_AUC)}
            curve = {key[len(_CURVE):]: npz[key] for key in npz.files if key.startswith(_CURVE)}
            return cls(rates=npz["rates"], r_max=float(npz["r_max"]), image_auc=auc, mean_curve=curve)


def _check_kind(npz, expected: str, path) -> None:
    kind = str(npz["kind"])
    if kind != expected:
        raise ValueError(f"{path} holds {kind}-level data, expected {expected}-level.")
