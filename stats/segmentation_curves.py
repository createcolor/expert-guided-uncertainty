"""Per-image Dice-rejection curves for segmentation.

The old ``rejection_curves.py`` (LIDC-IDRI, binary) and
``rejection_curves_microavg.py`` (RIGA, multiclass) sort each image's pixels by
uncertainty, drop the most uncertain fraction, compute Dice *per image* and
then average over images. Averaging over images is linear, so the dataset AAC is
the mean of per-image AACs -- images are the natural paired units.

This module returns the per-image curves instead of their mean. Rather than
re-slicing the sorted pixels at each of 10k rates, it uses cumulative counts,
which gives the same Dice values. Both old Dice variants share one formula:

  - ``stacked_Dice`` (binary): predictions and labels are {0, 1}.
  - ``stacked_micro_Dice`` (multiclass): foreground classes are pooled; a pixel
    counts as an intersection when the predicted class equals a foreground label.

  dice = (2 * #(pred == label > 0) + smooth) / (#(pred > 0) + #(label > 0) + smooth)
"""

from __future__ import annotations

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 compatibility


def old_throwaway_rates(start: float = 0.0001, stop: float = 1.0, num: int = 10_000) -> np.ndarray:
    """Rates used by the old scripts, with the prepended 0.0 (``np.linspace(start, stop, num)``)."""
    return np.insert(np.linspace(start, stop, num), 0, 0.0)


def old_keep_counts(rates: np.ndarray, n_pixels: int) -> np.ndarray:
    """Number of least-uncertain pixels kept at each rate, as the old loop slices them.

    The old code scored rate 0 on all pixels, then sliced ``[:, :-throwaway_num]``
    with ``throwaway_num = round(rate * n_pixels)``. When that rounds to 0 the
    slice ``[:, :-0]`` is *empty*; this is reproduced so curves match exactly.
    """
    counts = [n_pixels]
    for rate in rates[1:]:
        throwaway_num = round(rate * n_pixels)
        counts.append(n_pixels - throwaway_num if throwaway_num > 0 else 0)
    return np.asarray(counts, dtype=np.int64)


def per_image_dice_curves(pred_classes: np.ndarray, labels: np.ndarray, order: np.ndarray,
                          keep_counts: np.ndarray, smooth: float = 1.0) -> np.ndarray:
    """Dice-rejection curve of every image for one net.

    Args:
        pred_classes: ``[S, P]`` predicted class per pixel (0 = background).
        labels: ``[S, P]`` ground-truth class per pixel (0 = background).
        order: ``[S, P]`` ascending-uncertainty pixel order per image
            (``np.argsort(uncertainty, axis=1)``, as in the old code).
        keep_counts: ``[R]`` pixels kept at each rate (see ``old_keep_counts``).

    Returns:
        ``[S, R]`` Dice per image and rate.
    """
    pred = np.take_along_axis(pred_classes, order, axis=1)
    lab = np.take_along_axis(labels, order, axis=1)

    def cumulative(mask: np.ndarray) -> np.ndarray:
        out = np.zeros((mask.shape[0], mask.shape[1] + 1), dtype=np.int64)
        np.cumsum(mask, axis=1, out=out[:, 1:])
        return out

    intersection = cumulative((pred == lab) & (lab > 0))[:, keep_counts]
    pred_fg = cumulative(pred > 0)[:, keep_counts]
    label_fg = cumulative(lab > 0)[:, keep_counts]
    return (2.0 * intersection + smooth) / (pred_fg + label_fg + smooth)


def area_under_curves(curves: np.ndarray, rates: np.ndarray, r_max: float = 0.8) -> np.ndarray:
    """Integral of each curve over ``[0, r_max]`` (the old ``aac_std.py`` convention).

    ``AAC = AUC(oracle) - AUC(method)``, so differences between methods only
    need these areas.
    """
    mask = rates <= r_max + 1e-9
    return _trapezoid(curves[..., mask], rates[mask], axis=-1)
