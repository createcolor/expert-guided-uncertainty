from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


PathLike = Union[str, Path]


def _load_curve(path: PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one pickle file which is a CurveData file.

    Returns
    -------
    (rates, means, stds): each a 1D numpy array sorted by increasing rate.
    """

    path = Path(path)
    with path.open("r", encoding='utf-8') as f:
        data: dict[str, list[float] | float] = json.load(f)

    # Keys are strings of floats; convert and sort
    rates_floats: List[float] = []
    means_list: List[float] = []
    stds_list: List[float] = []
    for k, v in data.items():
        if not isinstance(v, list):
            v = [v]
        rate = float(k)
        mean = float(v[0])
        std = float(v[1]) if len(v) > 1 else 0.0
        rates_floats.append(rate)
        means_list.append(mean)
        stds_list.append(std)

    # sort by rate
    order = np.argsort(rates_floats)
    rates = np.array(rates_floats, dtype=float)[order]
    means = np.array(means_list, dtype=float)[order]
    stds = np.array(stds_list, dtype=float)[order]
    return rates, means, stds


def plot_throwaway_curves(
    paths: Sequence[PathLike],
    labels: Optional[Sequence[str]] = None,
    show_std: bool = False,
    std_style: str = "band",  # "band" or "errorbar"
    title: Optional[str] = None,
    y_label: str = "Accuracy",
    x_label: str = "Throwaway rate",
    save_path: Optional[PathLike] = None,
    figsize: Tuple[float, float] = (5.0, 3.5),
    dpi: int = 200,
    palette: Optional[Sequence[str]] = None,
    markers: Optional[Sequence[str]] = None
) -> plt.Figure:
    """Plot mean vs throwaway rate for multiple JSON files as histograms (grouped bars)."""

    # Academic-style aesthetics
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if labels is None:
        labels = [Path(p).stem for p in paths]
    if len(labels) != len(paths):
        raise ValueError("Length of labels must match number of paths")

    if palette is not None and len(palette) < len(paths):
        raise ValueError("Palette must have at least as many colors as paths")

    # Use a subtle grid for readability
    ax.grid(True, which="major", axis="both", linewidth=0.6, alpha=0.3)
    ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.2)

    lowest_point: float = float("inf")
    highest_point: float = 0.

    # Load all data first to determine common X values and number of groups
    all_data = []
    common_rates = None
    for path in paths:
        rates, means, stds = _load_curve(path)
        if common_rates is None:
            common_rates = np.array(rates)
        else:
            if not np.array_equal(common_rates, rates):
                raise ValueError("All files must have the same X (rates) values for grouped histogram.")
        all_data.append((means, stds))
        lowest_point = min(float(np.min(means)), lowest_point)
        highest_point = max(float(np.max(means)), highest_point)

    n_files = len(paths)
    n_bins = len(common_rates)
    bar_width = 0.8 / n_files  # total width per group is 0.8, divided among files

    # Plot bars
    for i, (path, label, (means, stds)) in enumerate(zip(paths, labels, all_data)):
        color = palette[i] if palette is not None else None
        offset = (i - n_files / 2 + 0.5) * bar_width
        x_positions = common_rates + offset

        ax.bar(
            x_positions,
            means,
            width=bar_width,
            label=label,
            color=color,
            alpha=1.,
            edgecolor='white',
            linewidth=1.5
        )

        # Optional: add error bars if show_std=True (not in original ask, but easy to add)
        if show_std and std_style == "errorbar":
            ax.errorbar(
                x_positions,
                means,
                yerr=stds,
                fmt='none',
                ecolor='black',
                capsize=1.5,
                linewidth=0.6,
                alpha=0.6
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # Set x-axis to show integer ticks only
    ax.set_xticks(common_rates)
    ax.set_xticklabels([str(int(x)) if float(x).is_integer() else str(x) for x in common_rates])

    ax.legend(loc="upper right", frameon=False)

    if len(paths) > 0:
        # ax.set_xlim(left=0.0, right=1.0)
        ax.set_ylim(bottom=0.95 * lowest_point, top=1.05 * highest_point)

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def _parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=
            "Plot mean vs throwaway rate from JSON files {rate: [mean, std]}"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Paths to JSON files",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each file (same length as files)",
    )
    parser.add_argument(
        "--palette",
        nargs="*",
        default=None,
        help="Optional color palette for each file (same length as files)",
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Display standard deviation (±std)",
    )
    parser.add_argument(
        "--std-style",
        choices=["band", "errorbar"],
        default="band",
        help="How to render std when enabled",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title",
    )
    parser.add_argument(
        "--y-label",
        default=r"AAC, $\times 10^3$",
        help="Y axis label",
    )
    parser.add_argument(
        "--x-label",
        default="Number of expert predictions",
        help="X axis label",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g., figure.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saving",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        default=(5.0, 3.5),
        help="Figure size in inches",
    )
    return parser.parse_args()


def main_cli():
    args = _parse_cli_args()
    fig = plot_throwaway_curves(
        paths=args.files,
        labels=args.labels,
        show_std=args.show_std,
        std_style=args.std_style,
        title=args.title,
        y_label=args.y_label,
        x_label=args.x_label,
        save_path=args.save,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        palette=args.palette,
    )
    if args.save is None:
        plt.show()
    return fig


if __name__ == "__main__":
    main_cli()


