import os
from pathlib import Path
import argparse

import json
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser("Analyzes experts")
    
    parser.add_argument('-r', '--dataset_dir', default=Path(
        "data/BloodyWell"), type=Path, \
        help='Path to images')

    parser.add_argument('-tm', '--test_markup', default=Path(
        "data/markup/test_dataset.json"), \
            type=Path, help='Path to test dataset markup')

    parser.add_argument('-e', '--path2experts_markup', default=Path(
        "data/markup/experts_info_per_alvs.json"), \
            type=Path, help='Path to experts markup')

    parser.add_argument('-log', '--log_dir', type=str, \
        default=Path("."), help='Log dir path')
    
    args = parser.parse_args()

    return args


experts_order = [5, 4, 2, 3, 0, 1]


def calculate_kappa_scores(experts_votes):
    '''
    Estimates experts' consistency using Cohen Kappa coefficient
    '''

    experts_number = 6
    matrix = np.zeros(shape=(experts_number, experts_number)) + 9999999

    for exp_id in range(experts_number):
        # exp_id = experts_order[exp_id]

        for another_exp_id in range(experts_number):
            # another_exp_id = experts_order[another_exp_id]

            pair_kappa = round(cohen_kappa_score(experts_votes[:, experts_order[exp_id]], experts_votes[:, experts_order[another_exp_id]]), 3)
            matrix[exp_id][another_exp_id] = pair_kappa
            matrix[another_exp_id][exp_id] = None
    
    return matrix


def plot_kappa_matrix(matrix, log_dir, accuracy):

    colors = ["#E0695E", "#FFFFFF", "#00999E"]

    # colors = ["#F7C815", "#FFFFFF", "#9048E1"]

    # "#4B3A8C" "#E67828" "#00796B" "#26C6DA" "#BF2A2A" "#1B75B4" "#9048E1" "#4CAF50" "#F7C815" "#C0587E"

    custom_cmap = LinearSegmentedColormap.from_list("RedWhiteTeal", colors, N=256)

    fig, ax = plt.subplots()
    caxes = ax.matshow(matrix, cmap=custom_cmap, vmin = 0.6, vmax = 1.0)
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.set_xlabel("Expert ID", fontsize=14)
    ax.set_ylabel("Expert ID", fontsize=14)
    ax.xaxis.set_label_position('top')

    for exp in range(6):
        matrix[exp, exp] = f"{accuracy[experts_order[exp]]:.3f}"
    
    for (i, j), z in np.ndenumerate(matrix):
        if i > j:
            ax.text(j, i, f"{z:.3f}", ha='center', va='center',
                # bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3')
                )
        if i == j:
            ax.text(j, i, f"{z * 100:.1f}%", ha='center', va='center', color="black")
    
    # Create a colorbar
    # cbar = ax.figure.colorbar(caxes, ax=ax, cmap=custom_cmap)
   
    cbar = fig.colorbar(caxes)
    cbar.ax.set_ylabel("Cohen's kappa coefficient", fontsize=14, rotation=-90, va="bottom")
    
    plt.savefig(log_dir / 'Cohen_Kappa_coef.pdf')

if __name__ == "__main__":
    args = parse_args()

    with open(args.path2experts_markup, "r", encoding="utf-8") as f:
        experts_alvs_unc_dict = json.load(f)

    gt_results = []
    accuracy = []
    experts_votes = []
    for alvs, data in experts_alvs_unc_dict.items():
        if "expert_votes_positions" in data and data["gt_result"] is not None:
            experts_votes.append(data["expert_votes_positions"])
            gt_results.append(data["gt_result"])

    for exp in range(6):
        scores = []
        votes = np.asarray(experts_votes)[:, exp]
        for i, vote in enumerate(votes):
            if vote != 2:
                scores.append(1 - np.abs(gt_results[i] - vote / 4))

        accuracy.append(np.mean(scores))
        # accuracy.append(np.mean(1 - np.abs(np.asarray(gt_results) - votes / 4)))

    print(accuracy)
    matrix = calculate_kappa_scores(np.asarray(experts_votes))
    plot_kappa_matrix(matrix, args.log_dir, accuracy)