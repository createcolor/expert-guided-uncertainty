import sys
import json
from pathlib import Path
from typing import Any, NamedTuple
# from collections import 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


# class MeanCurve():
#     """
#     Class to store and save ROC curves.
#     """
#     def __init__(
#             self,
#             tpr_mean: np.ndarray[Any, np.dtype[np.float64]],
#             tpr_std: np.ndarray[Any, np.dtype[np.float64]],
#             score: float,
#             score_std: float
#     ):
#         self.tpr = tpr_mean
#         self.tpr_std = tpr_std
#         self.score = score
#         sel, NamedTuplef.score_std = score_std
# CurveData = (
#     "CurveData",
#     ['tpr', 'tpr_std', 'score', 'score_std']
# )

class CurveData(NamedTuple):
    """
    Class to store and save ROC curves.
    """
    tpr_mean: np.ndarray[Any, np.dtype[np.float64]]
    tpr_std: np.ndarray[Any, np.dtype[np.float64]]
    score: float
    score_std: float


def predictions_from_dict_pmqa(
        errors: dict[str, list[list[bool]]], model_id: int
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.int32]]]:
    """
    Converts an uncertainty dictionary to an array of predictions and an array of true errors.
    Uncertainty dictionary is loaded from a `.json` file and is of format
        {uncertainty_score: list of list of bool}. 
    """
    predictions: list[float] = []
    trues: list[int] = []

    for uncertainty, true_list in errors.items():
        uncertainty = float(uncertainty)
        for item in true_list:
            predictions.append(uncertainty)
            trues.append(int(item[model_id]))
        # if len(true_list) > 1:
        #     print(len(true_list))  # Debug
        # true_list = true_list[0]  # For some reason, list of bool is nested with singular element. Shit I understand the reason now god damn it.
        # assert len(true_list) == 10  # Debug
        # for item in true_list:
            # predictions.append(uncertainty)
            # trues.append(int(item))

        # predictions.append(uncertainty)
        # trues.append(int(true_list[model_id]))

    predictions_arr = np.array(predictions)
    trues_arr = np.array(trues)

    return predictions_arr, trues_arr


def convert_ensemble_predictions_pmqa(
        uncertainty_path: str, ensemble_size: int, load_from_dir: bool = False
) -> tuple[list[np.ndarray[Any, np.dtype[np.float64]]], list[np.ndarray[Any, np.dtype[np.int32]]]]:
    """
    Loads uncertainty values and errors for an ensemble from either a .json or a directory.
    """
    uncertainty_p = Path(uncertainty_path)
    if uncertainty_p.is_file() and load_from_dir:
        raise ValueError(f"{uncertainty_p} is a file but a directory was expected.")
    elif uncertainty_p.is_dir() and not load_from_dir:
        raise ValueError(f"{uncertainty_p} is a directory but a file was expected.")

    y_scores: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    y_trues: list[np.ndarray[Any, np.dtype[np.int32]]] = []

    json_paths: list[Path] = []
    uncertainties: dict[str, list[list[bool]]] = {}

    if load_from_dir:
        json_paths = sorted(uncertainty_p.glob("*.json"))
        # print(list(enumerate(json_paths)))
        # print(json_paths)
        if ensemble_size != len(json_paths):
            raise ValueError(f"Ensemble size {ensemble_size} doesn't match directory size {len(json_paths)}.")
    else:
        with open(uncertainty_path, 'r', encoding='utf-8') as file:
            uncertainties = json.load(file)

    for model_id in range(ensemble_size):
        if load_from_dir:
            with open(json_paths[model_id], 'r', encoding='utf-8') as file:
                uncertainties = json.load(file)

        y_score, y_true = predictions_from_dict_pmqa(uncertainties, model_id)
        y_scores.append(y_score)
        y_trues.append(y_true)

    return y_scores, y_trues


def predictions_from_dict_bw(
        uncertainties: dict[str, list[str]], errors: list[str]
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.int32]]]:
    """
    Converts an uncertainty dictionary and an error dictionary to an array of predictions
    and an array of true errors.
    Uncertainty dictionary is loaded from a `.json` file and is of format
        {uncertainty_score: list of samples with the corresponding uncertainty score}.
    Error dictionary is loaded from a `.json` file and is of format
        {model_id: list of errors}.
    """

    predictions: list[float] = []
    trues: list[int] = []

    model_errors = errors

    for uncertainty, sample_list in uncertainties.items():
        if uncertainty == "NaN":
            uncertainty = 0.
        else:
            uncertainty = float(uncertainty)
        for item in sample_list:
            predictions.append(uncertainty)
            trues.append(int(item in model_errors))

    predictions_arr = np.array(predictions)
    trues_arr = np.array(trues)

    return predictions_arr, trues_arr


def convert_ensemble_predictions_bw(
        uncertainty_path: str, error_path: str, ensemble_size: int, load_from_dir: bool = False
) -> tuple[list[np.ndarray[Any, np.dtype[np.float64]]], list[np.ndarray[Any, np.dtype[np.int32]]]]:
    """
    Loads uncertainty values and errors for an ensemble from either a .json or a directory.
    """
    uncertainty_p = Path(uncertainty_path)
    if uncertainty_p.is_file() and load_from_dir:
        raise ValueError(f"{uncertainty_p} is a file but a directory was expected.")
    elif uncertainty_p.is_dir() and not load_from_dir:
        raise ValueError(f"{uncertainty_p} is a directory but a file was expected.")

    with open(error_path, 'r', encoding='utf-8') as file:
        errors: dict[str, list[str]] = json.load(file)

    y_scores: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    y_trues: list[np.ndarray[Any, np.dtype[np.int32]]] = []

    json_paths: list[Path] = []
    uncertainties: dict[str, list[str]] = {}

    if load_from_dir:
        json_paths = sorted(uncertainty_p.glob("*.json"))[110:120]
        # print(list(enumerate(json_paths)))
        # print(json_paths)
        if ensemble_size != len(json_paths):
            raise ValueError(f"Ensemble size {ensemble_size} doesn't match directory size {len(json_paths)}.")
    else:
        with open(uncertainty_path, 'r', encoding='utf-8') as file:
            uncertainties = json.load(file)

    for model_id in range(ensemble_size):
        if load_from_dir:
            with open(json_paths[model_id], 'r', encoding='utf-8') as file:
                uncertainties = json.load(file)

        model_errors = errors[str(model_id)]
        y_score, y_true = predictions_from_dict_bw(uncertainties, model_errors)
        y_scores.append(y_score)
        y_trues.append(y_true)

    return y_scores, y_trues


def main():
    # for i in range(1, len(sys.argv)):
    #     make_a_curve(sys.argv[i], i)
    # plt.legend()
    # plt.savefig('smth-idk.png')
    # y_score, y_true = predictions_from_dict_pmqa(sys.argv[1], 0)

    ###
    # y_score, y_true = predictions_from_dict_bw(sys.argv[1], sys.argv[2], 0)
    
    # print(y_score.shape, y_true.shape, y_score[:5], y_true[:5])
    # print(roc_auc_score(y_true, y_score))
    # fpr, tpr, _ = roc_curve(y_true, y_score)
    # plt.plot(fpr, tpr)
    # plt.savefig(f'plot_{sys.argv[3]}')

    ###
    # print(bool(int(sys.argv[3])), type(int(sys.argv[3])))
    # y_scores, y_trues = convert_ensemble_predictions_bw(sys.argv[1], sys.argv[2], 10, bool(int(sys.argv[3])))
    y_scores, y_trues = convert_ensemble_predictions_pmqa(sys.argv[1], 10, bool(int(sys.argv[2])))
    
    print(type(y_scores), len(y_scores))

    fpr_grid = np.linspace(0., 1., 1001)
    tprs = []
    scores = []

    for y_true, y_score in zip(y_trues, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        score = roc_auc_score(y_true, y_score)
        scores.append(score)
        # Interpolate TPR at common FPRs
        interp_tpr = np.interp(fpr_grid, fpr, tpr)
        interp_tpr[0] = 0.0  # ensure it starts at (0,0)
        tprs.append(interp_tpr)

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)

    mean_tpr[-1] = 1.0  # ensure it ends at (1,1)

    scores = np.array(scores)
    mean_score, std_score = scores.mean(axis=0), scores.std(axis=0)
    print(f"AUROC: {mean_score} +- {std_score}")

    # Plot
    plt.plot(fpr_grid, mean_tpr, color='b', label=f'Mean ROC, AUC={mean_score:.4f}')
    plt.fill_between(fpr_grid,
                    np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    color='b', alpha=0.2, label=f'std={std_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('Mean ROC Curve with Standard Deviation')
    plt.savefig(f"plot_{sys.argv[3]}")

    curve_data = CurveData(mean_tpr, std_tpr, mean_score, std_score)
    with open(f'curve_{sys.argv[3]}.pkl', 'wb') as pickle_file:
        pickle.dump(curve_data, pickle_file)


if __name__ == '__main__':
    main()
