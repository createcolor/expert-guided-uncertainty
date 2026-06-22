import json
from pathlib import Path
import numpy as np


def calc_accuracy4bin(x, throwaway_rates, accuracies):
    '''
    Approximates accuracy (the y coordinate at the plot of accuracy-rejection curve)
    at the given point x (which is a throwaway rate).

    TODO: Simplify logic maybe?
    '''
    if x in throwaway_rates:
        accuracy = accuracies[throwaway_rates.index(x)]
    else:
        throwaway_rates_less_x = list(filter(lambda rate: rate < x, throwaway_rates))
        left_closest_rate = min(throwaway_rates_less_x, key=lambda rate: x - rate)
        right_closest_rate = throwaway_rates[throwaway_rates.index(left_closest_rate) + 1]
        alpha = (x - left_closest_rate) / (right_closest_rate - left_closest_rate)

        accuracy1 = accuracies[throwaway_rates.index(left_closest_rate)]
        accuracy2 = accuracies[throwaway_rates.index(right_closest_rate)]
        accuracy = accuracy1 + alpha * (accuracy2 - accuracy1)

    return accuracy


def get_rc_info_averaging_mode(
    uncertainties: dict | list,
    n_bins: int = 20,
    path2save_rc_info: Path = None,
    net_ids=range(10),
    sign_figs_num: int=3
):
    '''
    Given uncertainties and list of classificator mistakes and test markups, obtains info 
    to plot rejection curves. If net_ids contains more than one id, rejection curves 
    for the corresponding classifying networks are averaged along y axis.   

    @param uncertainties: {"std": [[is_misclassified_by_net0: bool, is_misclassified_by_net1, ...], ...]}

    @param net_ids: ids of classifying neural networks; 

    @param n_bins: x axis (throwaway rate) bins for averaging curves. For example, 
    if n_bins is equal to 5, y axis values for curves (accuracies) are calculated 
    at the following x-axis points: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0;

    @param path2save_rc_info: where to save rejection curves info accuracies 
    for the corresponding throwaway rates.
    '''
    x_list = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ensemble_accuracies_dict = {x: [] for x in x_list}

    for net_id in net_ids:
        if type(uncertainties) == list:
            accuracies, throwaway_rates = get_accs_throwaway_rates(uncertainties[net_id], net_id, sign_figs_num)
        else:
            accuracies, throwaway_rates = get_accs_throwaway_rates(uncertainties, net_id, sign_figs_num)

        for x in x_list:
            accuracy = calc_accuracy4bin(x, throwaway_rates, accuracies)
            ensemble_accuracies_dict[x].append(accuracy)
    
    ensemble_accuracies = {str(round(x, 6)): (np.mean(ensemble_accuracies_dict[x]), \
                            np.std(ensemble_accuracies_dict[x])) for x in x_list}

    if path2save_rc_info is not None:
        with open(path2save_rc_info, "w", encoding="utf-8") as f:
            json.dump(ensemble_accuracies, f, indent=4)

    return ensemble_accuracies


def calculate_accuracy(mistakes_num, total_samples_num):
    return 1.0 - mistakes_num / total_samples_num


def get_accs_throwaway_rates(uncertainties_dict: dict, net_id: int, sign_figs_num: int=7):
    '''
    Calculates data for rejection curves --- accuracies relative to throwaway rates.
    Throwaway rate is the part of the test set being thrown away --- the higher the uncertainty,
    the earlier the sample is thrown away. Accuracy is calculated on the rest part of the test set.

    @param uncert_dict: {"uncertainty": [[is_misclassified_by_net0: bool,
        is_misclassified_by_net1, ...], ...]}
    '''
    total_samples_num = 0
    mistakes_num_left = 0

    uncertainties = {}
    for unc, samples_info in uncertainties_dict.items():
        mistakes_num = 0
        correct_num = 0
        for sample_info in samples_info:
            if sample_info[net_id]:
                mistakes_num += 1
            else:
                correct_num += 1
        uncertainties[unc] = (mistakes_num, correct_num)
        total_samples_num += (mistakes_num + correct_num)
        mistakes_num_left += mistakes_num

    initial_acc = calculate_accuracy(mistakes_num_left, total_samples_num)

    accuracies, throwaway_rates = [initial_acc], [0.0]

    sorted_uncertainties = sorted(list(map(float, uncertainties.keys())), reverse=True)

    samples_num_left = total_samples_num

    for uncertainty in sorted_uncertainties:
        uncertainty = f'{uncertainty:.{sign_figs_num}f}'
        mistakes_num_left -= uncertainties[uncertainty][0]
        samples_num_left -= (uncertainties[uncertainty][0] + uncertainties[uncertainty][1])

        if samples_num_left > 0:
            throwaway_rate = 1. - (samples_num_left / total_samples_num)
            throwaway_rates.append(throwaway_rate)

            accuracy = calculate_accuracy(mistakes_num_left, samples_num_left)
            accuracies.append(accuracy)

            if accuracy == 1.0:
                break

    accuracies.append(1.0)
    throwaway_rates.append(1.0)

    return accuracies, throwaway_rates
