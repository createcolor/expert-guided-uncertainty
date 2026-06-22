import argparse
from pathlib import Path

from uncertainty.calculate_uncertainty import calculate_uncertainties, get_classifier_mistakes
from tasks.qa.rejection_curves import get_rc_info_averaging_mode
from uncertainty.dash_visualizer import run_dash
from uncertainty.utils import load_json, dump_json

import argparse
from pathlib import Path

import numpy as np
from itertools import chain
from tqdm import tqdm


def get_rc_data(config):
    dir2save = Path(config["path2save_rc_data"])
    if not dir2save.exists():
        dir2save.mkdir()

    for method, info in config["UE_methods"].items():
        path2save_reg_curves_info = f"{dir2save}/{method}.json"
        
        if info['single_net_file_name_template'] is None:
            path2uncerts = f"{config['path2save_uncertainties']}/{method}.json"
            uncertainties = load_json(path2uncerts)
        else:
            uncertainties = []
            for net_id in range(config["nets_num"]):
                path2uncerts = f"{config['path2save_uncertainties']}/{method}/{info['single_net_file_name_template']}_{net_id}.json"
                uncertainties.append(load_json(path2uncerts))
        
        get_rc_info_averaging_mode(uncertainties, n_bins=1000, path2save_rc_info=path2save_reg_curves_info, 
                                   net_ids=range(config["nets_num"]), sign_figs_num=config["sign_figs_num"])


def calculate_aucs(accuracies_dict: dict, throwaway_rates_dict: dict, min_accuracy: float=None, init_acc: float=0.0):

    def normalize_accuracies(accuracies: list[float], min_acc: float):
        return [(accuracy - min_acc) / (1.0 - min_acc) for accuracy in accuracies]
    
    aucs = {}
    if min_accuracy is None:
        min_accuracy = min(list(chain(accuracies_dict.values())))

    for method_name, accuracies in accuracies_dict.items():
        accuracies = normalize_accuracies(accuracies, min_accuracy)
        auc = round(np.trapz(y=accuracies, x=throwaway_rates_dict[method_name]), 6)
        left_acc = 1. - init_acc
        aucs[method_name] = round((1.0 - auc - left_acc * left_acc / 2) * 1000)
       
    return aucs


def get_aucs_values(config, rc_data: list=None, path2save_aucs: Path=None):
    if rc_data is None:
        rej_curves_data_dir = config["path2save_rc_data"]
        accuracies_dict, throwaway_rates_dict = {}, {}

        for method in config["UE_methods"].keys():
            path2rej_curves_data = f"{rej_curves_data_dir}/{method}.json"
            data = load_json(path2rej_curves_data)
        
            throwaway_rates = sorted(list(map(float, list(data.keys()))))
            accuracies_dict[method] = []
            for throwaway_rate in throwaway_rates:
                accuracies_dict[method].append(data[str(throwaway_rate)][0])
                
            throwaway_rates_dict[method] = throwaway_rates

        init_accuracy = load_json(path2rej_curves_data)["0.0"][0]
        aucs = calculate_aucs(accuracies_dict, throwaway_rates_dict, min_accuracy=0.0, init_acc=init_accuracy)
    
    else:
        aucs = {method: [] for method in config["UE_methods"].keys()}
        for i in range(len(list(rc_data.values())[0])):
            for method, data in rc_data.items():
                data = data[i]
                accuracies_dict, throwaway_rates_dict = {}, {}

                throwaway_rates = sorted(list(map(float, list(data.keys()))))
                accuracies_dict[method] = []
                for throwaway_rate in throwaway_rates:
                    accuracies_dict[method].append(data[str(throwaway_rate)][0])
                    
                throwaway_rates_dict[method] = throwaway_rates

                init_accuracy = data["0.0"][0]
                
                aucs_cur = calculate_aucs(accuracies_dict, throwaway_rates_dict, min_accuracy=0.0, init_acc=init_accuracy)
                
                for method, auc in aucs_cur.items():
                    aucs[method].append(auc)
        
        for method, aucs_values in aucs.items():
            print(f"{method}:\nMean: {round(np.mean(aucs_values), 1)} +- {round(np.std(aucs_values), 1)}, median: {np.median(aucs_values)}, " 
                  f"min: {np.min(aucs_values)}, max: {np.max(aucs_values)}\n")

    if path2save_aucs is not None:
        dump_json(aucs, path2save_aucs)
    else:
        print(aucs)
    

def add_noise2unc(config, repeat_num = 100):
    dir2save = Path(config["path2save_rc_data"])
    if not dir2save.exists():
        dir2save.mkdir()
    
    noised_uncertainties = {method: [] for method in config["UE_methods"]}

    for _ in range(repeat_num): 
        for method, info in config["UE_methods"].items():        
            if info['single_net_file_name_template'] is None:
                path2uncerts = f"{config['path2save_uncertainties']}/{method}.json"
                uncertainties_init = load_json(path2uncerts)
                uncertainties_init = {float(unc): v for unc, v in uncertainties_init.items()}
                
                std = np.std(list(uncertainties_init.keys())) / 25
                noised_uncertainties[method].append({str(round(unc + np.random.normal(loc=0, scale=std), 7)): v for unc, v in uncertainties_init.items()})
            else:
                uncertainties = []
                for net_id in range(config["nets_num"]):
                    path2uncerts = f"{config['path2save_uncertainties']}/{method}/{info['single_net_file_name_template']}_{net_id}.json"
                    uncertainties_init = load_json(path2uncerts)
                    uncertainties_init = {float(unc): v for unc, v in uncertainties_init.items()}
                    
                    std = np.std(list(uncertainties_init.keys())) / 25
                    uncertainties_cur = {str(round(unc + np.random.normal(loc=0, scale=std), 7)): v for unc, v in uncertainties_init.items()}
                    uncertainties.append(uncertainties_cur)
                noised_uncertainties[method].append(uncertainties)
        
    return noised_uncertainties


def calculate_aucs(accuracies_dict: dict, throwaway_rates_dict: dict, min_accuracy: float=None, init_acc: float=0.0):

    def normalize_accuracies(accuracies: list[float], min_acc: float):
        return [(accuracy - min_acc) / (1.0 - min_acc) for accuracy in accuracies]
    
    aucs = {}
    if min_accuracy is None:
        min_accuracy = min(list(chain(accuracies_dict.values())))

    for method_name, accuracies in accuracies_dict.items():
        accuracies = normalize_accuracies(accuracies, min_accuracy)
        auc = round(np.trapz(y=accuracies, x=throwaway_rates_dict[method_name]), 6)
        left_acc = 1. - init_acc
        aucs[method_name] = round((1.0 - auc - left_acc * left_acc / 2) * 1000)
       
    return aucs


def get_aucs_values(config, rc_data: list=None, path2save_aucs: Path=None):
    if rc_data is None:
        rej_curves_data_dir = config["path2save_rc_data"]
        accuracies_dict, throwaway_rates_dict = {}, {}

        for method in config["UE_methods"].keys():
            path2rej_curves_data = f"{rej_curves_data_dir}/{method}.json"
            data = load_json(path2rej_curves_data)
        
            throwaway_rates = sorted(list(map(float, list(data.keys()))))
            accuracies_dict[method] = []
            for throwaway_rate in throwaway_rates:
                accuracies_dict[method].append(data[str(throwaway_rate)][0])
                
            throwaway_rates_dict[method] = throwaway_rates

        init_accuracy = load_json(path2rej_curves_data)["0.0"][0]
        aucs = calculate_aucs(accuracies_dict, throwaway_rates_dict, min_accuracy=0.0, init_acc=init_accuracy)
    
    else:
        aucs = {method: [] for method in config["UE_methods"].keys()}
        for i in range(len(list(rc_data.values())[0])):
            for method, data in rc_data.items():
                data = data[i]
                accuracies_dict, throwaway_rates_dict = {}, {}

                throwaway_rates = sorted(list(map(float, list(data.keys()))))
                accuracies_dict[method] = []
                for throwaway_rate in throwaway_rates:
                    accuracies_dict[method].append(data[str(throwaway_rate)][0])
                    
                throwaway_rates_dict[method] = throwaway_rates

                init_accuracy = data["0.0"][0]
                
                aucs_cur = calculate_aucs(accuracies_dict, throwaway_rates_dict, min_accuracy=0.0, init_acc=init_accuracy)
                
                for method, auc in aucs_cur.items():
                    aucs[method].append(auc)
        
        for method, aucs_values in aucs.items():
            print(f"{method}:\nMean: {round(np.mean(aucs_values), 1)} +- {round(np.std(aucs_values), 1)}, median: {np.median(aucs_values)}, " 
                  f"min: {np.min(aucs_values)}, max: {np.max(aucs_values)}\n")

    if path2save_aucs is not None:
        dump_json(aucs, path2save_aucs)
    else:
        print(aucs)
    

def add_noise2unc(config, repeat_num = 100):
    dir2save = Path(config["path2save_rc_data"])
    if not dir2save.exists():
        dir2save.mkdir()
    
    noised_uncertainties = {method: [] for method in config["UE_methods"]}

    for _ in range(repeat_num): 
        for method, info in config["UE_methods"].items():        
            if info['single_net_file_name_template'] is None:
                path2uncerts = f"{config['path2save_uncertainties']}/{method}.json"
                uncertainties_init = load_json(path2uncerts)
                uncertainties_init = {float(unc): v for unc, v in uncertainties_init.items()}
                
                std = np.std(list(uncertainties_init.keys())) / 25
                noised_uncertainties[method].append({str(round(unc + np.random.normal(loc=0, scale=std), 7)): v for unc, v in uncertainties_init.items()})
            else:
                uncertainties = []
                for net_id in range(config["nets_num"]):
                    path2uncerts = f"{config['path2save_uncertainties']}/{method}/{info['single_net_file_name_template']}_{net_id}.json"
                    uncertainties_init = load_json(path2uncerts)
                    uncertainties_init = {float(unc): v for unc, v in uncertainties_init.items()}
                    
                    std = np.std(list(uncertainties_init.keys())) / 25
                    uncertainties_cur = {str(round(unc + np.random.normal(loc=0, scale=std), 7)): v for unc, v in uncertainties_init.items()}
                    uncertainties.append(uncertainties_cur)
                noised_uncertainties[method].append(uncertainties)
        
    return noised_uncertainties


def check_methods_stability(config):
    mistakes = get_classifier_mistakes(config)
    calculate_uncertainties(config, mistakes)
    noised_uncertainties = add_noise2unc(config)
    
    rc_data = {method: [] for method in noised_uncertainties}
    for method, uncertainties in tqdm(noised_uncertainties.items()):
        for cur_uncertainties in uncertainties:
            rc_data[method].append(get_rc_info_averaging_mode(cur_uncertainties, n_bins=1000, net_ids=range(config["nets_num"])))
    
    get_aucs_values(config, rc_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="uncertainty_estimation/config_UE.json", type=Path)

    args = parser.parse_args()

    config = load_json(args.config)

    return config


def main():
    config = parse_args()
    mistakes = get_classifier_mistakes(config)
    calculate_uncertainties(config, mistakes)
    get_rc_data(config)
    get_aucs_values(config)
    run_dash(config)


if __name__ == "__main__":
    main()
