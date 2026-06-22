from pathlib import Path
import numpy as np
from uncertainty.utils import load_json, dump_json, load_pickle, mkdirs  # pylint: disable=import-error
import argparse
import re
from scipy.stats import rankdata


MATCHING_DICT = {"no": 0.0, "maybe": 0.5, "yes": 1.0}
# MATCHING_DICT = {"no": [1, 0, 0], "maybe": [0, 1, 0], "yes": [0, 0, 1]}

def calc_ensemble_uncs(data: Path | dict[str, dict[str, dict[str, int | list]]], mistakes, path2save_uncerts: Path = None, sign_figs_num: int = 3, mode: str = "std"):
    '''
    @param data: classifiers' outputs in the format: 
        {"net_id": {"pubid": {"gt": 1, "pred": 0, "probas": [0.1, 0.7, 0.2]}, ...}, ...};

    @param path2save_uncerts:
        {"std": [[is_misclassified_by_net0: bool, is_misclassified_by_net1, ...], ...]}.
    '''
    if not isinstance(data, dict):
        data_per_nets = load_pickle(data)
    else:
        data_per_nets = data.copy()

    uncert_dict = {}
    for pubid in mistakes[0].keys():
        # pubid = int(pubid)
        sample_probas = [np.array(net_outputs_dict[pubid]["probas"]) 
                        for net_outputs_dict in data_per_nets.values()]
        sample_probas = np.array(sample_probas)
        
        if mode == "std":
            unc = np.sum(np.std(sample_probas, axis=0))
        elif mode == "p(1-p)":
            # unc = np.sum(np.mean(sample_probas, axis=0) * (1 - np.mean(sample_probas, axis=0)))
            unc = np.sum(np.mean(sample_probas * (1. - sample_probas), axis=0))
        elif mode == "entropy":
            # p = np.mean(sample_probas, axis=0)
            # unc = - np.sum(p * np.log(p))

            p_mean = np.mean(sample_probas, axis=0)
            D = np.sum(sample_probas * np.log(sample_probas / p_mean), axis=1)
            unc = np.mean(D, axis=0)

            # entropy = - np.sum(sample_probas * np.log(sample_probas), axis=1)
            # unc = np.mean(entropy) 
        
        unc = f'{unc:.{sign_figs_num}f}'
        if unc not in uncert_dict:
            uncert_dict[unc] = []

        uncert_dict[unc].append([mistakes[net_id][pubid] for net_id in mistakes.keys()])

    if path2save_uncerts is not None:
        dump_json(uncert_dict, path2save_uncerts)

    return uncert_dict


def count_entropy(sample_probas):
    log_probs = np.zeros_like(sample_probas)  # Initialize to avoid log(0)
    mask = sample_probas > 0  # Only compute log where prob > 0
    log_probs[mask] = sample_probas[mask] * np.log(sample_probas[mask])
    entropy = -np.sum(log_probs, axis=1)
    return entropy 

def calc_mcmc(ensemble_outputs: Path, mistakes, path2save_uncerts: Path=None, sign_figs_num: int=3):
    '''
    @param path2ensemble_outputs: classifiers' outputs in the format:
        {"net_id": {"pubid": {"gt": 1, "pred": 0, "probas": [0.1, 0.7, 0.2]}, ...}, ...};

    @param path2save_uncerts:
        {"std": [[is_misclassified_by_net0: bool, is_misclassified_by_net1, ...], ...]}.
    '''
    path2save_uncerts = f"{path2save_uncerts}/mcmc"
    mkdirs(path2save_uncerts)

    outputs_files_list = Path(ensemble_outputs).glob("*.pkl")
    mcmc_ens = {}

    for outputs_file_path in outputs_files_list:
        ensemble_outputs_per_nets = load_pickle(outputs_file_path)
        
        net_ids = ensemble_outputs_per_nets.keys()

        for net_i in net_ids:
            if net_i not in mcmc_ens:
                mcmc_ens[net_i] = {outputs_file_path.stem: ensemble_outputs_per_nets[net_i]}
            else:
                mcmc_ens[net_i][outputs_file_path.stem] = ensemble_outputs_per_nets[net_i]
            
        for net_i, ens in mcmc_ens.items():
            unc_path = f"{path2save_uncerts}/mcmc_{net_i}.json"

            calc_ensemble_uncs(ens, mistakes, unc_path, sign_figs_num, "std")


def calc_test_aug(path2nets_outputs: Path, mistakes, path2save_uncerts: Path=None, sign_figs_num: int=3):
    path2save_uncerts = f"{path2save_uncerts}/test_augmentations"
    mkdirs(path2save_uncerts)

    # outputs_files_list = Path(path2nets_outputs).glob("*.pkl")

    # for net_id, outputs_file_path in enumerate(outputs_files_list):
    #     cur_net_outputs = load_pickle(outputs_file_path)

    #     unc_path = f"{path2save_uncerts}/test_aug_{net_id}.json"
    #     calc_ensemble_uncs(cur_net_outputs, mistakes, unc_path, sign_figs_num, "std")

    net_outputs = load_pickle(path2nets_outputs)
    for net_id, cur_net_outputs in net_outputs.items():
        unc_path = f"{path2save_uncerts}/test_aug_{net_id}.json"
        calc_ensemble_uncs(cur_net_outputs, mistakes, unc_path, sign_figs_num, "std")


def calc_mc_dropout(path2nets_outputs: Path, mistakes, path2save_uncerts: Path=None, sign_figs_num: int=3):
    path2save_uncerts = f"{path2save_uncerts}/mc_dropout"
    mkdirs(path2save_uncerts)

    net_outputs = load_pickle(path2nets_outputs)
    for net_id, cur_net_outputs in net_outputs.items():
        unc_path = f"{path2save_uncerts}/mc_dropout_{net_id}.json"
        calc_ensemble_uncs(cur_net_outputs, mistakes, unc_path, sign_figs_num, "std")


def calc_ddu(path2uncs: Path, mistakes, path2save_uncerts: Path=None, sign_figs_num: int=3):
    mkdirs(path2save_uncerts)

    net_outputs = load_pickle(path2uncs)
    
    uncert_dict = {}
    for net_id, cur_uncs in net_outputs.items():
        unc_path = f"{path2save_uncerts}/ddu_{net_id}.json"

        for pubid in mistakes[0].keys():
            unc = cur_uncs[pubid]
            unc = f'{unc:.{sign_figs_num}f}'
            if unc not in uncert_dict:
                uncert_dict[unc] = []

            uncert_dict[unc].append([mistakes[net_id][pubid] for net_id in mistakes.keys()])
            
        if path2save_uncerts is not None:
            dump_json(uncert_dict, unc_path)


def calc_huq(path2outputs: Path, path2densities: Path, mistakes, path2save_uncerts: Path=None, sign_figs_num: int=3):
    mkdirs(path2save_uncerts)

    net_outputs = load_pickle(path2outputs)
    net_densities = load_pickle(path2densities)
    
    uncert_dict = {}
    for net_id, cur_outputs in net_outputs.items():
        unc_path = f"{path2save_uncerts}/huq_{net_id}.json"

        entropies = []
        densities = []
        pubids = []

        for pubid in mistakes[0].keys():

            pubids.append(pubid)
            sample_probas = np.array(cur_outputs[pubid]['probas'])
            sample_probas[sample_probas == 0] = 1e-10
            
            entropies.append(abs(-np.sum(sample_probas * np.log(sample_probas))))
            densities.append(net_densities[net_id][pubid])

        entropies = (entropies - np.mean(entropies)) / np.std(entropies)
        densities = (densities - np.mean(densities)) / np.std(densities)
        # entropies = rankdata(entropies)
        # densities = rankdata(densities) 

        for pubid, unc_a, unc_e in zip(pubids, entropies, densities):           
            unc = unc_a + unc_e
        
            unc = f'{unc:.{sign_figs_num}f}'
            if unc not in uncert_dict:
                uncert_dict[unc] = []

            uncert_dict[unc].append([mistakes[net_id][pubid] for net_id in mistakes.keys()])
            
        if path2save_uncerts is not None:
            dump_json(uncert_dict, unc_path)


def calc_one_net(path2nets_outputs: Path, mistakes, path2save_uncerts: Path=None, sign_figs_num: int=3):
    path2save_uncerts = f"{path2save_uncerts}/one_net"
    mkdirs(path2save_uncerts)

    net_outputs = load_pickle(path2nets_outputs)
    for net_id, cur_net_outputs in net_outputs.items():
       
        uncert_dict = {}
        
        for pubid in mistakes[0].keys():
            sample_probas = np.array(cur_net_outputs[pubid]["probas"]) 
            
            unc = 1. - np.max(sample_probas)
            
            unc = f'{unc:.{sign_figs_num}f}'
            if unc not in uncert_dict:
                uncert_dict[unc] = []

            uncert_dict[unc].append([mistakes[net_id][pubid] for net_id in mistakes.keys()])

        if path2save_uncerts is not None:
            dump_json(uncert_dict, f"{path2save_uncerts}/one_net_{net_id}.json")


def calc_gt_experts_uncertainty(test_markup: Path, mistakes, path2save_uncerts: Path=None, sign_figs_num: int=3, mode: str="p(1-p)", expert: str="all"):
    '''
    @param path2save_uncerts: {"std": [[is_misclassified_by_net0: bool, is_misclassified_by_net1, ...], ...]}.
    '''
    test_markup = load_json(test_markup)

    uncert_dict = {}
    for pubid, data in test_markup.items():
        if expert == "all":
            reasoning_label = data['reasoning_required_pred']
            free_label = data['reasoning_free_pred']
            final_label = data['final_decision']
            sample_probas = np.array((MATCHING_DICT[reasoning_label], MATCHING_DICT[free_label], MATCHING_DICT[final_label]))
        elif expert == "rr":
            sample_probas = np.array((MATCHING_DICT[data['reasoning_required_pred']]))
        elif expert == "rf":
            sample_probas = np.array((MATCHING_DICT[data['reasoning_free_pred']]))
        elif expert == "fa":
            sample_probas = np.array((MATCHING_DICT[data['final_decision']]))
        elif expert == "rr+rf":
            reasoning_label = data['reasoning_required_pred']
            free_label = data['reasoning_free_pred']
            sample_probas = np.array((MATCHING_DICT[reasoning_label], MATCHING_DICT[free_label]))
        else:
            raise ValueError(f"Expected `expert` to be one of `all`, `rr`, `rf`, `fa`, `rr+rf`; got {expert}")
       
        if mode == "std":
            unc = np.sum(np.std(sample_probas, axis=0))
        elif mode == "p(1-p)":
            try:
                unc = np.sum(np.mean(sample_probas, axis=0) * (1 - np.mean(sample_probas, axis=0)))
            except np.AxisError:
                unc = np.sum(sample_probas * (1 - sample_probas))
        elif mode == "entropy":
            try:
                p = np.mean(sample_probas, axis=0)
            except np.AxisError:
                p = sample_probas
            if p == 0:
                unc = 0
            else: 
                unc = abs(-np.sum(p * np.log(p)))
        
        unc = f'{unc:.{sign_figs_num}f}'
        if unc not in uncert_dict:
            uncert_dict[unc] = []

        uncert_dict[unc].append([mistakes[net_id][pubid] for net_id in mistakes.keys()])

    if path2save_uncerts is not None:
        dump_json(uncert_dict, path2save_uncerts)

    return uncert_dict


def calc_oracle_uncertainty(path2oracle_outputs: Path, path2test_markup: Path, mistakes, path2oracle_confidences: Path=None,
                            path2save_uncerts: Path=None, sign_figs_num: int=3):
    '''
    @param path2oracle_outputs: classifiers' outputs in the format: 
        {"pubid": confidence_in_answer_percent, ...};

    @param path2save_uncerts: {"std": [[is_misclassified_by_net0: bool, is_misclassified_by_net1, ...], ...]}.
    '''    
    # oracle_answers = load_json(path2oracle_outputs)
    oracle_confidences = load_json(path2oracle_confidences)
    test_markup = load_json(path2test_markup)
    
    uncert_dict = {}
    for pubid in test_markup.keys():
        # answer = np.array([oracle_answers[pubid].count("no"), oracle_answers[pubid].count("maybe"), oracle_answers[pubid].count("yes")]) / len(oracle_answers[pubid])
        # unc = str(round(np.sum(answer * (1 - answer)), 7))

        # difficulties_mode:
        p = 1.0 - np.max(np.mean(oracle_confidences[pubid], axis=0))
        # print(np.mean(oracle_confidences[pubid], axis=0)) 
        unc = p 

        # p = np.mean(oracle_confidences[pubid]) / 100
        # unc = str(round(1.0 - np.mean(oracle_confidences[pubid]) / 100, 7))
        
        # answer = np.mean([MATCHING_DICT[output] for output in oracle_answers[pubid]])
        # unc = str(round(answer * (1 - answer), 7))

        # unc = str(round(np.std([MATCHING_DICT[output] for output in oracle_answers[pubid]]), 7))

        # answer = np.mean([MATCHING_DICT[output] for output in oracle_answers[pubid]])
        # confidence = np.mean(oracle_confidences[pubid]) / 100
        # answer_unc = 0.5 - np.abs(0.5 - answer)
        # confidence_unc = 1.0 - confidence
        # unc = answer_unc * confidence_unc
        # unc = str(round(answer_unc / confidence_unc / 10 + answer_unc * confidence_unc, 7))
        # unc = str(round(answer_unc * confidence_unc * (1. - answer_unc * confidence_unc), 7))
        # unc = str(round(np.abs(2 * answer_unc - confidence_unc), 7))
        
        unc = f'{unc:.{sign_figs_num}f}'
        if unc not in uncert_dict:
            uncert_dict[unc] = []

        uncert_dict[unc].append([mistakes[net_id][pubid] for net_id in mistakes.keys()])

    if path2save_uncerts is not None:
        dump_json(uncert_dict, path2save_uncerts)

    return uncert_dict


def calc_TV_uncs(epist_outputs, aleat_type: str, aleat_outputs: Path | dict, mistakes: dict, 
                 path2save_uncerts: Path = None, aggregation_type: str = "sum", 
                 weighting_coef: float = 0.5, path2oracle_confidences: Path = None, sign_figs_num: int = 3, mode: str="mean"):
    '''
    Combines experts inconsistency (that models aleatoric uncertatinty) and neural ensemble std
    (~epistemic one) to obtain the total uncertainty.

    @param ensemble_outputs: path to the classification nets ensemble predictions or them;

    @param path2save_markup: where to save the aggregated uncertainty;

    @param aggregation_type: how to aggregate aleatoric and epistemic uncertainty:
    "sum" (weighted one), "max" (maximum), "calib";

    @param weighting_coef: if aggregation_type is "sum", it is the weight of the ensemble uncertainty;

    @param aleat_type: "pred_experts", "gt_experts", "oracle"
    '''

    if not isinstance(epist_outputs, dict):
        epist_outputs_per_nets = load_pickle(epist_outputs)
    else:
        epist_outputs_per_nets = epist_outputs.copy()

    if aleat_type == "pred_experts":
        if not isinstance(aleat_outputs, dict):
            aleat_outputs_per_nets = load_pickle(aleat_outputs)
        else:
            aleat_outputs_per_nets = aleat_outputs.copy()
    elif aleat_type == "gt_experts":   
        test_markup = load_json(aleat_outputs)
    elif aleat_type == "oracle":
        oracle_outputs = load_json(aleat_outputs) 
        oracle_confidences = load_json(path2oracle_confidences)
    else:
        raise ValueError(f"Expected `aleat_type` to be on of `pred_experts`, `gt_experts`, `oracle`; got {aleat_type}.")

    uncert_dict = {}
    for pubid in epist_outputs_per_nets[0].keys():
        sample_probas_epist = [np.array(net_outputs_list[pubid]["probas"])
                               for net_outputs_list in epist_outputs_per_nets.values()]
        epist = np.sum(np.var(sample_probas_epist, axis=0))

        if aleat_type == "pred_experts":
            sample_probas_aleat = [np.array(net_outputs_list[pubid]["probas"])
                                   for net_outputs_list in aleat_outputs_per_nets.values()]
            sample_probas_aleat = np.array(sample_probas_aleat)

            aleat = np.sum(np.mean(sample_probas_aleat * (1. - sample_probas_aleat), axis=0))
        
        elif aleat_type == "gt_experts": 
            reasoning_label = test_markup[pubid]['reasoning_required_pred']
            free_label = test_markup[pubid]['reasoning_free_pred']
            # final_decision = test_markup[pubid]['final_decision']
            if mode == "mean":
                # p = np.array((MATCHING_DICT[reasoning_label], MATCHING_DICT[free_label], MATCHING_DICT[final_decision]))
                p = np.array((MATCHING_DICT[reasoning_label], MATCHING_DICT[free_label]))
            elif mode == "std":
                aleat = np.array((MATCHING_DICT[reasoning_label], MATCHING_DICT[free_label])).std(axis=0)
            
            aleat = (p * (1 - p)).mean(axis=0)
        
        elif aleat_type == "oracle":
            # answer = np.array([oracle_outputs[pubid].count("no"), oracle_outputs[pubid].count("maybe"), oracle_outputs[pubid].count("yes")]) / len(oracle_outputs[pubid])
            # aleat = np.sum(answer * (1 - answer))

            # confidence = np.mean(oracle_confidences[pubid])
            p = 1.0 - np.max(np.mean(oracle_confidences[pubid], axis=0))

            # aleat = np.var([MATCHING_DICT[output] for output in oracle_outputs[pubid]])

            # confidence = np.mean([MATCHING_DICT[output] for output in oracle_outputs[pubid]])

            # answer = np.mean([MATCHING_DICT[output] for output in oracle_outputs[pubid]])
            # confidence = np.mean(oracle_confidences[pubid]) / 100
            # answer_unc = 0.5 - np.abs(0.5 - answer)
            # confidence_unc = 1.0 - confidence
            # aleat = (answer_unc / confidence_unc) **2 + answer_unc * confidence_unc

            aleat = p * (1 - p) / 3
            # aleat = (1 - confidence) ** 2 / 4 # the best guy
            # aleat = (1 - p ** 2)/ 4 # the weight 0.9 answer 99 guy
        else:
            raise ValueError(f"Expected `aleat_type` to be on of `pred_experts`, `gt_experts`, `oracle`; got {aleat_type}.")

        if aggregation_type == "max":
            unc = max(epist, aleat)

        elif aggregation_type == "sum":
            unc = weighting_coef * epist + (1 - weighting_coef) * aleat

        elif aggregation_type == "calib":
            unc = epist ** 4 + aleat ** 4

        unc = f'{unc:.{sign_figs_num}f}'
        if unc not in uncert_dict:
            uncert_dict[unc] = []

        uncert_dict[unc].append([mistakes[net_id][pubid] for net_id in mistakes.keys()])

    if path2save_uncerts is not None:
        dump_json(uncert_dict, path2save_uncerts)

    return uncert_dict


def calc_entropies_uncs(epist_outputs, aleat_type: str, aleat_outputs: Path | dict, mistakes: dict, 
                 path2save_uncerts: Path=None, aggregation_type: str="sum", 
                 weighting_coef=0.5, path2oracle_confidences: Path=None, sign_figs_num: int=3):
    '''
    Combines experts inconsistency (that models aleatoric uncertatinty) and neural ensemble std 
    (~epistemic one) to obtain the total uncertainty.

    @param ensemble_outputs: path to the classification nets ensemble predictions or them;

    @param path2save_markup: where to save the aggregated uncertainty;

    @param aggregation_type: how to aggregate aleatoric and epistemic uncertainty: 
    "sum" (weighted one), "max" (maximum), "calib";  

    @param weighting_coef: if aggregation_type is "sum", it is the weight of the ensemble uncertainty;

    @param aleat_type: "pred_experts", "gt_experts"
    '''

    if type(epist_outputs) != dict:
        epist_outputs_per_nets = load_pickle(epist_outputs)
    else:
        epist_outputs_per_nets = epist_outputs.copy()
    
    if aleat_type == "pred_experts":
        if type(aleat_outputs) != dict:
            aleat_outputs_per_nets = load_pickle(aleat_outputs)
        else:
            aleat_outputs_per_nets = aleat_outputs.copy()

    elif aleat_type == "gt_experts":   
        test_markup = load_json(aleat_outputs)

    uncert_dict = {}
    for pubid in epist_outputs_per_nets[0].keys():
        sample_probas_epist = [np.array(net_outputs_list[pubid]["probas"])
                               for net_outputs_list in epist_outputs_per_nets.values()]

        p_mean = np.mean(sample_probas_epist, axis=0)
        D = np.sum(sample_probas_epist * np.log(sample_probas_epist / p_mean), axis=1)
        I = np.mean(D, axis=0)

        if aleat_type == "pred_experts":
            sample_probas_aleat = [np.array(net_outputs_list[pubid]["probas"])
                                   for net_outputs_list in aleat_outputs_per_nets.values()]
    
            entropy = - np.sum(sample_probas_aleat * np.log(sample_probas_aleat), axis=1)
            E = np.mean(entropy) 
        elif aleat_type == "gt_experts": 
            # MATCHING_DICT = {"no": [1, 0, 0], "maybe": [0, 1, 0], "yes": [0, 0, 1]}
            reasoning_label = test_markup[pubid]['reasoning_required_pred']
            free_label = test_markup[pubid]['reasoning_free_pred']
            final_label = test_markup[pubid]['final_decision']
            sample_probas_aleat = np.array((MATCHING_DICT[reasoning_label], MATCHING_DICT[free_label], MATCHING_DICT[final_label]))

            p = np.mean(sample_probas_aleat, axis=0)
            if p == 0:
                E = 0
            else:
                E = - np.sum(p * np.log(p)) 
        else:
            raise ValueError(f"Expected `aleat_type` to be one of `pred_experts`, `gt_experts`; got {aleat_type}.")

        # print(I, E)
        if aggregation_type == "sum":
            unc = (weighting_coef * I + (1 - weighting_coef) * E)

        unc = f'{unc:.{sign_figs_num}f}'
        if unc not in uncert_dict:
            uncert_dict[unc] = []

        uncert_dict[unc].append([mistakes[net_id][pubid] for net_id in mistakes.keys()])

    if path2save_uncerts is not None:
        dump_json(uncert_dict, path2save_uncerts)

    return uncert_dict


def calculate_uncertainties(config: dict, mistakes: dict):
    sign_figs_num = config["sign_figs_num"]
    dir2save = Path(config["path2save_uncertainties"])
    if not dir2save.exists():
        dir2save.mkdir()

    if "huq" in config["UE_methods"]:
        calc_huq(config["path2outputs_classifier"], config["UE_methods"]["huq"]["path2densities"], 
                 mistakes, dir2save / "huq", sign_figs_num)

    if "ddu" in config["UE_methods"]:
        calc_ddu(config["UE_methods"]["ddu"]["path2densities"], mistakes, dir2save / "ddu", sign_figs_num)

    if "one_net" in config["UE_methods"]:
        calc_one_net(config["path2outputs_classifier"], mistakes, dir2save, sign_figs_num)
    
    if "test_augmentations" in config["UE_methods"]:
        calc_test_aug(config["UE_methods"]["test_augmentations"]["path2outputs"], mistakes, dir2save, sign_figs_num)

    if "mc_dropout" in config["UE_methods"]:
        calc_mc_dropout(config["UE_methods"]["mc_dropout"]["path2outputs"], mistakes, dir2save, sign_figs_num)

    if "mixed_loss_std" in config["UE_methods"]:
        calc_ensemble_uncs(config["path2outputs_mixed_loss"], mistakes, dir2save / "mixed_loss_std.json", sign_figs_num, mode="std")
    
    if "pred_experts" in config["UE_methods"]:
        calc_ensemble_uncs(config["path2outputs_aleat"], mistakes, dir2save / "pred_experts.json", sign_figs_num, mode="p(1-p)")

    if "oracle" in config["UE_methods"]:
        calc_oracle_uncertainty(config["path2oracle_outputs"], config["test_markup"], mistakes, config["path2oracle_confidences"], 
                                dir2save / "oracle.json", sign_figs_num)

    for mode in ["p(1-p)", "std", "entropy"]:
        if f"CE_{mode}" in config["UE_methods"]:
            calc_ensemble_uncs(config["path2outputs_epist"], mistakes, dir2save / f"CE_{mode}.json", sign_figs_num, mode)

        for expert in ["all", "rr", "rf", "fa", "rr+rf"]:
            if f"gt_experts_{expert}_{mode}" in config["UE_methods"]:
                calc_gt_experts_uncertainty(config["test_markup"], mistakes, dir2save / f"gt_experts_{expert}_{mode}.json", sign_figs_num, mode, expert)


    if "mcmc" in config["UE_methods"]:
        calc_mcmc(config["UE_methods"]["mcmc"]["path2outputs"], mistakes, dir2save, sign_figs_num)
    
    for aggregation_type in ["sum", "calib", "max"]:
        if f"TV_CE_p(1-p)_{aggregation_type}_CE_var" in config["UE_methods"]:
            calc_TV_uncs(config["path2outputs_epist"], "pred_experts", config["path2outputs_epist"], mistakes, 
            dir2save / f"TV_CE_p(1-p)_{aggregation_type}_CE_var.json", \
            aggregation_type, weighting_coef=0.5, sign_figs_num=sign_figs_num)
        
        if f"TV_pred_exps_p(1-p)_{aggregation_type}_CE_var" in config["UE_methods"]:
            calc_TV_uncs(config["path2outputs_epist"], "pred_experts", config["path2outputs_aleat"], mistakes, 
            dir2save / f"TV_pred_exps_p(1-p)_{aggregation_type}_CE_var.json", \
            aggregation_type, weighting_coef=0.5, sign_figs_num=sign_figs_num)
        
        if f"TV_gt_exps_p(1-p)_{aggregation_type}_CE_var" in config["UE_methods"]:
            calc_TV_uncs(config["path2outputs_epist"], "gt_experts", config["test_markup"], mistakes, 
            dir2save / f"TV_gt_exps_p(1-p)_{aggregation_type}_CE_var.json", \
            aggregation_type, weighting_coef=0.5, sign_figs_num=sign_figs_num)

        if f"TV_mixed_loss_p(1-p)_{aggregation_type}_mixed_loss_var" in config["UE_methods"]:
            calc_TV_uncs(config["path2outputs_mixed_loss"], "pred_experts", config["path2outputs_mixed_loss"], mistakes, 
            dir2save / f"TV_mixed_loss_p(1-p)_{aggregation_type}_mixed_loss_var.json", \
            aggregation_type, weighting_coef=0.5, sign_figs_num=sign_figs_num)
        
        if f"TV_oracle_p(1-p)_{aggregation_type}_CE_var" in config["UE_methods"]:
            calc_TV_uncs(config["path2outputs_epist"], "oracle", config["path2oracle_outputs"], mistakes, 
            dir2save / f"TV_oracle_p(1-p)_{aggregation_type}_CE_var.json", \
            aggregation_type, weighting_coef=0.5, path2oracle_confidences=config["path2oracle_confidences"], sign_figs_num=sign_figs_num)
    
        if f"entropy_CE_{aggregation_type}_CE" in config["UE_methods"]:
            calc_entropies_uncs(config["path2outputs_epist"], "pred_experts", config["path2outputs_epist"], mistakes, 
            dir2save / f"entropy_CE_{aggregation_type}_CE.json", \
            aggregation_type, weighting_coef=0.5, sign_figs_num=sign_figs_num)
        
        if f"entropy_pred_exps_{aggregation_type}_CE" in config["UE_methods"]:
            calc_entropies_uncs(config["path2outputs_epist"], "pred_experts", config["path2outputs_aleat"], mistakes, 
            dir2save / f"entropy_pred_exps_{aggregation_type}_CE.json", \
            aggregation_type, weighting_coef=0.5, sign_figs_num=sign_figs_num)
        
        if f"entropy_gt_exps_{aggregation_type}_CE" in config["UE_methods"]:
            calc_entropies_uncs(config["path2outputs_epist"], "gt_experts", config["test_markup"], mistakes, 
            dir2save / f"entropy_gt_exps_{aggregation_type}_CE.json", \
            aggregation_type, weighting_coef=0.5, sign_figs_num=sign_figs_num)

        # if f"TV_mixed_loss_p(1-p)_{aggregation_type}_mixed_loss_var" in config["UE_methods"]:
        #     calc_TV_uncs(config["path2outputs_mixed_loss"], "pred_experts", config["path2outputs_mixed_loss"], mistakes, 
        #     dir2save / f"TV_mixed_loss_p(1-p)_{aggregation_type}_mixed_loss_var.json", \
        #     aggregation_type, weighting_coef=0.5, sign_figs_num=sign_figs_num)
        
        # if f"TV_oracle_p(1-p)_{aggregation_type}_CE_var" in config["UE_methods"]:
        #     calc_TV_uncs(config["path2outputs_epist"], "oracle", config["path2oracle_outputs"], mistakes, 
        #     dir2save / f"TV_oracle_p(1-p)_{aggregation_type}_CE_var.json", \
        #     aggregation_type, weighting_coef=0.5, path2oracle_confidences=config["path2oracle_confidences"], sign_figs_num=sign_figs_num)


def get_classifier_mistakes(config, dir2save: Path=None, path2outputs: str=None):
    if dir2save is None:
        dir2save = Path(config["dir2save_classifier_mistakes"])

    if not dir2save.exists():
        dir2save.mkdir()

    if path2outputs is None:
        path2outputs = config["path2outputs_classifier"]
    outputs = load_pickle(path2outputs)

    test_markup = load_json(config["test_markup"])
    
    mistakes = {}
    for net_id, net_outputs in outputs.items():
        net_id = int(net_id)
        mistakes[net_id] = {}
        for pubid, info in net_outputs.items():
            if pubid in test_markup:
                mistakes[net_id][pubid] = info["gt"] != info["pred"]

    method_name_borders = [m.start() for m in re.finditer('/', str(path2outputs))]
    method_name = str(path2outputs)[method_name_borders[-2] + 1 : method_name_borders[-1]]  
    path2save_mistakes = f"{dir2save}/{method_name}/{method_name}_mistakes.json"
    dump_json(mistakes, path2save_mistakes)    
    
    return mistakes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="config_UE.json", type=Path)

    args = parser.parse_args()

    config = load_json(args.config)
    
    return config


def main():
    config = parse_args()

    mistakes = get_classifier_mistakes(config)
    calculate_uncertainties(config, mistakes)
    

if __name__ == "__main__":
    main()