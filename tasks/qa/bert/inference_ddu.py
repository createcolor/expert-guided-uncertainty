import sys
import json
import pickle
# from pathlib import Path
from typing import Any, Callable
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import Dataset
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from load_dataset import load_labelled_ds

# Apparently gives better precision on higher-end GPUs
if torch.backends.cuda.matmul.allow_tf32:
    print("TensorFloat32 tensor cores are available and will be used.")
    torch.set_float32_matmul_precision('high')


def get_modernbert_embeddings(model, tokenizer, train_dataset):
    all_embeddings = []
    pubids = []
    with tqdm(train_dataset) as pbar:

        for sample in pbar:
            inputs = tokenizer(f"{''.join(sample['context']['contexts'])} {sample['question']}", return_tensors="pt", padding=True, truncation=True).to('cuda')
            pubids.append(str(sample['pubid']))

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

                hidden_states = outputs.hidden_states
                # [CLS] token embedding from second-to-last layer embeddingss
                embeddings = hidden_states[-1][:, 0, :]
                # L2 normalization
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings), pubids


def choose_n_components(train_embeddings):

    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)


    param_grid = {
        "n_components": range(1, 4),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    grid_search = GridSearchCV(GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score)
    grid_search.fit(train_embeddings)

    df = pd.DataFrame(grid_search.cv_results_)[["param_n_components", "param_covariance_type", "mean_test_score"]]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )
    print(df.sort_values(by="BIC score"))


def fit_pca(features):
    pca = PCA(n_components=100)
        
    features_mean_before = np.mean(features, axis=0)
    features_std_before = np.std(features, axis=0)

    features = (features - features_mean_before) / features_std_before
    pca.fit(features)
    # print(pca.explained_variance_ratio_)

    return pca
    

def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    uncs = {}

    for net_id in tqdm(range(10)):
        train_dataset, _, test_dataset = load_labelled_ds(net_id=net_id)

        model_dir = f"{config['model_name']}/{net_id}"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, from_tf=False).to('cuda')
        
        model.eval()

        train_embeddings, _ = get_modernbert_embeddings(model, tokenizer, train_dataset)
        test_embeddings, pubids = get_modernbert_embeddings(model, tokenizer, test_dataset)

        # pca = fit_pca(train_embeddings)
        # train_embeddings = pca.transform(train_embeddings)

        # test_embeddings = (test_embeddings - np.mean(test_embeddings, axis=0)) / np.std(test_embeddings, axis=0)
        # test_embeddings = pca.transform(test_embeddings)
    
        gmm = GaussianMixture(n_components=1, covariance_type="diag", random_state=0)
        gmm.fit(train_embeddings)
        log_density = gmm.score_samples(test_embeddings)
        
        uncs[net_id] = {pubid: -log_density for pubid, log_density in zip(pubids, log_density)}

        
    with open(f"ddu.pkl", 'wb') as f:
        pickle.dump(uncs, f)


if __name__ == "__main__":
    main(sys.argv[1])
