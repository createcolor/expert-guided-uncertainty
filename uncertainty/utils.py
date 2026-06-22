import json
import pickle
from pathlib import Path
import pandas as pd  # pylint: disable=import-error


def load_json(file_path: Path):
    """
    Loads a .json file from file path.
    """
    with open(file_path, "r", encoding='utf-8') as f:
        markup = json.load(f)
    return markup


def dump_json(file, file_path: Path):
    """
    Dumps a .json to the file from file path.
    """
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(file, f, indent=4)


def load_pickle(file_path):
    """
    Loads a .pkl file from file path.
    """
    with open(str(file_path), 'rb') as f:
        markup = pickle.load(f)

    return markup


def dump_pickle(file, file_path: Path):
    """
    Dumps a serializable object to the .pkl file from file path.
    """
    with open(file_path, "wb") as f:
        pickle.dump(file, f)


def load_df(dataset_path):
    """
    Loads the PubMedQA dataset as pandas DataFrame.
    """
    pubmed_df = pd.read_parquet(dataset_path)
    pubmed_df['contexts'] = pubmed_df['context'].apply(lambda x: ' '.join(x['contexts']))
    pubmed_df['reasoning_pred'] = pubmed_df['context'].apply(lambda x: x['reasoning_required_pred'])
    pubmed_df['free_pred'] = pubmed_df['context'].apply(lambda x: x['reasoning_free_pred'])
    pubmed_df = pubmed_df.drop(columns=['context'])
    return pubmed_df


def mkdirs(*dir_paths):
    """
    Creates directories if they do not exist. This behavior is different from os.makedirs.
    """
    for dir_path in dir_paths:
        if not Path(dir_path).exists():
            Path(dir_path).mkdir()
