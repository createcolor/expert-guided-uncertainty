import unsloth
import os
import sys
import json
from pathlib import Path
import pickle
import pandas as pd
from transformers import set_seed
from tqdm import tqdm

from dataset import prepare_finetune_data, chat_template_inference  # pylint: disable=import-error


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def extract_embeds(config_path: Path, save_path: Path):
    """
    Loads the model, the data and runs the training process.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)

    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(**config['load_params'])

    data = pd.read_parquet("hf://datasets/qiaojin/PubMedQA/"
                           "pqa_labeled/train-00000-of-00001.parquet")
    data['contexts'] = data['context'].apply(lambda x: ' '.join(x['contexts']))
    dataset, _ = prepare_finetune_data(data, tokenizer, test_size=None)

    tokenizer = unsloth.chat_templates.get_chat_template(tokenizer, chat_template='qwen-2.5')
    unsloth.FastLanguageModel.for_inference(model)

    set_seed(config['seed'])
    embeds = {}
    for sample in tqdm(dataset, desc="Generating embeddings"):
        inputs = chat_template_inference(sample, tokenizer).to('cuda')
        outputs = model.generate(input_ids=inputs, min_new_tokens=1, max_new_tokens=1,
                                 use_cache=True, output_hidden_states=True,
                                 return_dict_in_generate=True, **config['extract_params'])
        embedding = outputs.hidden_states[0][-1][0, -1]
        embeds[sample['pubid']] = embedding.cpu()

    with open(save_path, 'wb') as file:
        pickle.dump(embeds, file)


if __name__ == "__main__":
    extract_embeds(Path(sys.argv[1]), Path(sys.argv[2]))
