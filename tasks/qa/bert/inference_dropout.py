import sys
import json
import pickle
# from pathlib import Path
from typing import Any, Callable
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import Dataset
import torch
import numpy as np
from tqdm import tqdm

from process_dataset import id2label, label2id
from load_dataset import load_labelled_ds


# Apparently gives better precision on higher-end GPUs
if torch.backends.cuda.matmul.allow_tf32:
    print("TensorFloat32 tensor cores are available and will be used.")
    torch.set_float32_matmul_precision('high')


def generate_probas(config: dict[str, Any]) -> dict[dict[str, float | np.ndarray]]:
    _, _, test_ds = load_labelled_ds()

    predictions = {}
    # embeddings = {}
    for net_id in tqdm(range(10)):
        predictions_per_model = {repeat_id: {} for repeat_id in range(10)}
        model_dir = f"{config['model_name']}/{net_id}"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, classifier_dropout=0.2, from_tf=False).to('cuda')

        # model.eval()
        model.train()

        correct = 0

        with tqdm(test_ds) as pbar:
            for sample in pbar:
                inputs = tokenizer(f"{''.join(sample['context']['contexts'])} {sample['question']}", return_tensors="pt", padding=True, truncation=True).to('cuda')

                for repeat_id in range(10):
                    with torch.no_grad():
                        outputs = model(**inputs).logits

                    prediction = torch.argmax(outputs).item()
                    gt = label2id[sample['final_decision']]
                    predictions_per_model[repeat_id][str(sample['pubid'])] = {
                        "gt": gt,
                        "pred": prediction,
                        "probas": torch.softmax(outputs, dim=-1)[0].cpu().numpy()
                    }
                    correct += int(prediction == gt)
                    pbar.set_postfix({"accuracy": correct / (pbar.n + 1)})

            predictions[net_id] = predictions_per_model

    with open("mc_dropout.pkl", 'wb') as f:
        pickle.dump(predictions, f)

    return predictions


def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    generate_probas(config)


if __name__ == "__main__":
    main(sys.argv[1])
