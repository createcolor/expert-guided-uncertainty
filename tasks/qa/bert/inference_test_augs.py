import sys
import json
import pickle
# from pathlib import Path
from typing import Any, Callable
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
    with open('synonyms.json', 'r') as file:
        synonymous_questions = json.load(file)

    predictions = {}
    # embeddings = {}
    for net_id in tqdm(range(10)):
        predictions_per_model = {repeat_id: {} for repeat_id in range(10)}
        model_dir = f"{config['model_name']}/{net_id}"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, from_tf=False).to('cuda')

        model.eval()

        correct = 0

        with tqdm(test_ds) as pbar:
            for sample in pbar:
                pubid = str(sample['pubid'])
                gt = label2id[sample['final_decision']]

                for repeat_id, question in enumerate(synonymous_questions[pubid]):

                    inputs = tokenizer(f"{''.join(sample['context']['contexts'])} {question}", return_tensors="pt", padding=True, truncation=True).to('cuda')

                    with torch.no_grad():
                        outputs = model(**inputs).logits

                    prediction = torch.argmax(outputs).item()
                    predictions_per_model[repeat_id][pubid] = {
                        "gt": gt,
                        "pred": prediction,
                        "probas": torch.softmax(outputs, dim=-1)[0].cpu().numpy()
                    }
                    correct += int(prediction == gt)
                    pbar.set_postfix({"accuracy": correct / (pbar.n + 1)})

            predictions[net_id] = predictions_per_model

    with open("test_augs.pkl", 'wb') as f:
        pickle.dump(predictions, f)

    return predictions


def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    generate_probas(config)


if __name__ == "__main__":
    main(sys.argv[1])
