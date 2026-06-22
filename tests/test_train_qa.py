"""
Smoke tests for QA (BERT) training.

The HuggingFace Trainer and model/tokenizer are mocked — no model download required.
Run with: pytest tests/test_train_qa.py -v
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from tasks.qa.bert.train_single import train_on_PubMedQA


def _make_minimal_hf_dataset():
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict({
        "pubid":            ["1", "2", "3", "4"],
        "QUESTION":         ["Q1", "Q2", "Q3", "Q4"],
        "CONTEXTS":         [["C1"], ["C2"], ["C3"], ["C4"]],
        "final_decision":   ["yes", "no", "maybe", "yes"],
    })


def _write_minimal_config(path):
    config = {
        "model_name": "bert-base-uncased",
        "tokenizer_params": {},
        "model_params": {},
        "train_params": {"num_train_epochs": 1, "seed": 42},
    }
    path.write_text(json.dumps(config))


@pytest.fixture()
def mocked_hf(tmp_path):
    """Patches AutoTokenizer, AutoModel, and Trainer; yields (config_path, save_path, mocks)."""
    with patch("tasks.qa.bert.train_single.AutoTokenizer.from_pretrained") as MockTok, \
         patch("tasks.qa.bert.train_single.AutoModelForSequenceClassification.from_pretrained") as MockMod, \
         patch("tasks.qa.bert.train_single.Trainer") as MockTrainer:

        MockTok.return_value = MagicMock()
        MockMod.return_value = MagicMock()
        mock_trainer_instance = MagicMock()
        MockTrainer.return_value = mock_trainer_instance

        config_path = tmp_path / "config.json"
        _write_minimal_config(config_path)

        yield config_path, tmp_path / "model", mock_trainer_instance, MockTrainer


class TestBERTFinetuning:
    def test_train_calls_trainer_train(self, mocked_hf):
        config_path, save_path, mock_trainer, _ = mocked_hf
        ds = _make_minimal_hf_dataset()

        train_on_PubMedQA(
            train_ds=ds,
            test_ds=ds,
            prepare_data=lambda d, t: d,
            config_path=config_path,
            save_path=save_path,
        )

        mock_trainer.train.assert_called_once()

    def test_train_saves_model(self, mocked_hf):
        config_path, save_path, mock_trainer, _ = mocked_hf
        ds = _make_minimal_hf_dataset()

        train_on_PubMedQA(
            train_ds=ds,
            test_ds=ds,
            prepare_data=lambda d, t: d,
            config_path=config_path,
            save_path=save_path,
        )

        mock_trainer.save_model.assert_called_once()

    def test_correct_output_dir_passed(self, mocked_hf):
        config_path, save_path, _, MockTrainer = mocked_hf
        ds = _make_minimal_hf_dataset()

        train_on_PubMedQA(
            train_ds=ds,
            test_ds=ds,
            prepare_data=lambda d, t: d,
            config_path=config_path,
            save_path=save_path,
        )

        training_args = MockTrainer.call_args.kwargs["args"]
        assert str(training_args.output_dir) == str(save_path)

    def test_model_name_override_respected(self, tmp_path):
        with patch("tasks.qa.bert.train_single.AutoTokenizer.from_pretrained") as MockTok, \
             patch("tasks.qa.bert.train_single.AutoModelForSequenceClassification.from_pretrained") as MockMod, \
             patch("tasks.qa.bert.train_single.Trainer") as MockTrainer:

            MockTok.return_value = MagicMock()
            MockMod.return_value = MagicMock()
            MockTrainer.return_value = MagicMock()

            config_path = tmp_path / "config.json"
            _write_minimal_config(config_path)

            ds = _make_minimal_hf_dataset()
            override = "distilbert-base-uncased"

            train_on_PubMedQA(
                train_ds=ds,
                test_ds=ds,
                prepare_data=lambda d, t: d,
                config_path=config_path,
                save_path=tmp_path / "model",
                model_name_override=override,
            )

            # The override name should be used for both tokenizer and model
            assert MockTok.call_args.args[0] == override
            assert MockMod.call_args.args[0] == override
