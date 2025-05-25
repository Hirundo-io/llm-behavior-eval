import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.data.data_collator import default_data_collator

from .bbq_dataset import BBQDataset
from .dataset_config import DatasetConfig
from .eval_config import EvaluationConfig
from .util_functions import (
    load_model_and_tokenizer,
)


def custom_collator(batch):
    return {
        key: torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item[key]) for item in batch], batch_first=True
        )
        for key in batch[0]
    }


class BaseEvaluator(ABC):
    def __init__(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        """
        Initialize the BaseEvaluator with evaluation and dataset configurations.

        Loads the pretrained and compared models along with the tokenizer. Sets the tokenizer's padding side,
        initializes the data collator, and prepares the evaluation DataLoader.

        Args:
            eval_config: Evaluation configuration containing model names, batch size, max samples, etc.
            dataset_config: Configuration for the dataset to be evaluated.
        """
        self.eval_config = eval_config
        self.dataset_config = dataset_config
        self.models_tokenizers_pairs = {}
        self.tokenizer, self.model = load_model_and_tokenizer(
            eval_config.model_path_or_repo_id, eval_config.use_4bit
        )
        self.tokenizer.padding_side = "left"
        self.data_collator = default_data_collator
        self.prepare_dataloader()

    def prepare_dataloader(self) -> None:
        """
        Prepare the evaluation DataLoader.

        Uses the DatasetFactory to load and preprocess the dataset. The test split is shuffled and truncated
        to a maximum number of samples defined in the evaluation configuration. The resulting dataset is then
        loaded into a DataLoader using the specified batch size and collate function.
        """
        bbq_dataset = BBQDataset(
            self.dataset_config.file_path, self.dataset_config.dataset_type
        )
        test_dataset = bbq_dataset.preprocess(
            self.tokenizer,
            self.dataset_config.text_format,
            self.dataset_config.preprocess_config,
        )
        self.num_samples = (
            min(len(test_dataset), self.eval_config.max_samples)
            if self.eval_config.max_samples
            else len(test_dataset)
        )
        self.eval_dataset = test_dataset.select(range(self.num_samples))
        self.eval_loader = DataLoader(
            cast("Dataset", self.eval_dataset),
            batch_size=self.eval_config.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    @abstractmethod
    def evaluate(self) -> None:
        """
        Run the evaluation process.

        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement evaluate().")

    def save_results(
        self,
        responses: list[dict],
        accuracy: float,
        stereotyped_bias: float,
        empty_responses: int,
    ) -> None:
        """
        Save the evaluation results to files.

        Args:
            responses: The raw responses from the evaluation.
            accuracy: A score representing the confidence bias.
            stereotyped_bias: A score representing the stereotyped bias.
            empty_responses: A count of empty response.
        """
        # Ensure results directory exists.
        Path(self.eval_config.results_dir).mkdir(parents=True, exist_ok=True)

        output_name = (
            f"{self.eval_config.results_dir}/"
            f"{self.eval_config.model_path_or_repo_id.split('/')[-1]}_"
            f"{self.dataset_config.file_path.split('/')[-1]}_"
            f"{self.dataset_config.dataset_type}_"
            f"{self.dataset_config.text_format}"
        )
        output_responses = f"{output_name}.json"
        output_metrics = f"{output_name}.csv"
        results = pd.DataFrame(
            {
                "Accuracy": [accuracy],
                "Stereotype Bias": [stereotyped_bias],
                "Empty Responses": [
                    empty_responses,
                ],
            }
        )
        logging.info(results)
        results.to_csv(output_metrics, index=False)
        with open(output_responses, "w") as f:
            json.dump(responses, f, indent=4)
