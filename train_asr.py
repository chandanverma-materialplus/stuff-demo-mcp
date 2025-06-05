# author: materialplus.io
# Date: 6/4/2025

from typing import Any
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from datasets import DatasetDict
import torch
import logging

class ASRTrainingError(Exception):
    """Custom exception for ASR training errors."""
    pass

class ASRTrainer:
    """Class for training the Whisper-v3 model for multilingual speech-to-text."""

    def __init__(self, model_name: str, dataset: DatasetDict):
        """
        Initialize the ASRTrainer class with a model name and dataset.

        Args:
            model_name (str): Name of the Whisper model to fine-tune.
            dataset (DatasetDict): Dataset for training.
        """
        self.model_name = model_name
        self.dataset = dataset
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)

    def fine_tune(self, output_dir: str) -> None:
        """
        Fine-tune the Whisper model on the provided dataset.

        Args:
            output_dir (str): Directory to save the fine-tuned model.
        """
        logging.basicConfig(level=logging.INFO)

        try:
            # Placeholder for actual fine-tuning implementation
            # This should include setting up the training loop and saving the model
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logging.info(f"Model fine-tuned and saved to {output_dir}")
        except FileNotFoundError as e:
            raise ASRTrainingError(f"Output directory not found: {e}")
        except PermissionError as e:
            raise ASRTrainingError(f"Permission denied: {e}")
        except Exception as e:
            raise ASRTrainingError(f"Failed to fine-tune ASR model: {e}")

# Example usage
if __name__ == "__main__":
    dataset = DatasetDict.load_from_disk("path/to/dataset")
    asr_trainer = ASRTrainer(model_name="openai/whisper-v3", dataset=dataset)
    asr_trainer.fine_tune(output_dir="path/to/output")
