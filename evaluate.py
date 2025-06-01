# author: materialplus.io
# date: 6/1/2025

import os
from typing import Dict
import mlflow
from jiwer import wer
from sacrebleu import corpus_bleu, corpus_chrf

class EvaluationError(Exception):
    """Custom error class for evaluation issues."""
    pass

class Evaluation:
    def __init__(self, ground_truth_path: str, predictions_path: str):
        self.ground_truth_path = ground_truth_path
        self.predictions_path = predictions_path

    def evaluate(self) -> Dict:
        """Calculate evaluation metrics and log results to MLflow."""
        try:
            ground_truth = self._load_file(self.ground_truth_path)
            predictions = self._load_file(self.predictions_path)
            metrics = self._calculate_metrics(ground_truth, predictions)
            self._log_to_mlflow(metrics)
            return metrics
        except Exception as e:
            raise EvaluationError(f"Failed to evaluate: {e}")

    def _load_file(self, path: str) -> str:
        """Load text file."""
        with open(path, 'r') as file:
            return file.read()

    def _calculate_metrics(self, ground_truth: str, predictions: str) -> Dict:
        """Calculate WER, BLEU, chrF, and latency."""
        wer_score = wer(ground_truth, predictions)
        bleu_score = corpus_bleu([predictions], [[ground_truth]]).score
        chrf_score = corpus_chrf([predictions], [[ground_truth]]).score
        latency = 0.0  # Placeholder for actual latency calculation
        return {'WER': wer_score, 'BLEU': bleu_score, 'chrF': chrf_score, 'Latency': latency}

    def _log_to_mlflow(self, metrics: Dict) -> None:
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics)

# Example usage
if __name__ == "__main__":
    evaluation = Evaluation(ground_truth_path='path/to/ground_truth.txt', predictions_path='path/to/predictions.txt')
    results = evaluation.evaluate()
    print("Evaluation completed successfully.")