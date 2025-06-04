# author: materialplus.io
# Date: 6/4/2025

from typing import Dict
import mlflow
from jiwer import wer
from sacrebleu import corpus_bleu, corpus_chrf
import time

class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass

class Evaluator:
    """Class for evaluating model performance and logging results."""

    def __init__(self, mlflow_tracking_uri: str):
        """
        Initialize the Evaluator class with MLflow tracking URI.

        Args:
            mlflow_tracking_uri (str): URI for MLflow tracking server.
        """
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    def evaluate(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Evaluate model performance using WER, BLEU, and chrF.

        Args:
            references (List[str]): List of reference texts.
            hypotheses (List[str]): List of hypothesis texts.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        try:
            start_time = time.time()
            wer_score = wer(references, hypotheses)
            bleu_score = corpus_bleu(hypotheses, [references]).score
            chrf_score = corpus_chrf(hypotheses, [references]).score
            latency = time.time() - start_time

            metrics = {
                "WER": wer_score,
                "BLEU": bleu_score,
                "chrF": chrf_score,
                "Latency": latency
            }

            self.log_metrics(metrics)
            return metrics
        except Exception as e:
            raise EvaluationError(f"Failed to evaluate model: {e}")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log evaluation metrics to MLflow.

        Args:
            metrics (Dict[str, float]): Dictionary of evaluation metrics.
        """
        with mlflow.start_run():
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

# Example usage
if __name__ == "__main__":
    evaluator = Evaluator(mlflow_tracking_uri="http://localhost:5000")
    metrics = evaluator.evaluate(references=["This is a test."], hypotheses=["This is a test."])
    print(metrics)
