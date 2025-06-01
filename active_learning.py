# author: materialplus.io
# date: 6/1/2025

import os
from typing import List
from datetime import datetime

class ActiveLearningError(Exception):
    """Custom error class for active learning issues."""
    pass

class ActiveLearning:
    def __init__(self, corrected_srt_dir: str, model_update_callback):
        self.corrected_srt_dir = corrected_srt_dir
        self.model_update_callback = model_update_callback

    def ingest_and_retrain(self) -> None:
        """Ingest corrected SRTs and trigger model retraining."""
        try:
            corrected_srts = self._get_corrected_srts()
            self._update_models(corrected_srts)
        except Exception as e:
            raise ActiveLearningError(f"Failed to perform active learning: {e}")

    def _get_corrected_srts(self) -> List[str]:
        """Get list of corrected SRT files."""
        return [os.path.join(self.corrected_srt_dir, f) for f in os.listdir(self.corrected_srt_dir) if f.endswith('.srt')]

    def _update_models(self, corrected_srts: List[str]) -> None:
        """Update models with corrected SRTs."""
        # Placeholder for actual model update logic
        self.model_update_callback(corrected_srts)

# Example usage
def model_update_callback(corrected_srts: List[str]):
    print(f"Updating models with {len(corrected_srts)} corrected SRTs.")

if __name__ == "__main__":
    active_learning = ActiveLearning(corrected_srt_dir='path/to/corrected_srts', model_update_callback=model_update_callback)
    active_learning.ingest_and_retrain()
    print("Active learning process completed successfully.")