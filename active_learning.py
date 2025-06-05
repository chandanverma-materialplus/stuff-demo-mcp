# author: materialplus.io
# Date: 6/4/2025

from typing import List
import os
import shutil
import logging

class ActiveLearningError(Exception):
    """Custom exception for active learning errors."""
    pass

class ActiveLearning:
    """Class for managing active learning and incremental retraining."""

    def __init__(self, srt_dir: str, model_dir: str):
        """
        Initialize the ActiveLearning class with directories for SRTs and models.

        Args:
            srt_dir (str): Directory containing human-corrected SRT files.
            model_dir (str): Directory containing models for retraining.
        """
        self.srt_dir = srt_dir
        self.model_dir = model_dir

    def ingest_srts(self) -> List[str]:
        """
        Ingest human-corrected SRT files for retraining.

        Returns:
            List[str]: List of ingested SRT file paths.
        """
        logging.basicConfig(level=logging.INFO)

        try:
            ingested_files = []
            for srt_file in os.listdir(self.srt_dir):
                if srt_file.endswith('.srt'):
                    shutil.copy(os.path.join(self.srt_dir, srt_file), self.model_dir)
                    ingested_files.append(srt_file)
            logging.info(f"Ingested files: {ingested_files}")
            return ingested_files
        except FileNotFoundError as e:
            raise ActiveLearningError(f"SRT directory not found: {e}")
        except PermissionError as e:
            raise ActiveLearningError(f"Permission denied: {e}")
        except Exception as e:
            raise ActiveLearningError(f"Failed to ingest SRT files: {e}")

    def trigger_retraining(self) -> None:
        """
        Trigger incremental retraining of ASR and MT models.
        """
        try:
            # Placeholder for actual retraining implementation
            # This should include loading new data and updating models
            logging.info("Retraining triggered.")
        except Exception as e:
            raise ActiveLearningError(f"Failed to trigger retraining: {e}")

# Example usage
if __name__ == "__main__":
    active_learning = ActiveLearning(srt_dir="path/to/srts", model_dir="path/to/models")
    ingested_files = active_learning.ingest_srts()
    print(f"Ingested files: {ingested_files}")
    active_learning.trigger_retraining()
