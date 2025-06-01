# author: materialplus.io
# date: 6/1/2025

import os
from typing import List, Dict
from datasets import Dataset, DatasetDict

class DataPrepError(Exception):
    """Custom error class for data preparation issues."""
    pass

class DataPrep:
    def __init__(self, video_dir: str, subtitle_dir: str):
        self.video_dir = video_dir
        self.subtitle_dir = subtitle_dir

    def load_data(self) -> DatasetDict:
        """Load video and subtitle files, convert to dataset."""
        try:
            video_files = self._get_files(self.video_dir, '.mp4')
            subtitle_files = self._get_files(self.subtitle_dir, '.srt')
            data = self._create_dataset(video_files, subtitle_files)
            return data
        except Exception as e:
            raise DataPrepError(f"Failed to load data: {e}")

    def _get_files(self, directory: str, extension: str) -> List[str]:
        """Get list of files with the given extension."""
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]

    def _create_dataset(self, video_files: List[str], subtitle_files: List[str]) -> DatasetDict:
        """Create a Hugging Face Dataset from video and subtitle files."""
        # Placeholder for actual dataset creation logic
        data = {'video': video_files, 'subtitle': subtitle_files}
        return DatasetDict({'train': Dataset.from_dict(data)})

# Example usage
if __name__ == "__main__":
    data_prep = DataPrep(video_dir='path/to/videos', subtitle_dir='path/to/subtitles')
    dataset = data_prep.load_data()
    print("Dataset loaded successfully.")