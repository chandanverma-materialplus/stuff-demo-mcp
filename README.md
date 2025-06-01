# author: materialplus.io
# date: 6/1/2025

# AI Model Development Pipeline

This repository contains the code for developing an AI model pipeline for multilingual speech-to-text and translation. The pipeline includes data preparation, model training, inference, evaluation, and active learning.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## CLI Examples

- **Data Preparation**: Convert raw video and subtitle pairs into a dataset.
  ```bash
  python data_prep.py --video_dir path/to/videos --subtitle_dir path/to/subtitles
  ```

- **Train ASR Model**: Fine-tune the Whisper model for speech-to-text.
  ```bash
  python train_asr.py --model_name openai/whisper-v3 --dataset_path path/to/dataset
  ```

- **Train Translation Model**: Domain-adapt the SeamlessM4T model for translation.
  ```bash
  python train_translate.py --model_name facebook/seamless-m4t-v2-large --dataset_path path/to/dataset
  ```

- **Inference**: Run end-to-end inference to generate translated subtitles.
  ```bash
  python infer.py --audio_path path/to/audio
  ```

- **Evaluation**: Calculate evaluation metrics and log results to MLflow.
  ```bash
  python evaluate.py --ground_truth_path path/to/ground_truth.txt --predictions_path path/to/predictions.txt
  ```

- **Active Learning**: Ingest corrected SRTs and trigger model retraining.
  ```bash
  python active_learning.py --corrected_srt_dir path/to/corrected_srts
  ```

## MLflow Dashboard

Access the MLflow dashboard to track model performance and drift:
- [MLflow Dashboard](http://localhost:5000)

## Benchmark Targets

- **WER**: Target < 10%
- **BLEU**: Target > 30
- **chrF**: Target > 50
- **Latency**: Target < 500ms
