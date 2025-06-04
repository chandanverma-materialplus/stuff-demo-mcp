# author: materialplus.io
# Date: 6/4/2025

# AI Subtitle Generation System

## Overview

This project implements an AI system for generating subtitles from audio/video content in multiple languages. The system includes components for data preparation, ASR model training, translation model training, inference, evaluation, and active learning.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up MLflow for tracking:
   - Start the MLflow server:
     ```bash
     mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
     ```

## CLI Examples

- Data Preparation:
  ```bash
  python data_prep.py
  ```

- ASR Model Training:
  ```bash
  python train_asr.py
  ```

- Translation Model Training:
  ```bash
  python train_translate.py
  ```

- Inference:
  ```bash
  python infer.py
  ```

- Evaluation:
  ```bash
  python evaluate.py
  ```

- Active Learning:
  ```bash
  python active_learning.py
  ```

## MLflow Dashboard

Access the MLflow dashboard at `http://localhost:5000` to view experiment results and metrics.

## Benchmark Targets

- WER: <target-value>
- BLEU: <target-value>
- chrF: <target-value>
- Latency: <target-value>
