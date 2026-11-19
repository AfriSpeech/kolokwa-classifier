#!/usr/bin/env python3
"""Fine-tune facebook/wav2vec2-xls-r-300m for binary classification:
Class 0: Formal Liberian English
Class 1: Pure Kolokwa
"""
import os
import sys
import torch
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Union
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

set_seed(42)

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
DATA_DIR = Path("/mnt/volume_d2wey28/data/kolokwa_classifier/processed")
OUTPUT_DIR = Path("/mnt/volume_d2wey28/projects/kolokwa-classifier/checkpoints")
FINAL_MODEL_DIR = Path("/mnt/volume_d2wey28/projects/kolokwa-classifier/kolokwa_classifier_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
MAX_DURATION_SEC = 6.0
MAX_SAMPLES = int(TARGET_SR * MAX_DURATION_SEC)


class AudioClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, feature_extractor, max_samples=MAX_SAMPLES):
        self.df = pd.read_csv(csv_file)
        self.feature_extractor = feature_extractor
        self.max_samples = max_samples

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["path"]
        label = int(row["label"])

        speech, sr = sf.read(audio_path)
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)

        # Pad or truncate
        if len(speech) > self.max_samples:
            speech = speech[:self.max_samples]
        elif len(speech) < self.max_samples:
            speech = np.pad(speech, (0, self.max_samples - len(speech)), mode="constant")

        inputs = self.feature_extractor(
            speech,
            sampling_rate=TARGET_SR,
            max_length=self.max_samples,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc
    }


def main():
    print("=" * 65)
    print(f"Loading feature extractor and model: {MODEL_NAME}")
    print("=" * 65)

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "formal_liberian_english", 1: "pure_kolokwa"},
        label2id={"formal_liberian_english": 0, "pure_kolokwa": 1},
        classifier_proj_size=256,
    )

    # Freeze convolutional feature encoder
    model.freeze_feature_encoder()
    print("✓ Convolutional feature encoder frozen. Fine-tuning Transformer + Classifier Head.")

    train_dataset = AudioClassificationDataset(DATA_DIR / "train.csv", feature_extractor)
    val_dataset = AudioClassificationDataset(DATA_DIR / "val.csv", feature_extractor)

    print(f"Loaded datasets: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        warmup_steps=168,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        bf16=True,
        
        dataloader_num_workers=4,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training on Hopper H200 GPU...")
    train_result = trainer.train()

    print("\nEvaluating best model on unseen validation set...")
    eval_result = trainer.evaluate()
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS ON UNSEEN VALIDATION SET:")
    for k, v in eval_result.items():
        if isinstance(v, float):
            print(f"  • {k:20s}: {v:.4f}")
    print("=" * 60)

    print(f"\nSaving best model to {FINAL_MODEL_DIR} ...")
    model.save_pretrained(FINAL_MODEL_DIR)
    feature_extractor.save_pretrained(FINAL_MODEL_DIR)
    print("✓ Model and feature extractor saved successfully!")


if __name__ == "__main__":
    main()
