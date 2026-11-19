#!/usr/bin/env python3
"""Inference module for Kolokwa vs Formal Liberian English classifier.
Includes single clip scoring and pairwise audio comparator:
`pick_between_two_audios(audio_a, audio_b)`
"""
import sys
import torch
import soundfile as sf
import numpy as np
import subprocess
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

DEFAULT_MODEL = "AfriSpeech/kolokwa-classifier"
LOCAL_MODEL_DIR = Path("/mnt/volume_d2wey28/projects/kolokwa-classifier/kolokwa_classifier_model")
MODEL_PATH = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else DEFAULT_MODEL
TARGET_SR = 16000
MAX_DURATION_SEC = 6.0
MAX_SAMPLES = int(TARGET_SR * MAX_DURATION_SEC)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = None
model = None


def load_model(model_name_or_path: str = None):
    global feature_extractor, model
    target = model_name_or_path or MODEL_PATH
    if model is None:
        print(f"Loading Kolokwa classifier from {target} ...", file=sys.stderr)
        feature_extractor = AutoFeatureExtractor.from_pretrained(target)
        model = AutoModelForAudioClassification.from_pretrained(target).to(device)
        model.eval()


def load_and_preprocess_audio(audio_path: str) -> np.ndarray:
    """Load audio with ffmpeg, resample to 16kHz mono, and pad/truncate."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(audio_path),
        "-ac", "1", "-ar", str(TARGET_SR), "-f", "s16le", "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(f"Failed to load audio from {audio_path}: {proc.stderr.decode()}")

    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    if len(pcm) > MAX_SAMPLES:
        # Take center or start
        pcm = pcm[:MAX_SAMPLES]
    elif len(pcm) < MAX_SAMPLES:
        pcm = np.pad(pcm, (0, MAX_SAMPLES - len(pcm)), mode="constant")
    return pcm


def score_audio(audio_path: str) -> dict:
    """Score a single audio clip: returns probabilities for Formal vs Kolokwa."""
    load_model()
    pcm = load_and_preprocess_audio(audio_path)
    inputs = feature_extractor(
        pcm,
        sampling_rate=TARGET_SR,
        max_length=MAX_SAMPLES,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    prob_formal = float(probs[0])
    prob_kolokwa = float(probs[1])
    label = "pure_kolokwa" if prob_kolokwa >= 0.5 else "formal_liberian_english"

    return {
        "file": str(audio_path),
        "predicted_label": label,
        "prob_kolokwa": prob_kolokwa,
        "prob_formal": prob_formal,
        "kolokwa_score_percent": f"{prob_kolokwa * 100:.1f}%"
    }


def pick_between_two_audios(audio_a: str, audio_b: str) -> dict:
    """Compare two audio clips and select which one is more Kolokwa vs Formal Liberian English."""
    res_a = score_audio(audio_a)
    res_b = score_audio(audio_b)

    score_a = res_a["prob_kolokwa"]
    score_b = res_b["prob_kolokwa"]

    if score_a > score_b:
        winner = "Audio A"
        winner_path = audio_a
        winner_score = score_a
        loser = "Audio B"
        loser_score = score_b
    else:
        winner = "Audio B"
        winner_path = audio_b
        winner_score = score_b
        loser = "Audio A"
        loser_score = score_a

    margin = abs(score_a - score_b)
    return {
        "more_kolokwa": winner,
        "more_kolokwa_file": winner_path,
        "confidence_margin": f"{margin * 100:.1f}%",
        "audio_a": {"file": str(audio_a), "prob_kolokwa": score_a, "label": res_a["predicted_label"]},
        "audio_b": {"file": str(audio_b), "prob_kolokwa": score_b, "label": res_b["predicted_label"]},
        "verdict": f"{winner} is more authentic Kolokwa ({winner_score*100:.1f}% vs {loser_score*100:.1f}%)"
    }


if __name__ == "__main__":
    if len(sys.argv) == 2:
        res = score_audio(sys.argv[1])
        print("File: " + res["file"])
        print("Prediction: " + res["predicted_label"])
        print(f"P(Kolokwa): {res["prob_kolokwa"]:.4f} ({res["kolokwa_score_percent"]})")
        print(f"P(Formal) : {res["prob_formal"]:.4f}")
    elif len(sys.argv) == 3:
        comp = pick_between_two_audios(sys.argv[1], sys.argv[2])
        print("\n" + "=" * 65)
        print("PAIRWISE AUDIOS COMPARISON RESULT:")
        print("=" * 65)
        print("Result: " + comp["verdict"])
        print("Margin: " + comp["confidence_margin"])
        print("  • Audio A: " + comp["audio_a"]["file"] + f" -> Kolokwa: {comp["audio_a"]["prob_kolokwa"]:.4f}")
        print("  • Audio B: " + comp["audio_b"]["file"] + f" -> Kolokwa: {comp["audio_b"]["prob_kolokwa"]:.4f}")
        print("=" * 65)
    else:
        print("Usage:")
        print("  Score 1 clip : python predict.py <audio_file>")
        print("  Compare 2    : python predict.py <audio_a> <audio_b>")
