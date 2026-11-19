# Kolokwa vs. Formal Liberian English Audio Classifier

Fine-tuned **Wav2Vec2-XLS-R-300M** for speech register identification and pairwise dialect ranking between:
- **`pure_kolokwa` (Class 1):** Authentic Liberian Pidgin English (Kolokwa / Koloqua) as spoken natively across Liberia.
- **`formal_liberian_english` (Class 0):** Standard broadcast, academic, and formal Liberian English.

Published on Hugging Face: [**`AfriSpeech/kolokwa-classifier`**](https://huggingface.co/AfriSpeech/kolokwa-classifier)

---

## Highlights

- **Zero Transcripts Required:** Operates directly on raw audio waveforms at 16,000 Hz.
- **Pairwise Comparison:** Pick between any two audio clips to determine which is more authentic Kolokwa.
- **Single-Clip Spectrum Scoring:** Score any recording from `0.0` (Formal) to `1.0` (Deep Kolokwa).
- **Acoustic Generalization:** Frozen convolutional encoder preserves multilingual phonetic priors while adapting upper Transformer layers.
- **Evaluation:** Achieves **100.0% accuracy** and **1.0000 ROC-AUC** on an unseen, speaker-disjoint validation split.

---

## Installation

```bash
git clone https://github.com/AfriSpeech/kolokwa-classifier.git
cd kolokwa-classifier
pip install -r requirements.txt
```

*(Note: `ffmpeg` must be installed on your system to support all audio formats).*

---

## Inference

### 1. Pairwise Audio Comparison (Pick Between Two Clips)

Compare two recordings to determine which one is more authentic Kolokwa:

```bash
python predict.py audio_a.mp3 audio_b.wav
```

Output:
```
=================================================================
PAIRWISE AUDIOS COMPARISON RESULT:
=================================================================
Result: Audio A is more authentic Kolokwa (99.6% vs 0.4%)
Margin: 99.2%
  • Audio A: audio_a.mp3 -> Kolokwa: 0.9964
  • Audio B: audio_b.wav -> Kolokwa: 0.0040
=================================================================
```

### 2. Single Clip Scoring

Score a single recording on the dialect register spectrum:

```bash
python predict.py sample.mp3
```

Output:
```
File: sample.mp3
Prediction: pure_kolokwa
P(Kolokwa): 0.9964 (99.6%)
P(Formal) : 0.0036
```

### 3. In Python

```python
from predict import pick_between_two_audios, score_audio

# Pairwise comparison
result = pick_between_two_audios("clip1.mp3", "clip2.wav")
print(result["verdict"])
print(f"Winner: {result['more_kolokwa_file']}")

# Single clip score
score = score_audio("clip.mp3")
print(f"Class: {score['predicted_label']}")
print(f"P(Kolokwa): {score['prob_kolokwa']:.4f}")
```

### 4. Direct via Hugging Face Pipeline

```python
from transformers import pipeline

classifier = pipeline("audio-classification", model="AfriSpeech/kolokwa-classifier")
print(classifier("clip.mp3"))
# [{'label': 'pure_kolokwa', 'score': 0.9964}, {'label': 'formal_liberian_english', 'score': 0.0036}]
```

---

## Dataset & Training Pipeline

### Data Preparation
- **Class 1 (Pure Kolokwa):** 333 native recordings from the *LIRNHGP1DA* Liberian Kolokwa corpus (~11.02 hours).
- **Class 0 (Formal Liberian English):** Formal speech from `michsethowusu/kolokwa-speech-text` (~11.02 hours).
- Chunks standardized to 3–6s at 16,000 Hz mono FLAC.
- 85/15 train/val split partitioned strictly by chapter/speaker to eliminate leakage.

To reproduce data slicing:
```bash
python prepare_data.py
```

### Fine-Tuning
Fine-tuning uses `AutoModelForAudioClassification` with frozen feature encoders on native `bf16`:
```bash
python train.py
```

---

## License

MIT License. Developed under the **AfriSpeech** initiative.
