#!/usr/bin/env python3
"""Prepare balanced training and validation sets for Kolokwa vs Formal Liberian English classifier.
- Class 1 (Pure Kolokwa): 333 LIRNHGP1DA files (~11 hours) sliced into 3-6s chunks.
- Class 0 (Formal Liberian English): Sampled formal speech from michsethowusu/kolokwa-speech-text (~11 hours).
- Audio formatted to 16,000 Hz mono FLAC.
- Episode/source-based 85/15 train/val split (zero speaker leakage).
"""
import csv
import glob
import io
import os
import random
import subprocess
import soundfile as sf
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

DATA_DIR = Path("/mnt/volume_d2wey28/data/kolokwa_classifier")
KOLOKWA_RAW_DIR = DATA_DIR / "raw" / "kolokwa"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
CHUNK_MIN_SEC = 3.0
CHUNK_MAX_SEC = 6.0
VAL_RATIO = 0.15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def slice_audio(audio_path, out_dir, prefix, label):
    """Slice audio into 3-6s chunks at 16kHz mono FLAC."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(audio_path),
        "-ac", "1", "-ar", str(TARGET_SR), "-f", "s16le", "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return []

    pcm_data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    total_samples = len(pcm_data)
    chunk_samples = int(CHUNK_MAX_SEC * TARGET_SR)
    min_samples = int(CHUNK_MIN_SEC * TARGET_SR)

    out_records = []
    chunk_idx = 0
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        if (end - start) < min_samples:
            continue
        chunk = pcm_data[start:end]
        # Skip silence
        if np.max(np.abs(chunk)) < 0.01:
            continue

        filename = f"{prefix}_{chunk_idx:04d}.flac"
        out_path = out_dir / filename
        sf.write(str(out_path), chunk, TARGET_SR, format="FLAC")
        dur = len(chunk) / TARGET_SR
        out_records.append((str(out_path), label, dur, prefix))
        chunk_idx += 1

    return out_records


def main():
    kolokwa_files = sorted(list(KOLOKWA_RAW_DIR.glob("*.mp3")))
    print(f"Found {len(kolokwa_files)} Kolokwa MP3 files.")

    # Group Kolokwa files by book/chapter to ensure speaker/recording separation in split
    random.shuffle(kolokwa_files)
    n_val_k = int(len(kolokwa_files) * VAL_RATIO)
    val_k_files = set(kolokwa_files[:n_val_k])
    train_k_files = set(kolokwa_files[n_val_k:])

    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    records_train = []
    records_val = []

    print("\n--- Slicing Class 1: Pure Kolokwa ---")
    for f in kolokwa_files:
        stem = f.stem
        is_val = f in val_k_files
        target_dir = val_dir if is_val else train_dir
        recs = slice_audio(f, target_dir, stem, label=1)
        if is_val:
            records_val.extend(recs)
        else:
            records_train.extend(recs)

    total_k_dur_train = sum(r[2] for r in records_train if r[1] == 1) / 3600.0
    total_k_dur_val = sum(r[2] for r in records_val if r[1] == 1) / 3600.0
    print(f"Kolokwa sliced: {total_k_dur_train:.2f}h train ({len([r for r in records_train if r[1]==1])} clips), {total_k_dur_val:.2f}h val ({len([r for r in records_val if r[1]==1])} clips)")

    print("\n--- Downloading and Slicing Class 0: Formal Liberian English ---")
    # Target same duration for formal class
    target_formal_train_sec = total_k_dur_train * 3600.0
    target_formal_val_sec = total_k_dur_val * 3600.0

    cur_formal_train_sec = 0.0
    cur_formal_val_sec = 0.0

    shard_idx = 0
    while (cur_formal_train_sec < target_formal_train_sec or cur_formal_val_sec < target_formal_val_sec) and shard_idx < 15:
        shard_path = f"data/train-{shard_idx:05d}.parquet"
        print(f"Reading {shard_path} from michsethowusu/kolokwa-speech-text ...")
        local_shard = hf_hub_download(repo_id="michsethowusu/kolokwa-speech-text", filename=shard_path, repo_type="dataset")
        table = pq.read_table(local_shard)
        pydict = table.to_pydict()
        num_rows = table.num_rows

        is_val_shard = (shard_idx % 7 == 0) # Shard-based separation for formal class
        target_dir = val_dir if is_val_shard else train_dir

        for i in range(num_rows):
            if is_val_shard and cur_formal_val_sec >= target_formal_val_sec:
                continue
            if not is_val_shard and cur_formal_train_sec >= target_formal_train_sec:
                continue

            audio_entry = pydict["audio"][i]
            audio_bytes = audio_entry["bytes"]
            audio_buf = io.BytesIO(audio_bytes)
            data, sr = sf.read(audio_buf)
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            # Resample to 16kHz
            if sr != TARGET_SR:
                import soxr
                data = soxr.resample(data, sr, TARGET_SR)

            total_samples = len(data)
            chunk_samples = int(CHUNK_MAX_SEC * TARGET_SR)
            min_samples = int(CHUNK_MIN_SEC * TARGET_SR)

            for c_i, start in enumerate(range(0, total_samples, chunk_samples)):
                end = min(start + chunk_samples, total_samples)
                if (end - start) < min_samples:
                    continue
                chunk = data[start:end]
                if np.max(np.abs(chunk)) < 0.01:
                    continue

                filename = f"formal_{shard_idx:03d}_{i:04d}_{c_i:02d}.flac"
                out_path = target_dir / filename
                sf.write(str(out_path), chunk, TARGET_SR, format="FLAC")
                dur = len(chunk) / TARGET_SR

                rec = (str(out_path), 0, dur, f"formal_{shard_idx}")
                if is_val_shard:
                    records_val.append(rec)
                    cur_formal_val_sec += dur
                else:
                    records_train.append(rec)
                    cur_formal_train_sec += dur

        os.remove(local_shard)
        shard_idx += 1

    total_f_dur_train = sum(r[2] for r in records_train if r[1] == 0) / 3600.0
    total_f_dur_val = sum(r[2] for r in records_val if r[1] == 0) / 3600.0
    print(f"Formal sliced: {total_f_dur_train:.2f}h train, {total_f_dur_val:.2f}h val")

    # Write CSV manifests
    random.shuffle(records_train)
    random.shuffle(records_val)

    with open(OUTPUT_DIR / "train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "duration", "speaker_id"])
        writer.writerows(records_train)

    with open(OUTPUT_DIR / "val.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "duration", "speaker_id"])
        writer.writerows(records_val)

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE!")
    print(f"Train set: {len(records_train)} clips ({sum(r[2] for r in records_train)/3600:.2f} hours)")
    print(f"  • Class 1 (Pure Kolokwa)          : {len([r for r in records_train if r[1]==1])} clips ({total_k_dur_train:.2f} h)")
    print(f"  • Class 0 (Formal Liberian English): {len([r for r in records_train if r[1]==0])} clips ({total_f_dur_train:.2f} h)")
    print(f"Val set:   {len(records_val)} clips ({sum(r[2] for r in records_val)/3600:.2f} hours)")
    print(f"  • Class 1 (Pure Kolokwa)          : {len([r for r in records_val if r[1]==1])} clips ({total_k_dur_val:.2f} h)")
    print(f"  • Class 0 (Formal Liberian English): {len([r for r in records_val if r[1]==0])} clips ({total_f_dur_val:.2f} h)")
    print(f"Manifests saved to: {OUTPUT_DIR}/train.csv and val.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
