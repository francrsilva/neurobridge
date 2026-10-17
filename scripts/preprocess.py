#!/usr/bin/env python
"""Preprocess WBCIC-SHU data: load .mat files, normalize, save .npy.

Run AFTER scripts/inspect_dataset.py has confirmed the dataset layout.

The WBCIC-SHU preprocessed .mat files have already been:
  - Bandpass filtered (0.5–40 Hz)
  - Downsampled to 250 Hz
  - Epoched and baseline-corrected

This script does NOT repeat any of those steps. It performs only:
  1. Per-channel z-score normalization (fitted on training subjects only)
  2. Subject-level splits (created from the manifest)
  3. Conversion to .npy files for efficient PyTorch DataLoader access

Usage (on Vulpix):
    python scripts/preprocess.py paths.data_raw=datasets

Output layout in paths.data_processed:
    sub-001_epochs.npy   # (N_trials, C, T) float32
    sub-001_labels.npy   # (N_trials,) int64, 0-indexed
    preprocessor.json    # fitted channel mean/std
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from neurobridge.data.loaders import load_subject_sessions
from neurobridge.data.preprocessing import EEGPreprocessor, PreprocessConfig
from neurobridge.data.splits import create_splits, load_splits, save_splits


# ── Dataset layout discovery ────────────────────────────────────────────────

def discover_subjects(raw_dir: Path, file_format: str = "mat") -> list[str]:
    """Find all subject IDs by scanning for BIDS-like file names."""
    subject_ids: set[str] = set()
    for path in raw_dir.rglob(f"*.{file_format}"):
        for part in path.parts:
            if part.startswith("sub-"):
                subject_ids.add(part)
                break
        else:
            # Flat layout: subject id embedded in filename.
            stem = path.stem  # sub-001_ses-01_task-motorimagery_eeg
            for token in stem.split("_"):
                if token.startswith("sub-"):
                    subject_ids.add(token)
                    break
    return sorted(subject_ids)


# ── Per-subject loading ──────────────────────────────────────────────────────

def load_subject(
    raw_dir: Path,
    subject_id: str,
    file_format: str,
    dataset_cfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all sessions for one subject, returning concatenated (epochs, labels).

    Handles both 2C (labels 0,1) and 3C (labels 0,1,2) subjects correctly
    because loaders.py maps 1-indexed labels to 0-indexed uniformly.
    """
    epochs, labels, _ = load_subject_sessions(
        raw_dir=raw_dir,
        subject_id=subject_id,
        file_format=file_format,
    )

    # Validate shape against expected dimensions from config.
    expected_c = int(dataset_cfg.n_channels)
    expected_t = int(dataset_cfg.n_times)

    if epochs.shape[1] != expected_c:
        raise ValueError(
            f"{subject_id}: expected {expected_c} channels, got {epochs.shape[1]}.\n"
            f"  Run inspect_dataset.py and update configs/dataset/wbcic_shu_*.yaml."
        )
    if epochs.shape[2] != expected_t:
        raise ValueError(
            f"{subject_id}: expected {expected_t} time points, got {epochs.shape[2]}.\n"
            f"  Run inspect_dataset.py and update n_times in model config."
        )

    return epochs, labels


# ── Main ────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raw_dir = Path(cfg.paths.data_raw)
    processed_dir = Path(cfg.paths.data_processed)
    splits_path = Path(cfg.paths.splits)
    preprocessor_path = Path(cfg.paths.preprocessor)

    if not raw_dir.exists():
        print(f"[preprocess] ERROR: {raw_dir} not found. Run inspect_dataset.py first.")
        sys.exit(1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_path.parent.mkdir(parents=True, exist_ok=True)

    file_format = cfg.dataset.file_format   # "mat" or "bdf"

    # ── Discover subjects ───────────────────────────────────────────────── #
    subject_ids = discover_subjects(raw_dir, file_format)
    if not subject_ids:
        print(f"[preprocess] ERROR: No .{file_format} files found under {raw_dir}.")
        sys.exit(1)
    print(f"[preprocess] Found {len(subject_ids)} subjects.")

    # ── Subject-level splits ─────────────────────────────────────────────── #
    if splits_path.exists():
        print(f"[preprocess] Loading existing splits from {splits_path}")
        splits = load_splits(splits_path)
        splits.validate()
        # Warn if discovered subjects differ from splits.
        known = splits.all_subjects()
        extra = set(subject_ids) - known
        missing = known - set(subject_ids)
        if extra:
            print(f"  WARNING: {len(extra)} subjects in data not in splits manifest: {sorted(extra)[:5]}")
        if missing:
            print(f"  WARNING: {len(missing)} subjects in manifest not found in data: {sorted(missing)[:5]}")
    else:
        print("[preprocess] Creating new subject splits ...")
        splits = create_splits(
            subject_ids=subject_ids,
            val_ratio=cfg.dataset.val_ratio,
            test_ratio=cfg.dataset.test_ratio,
            fewshot_ratio=cfg.dataset.fewshot_ratio,
            seed=cfg.training.seed,
            dataset_version=cfg.dataset.version,
        )
        save_splits(splits, splits_path)
        print(f"  train={len(splits.train)} val={len(splits.val)} "
              f"test={len(splits.test)} fewshot={len(splits.fewshot)}")

    # ── Fit normalization on training subjects only ──────────────────────── #
    print("[preprocess] Fitting per-channel normalization on training subjects ...")
    train_epochs_list: list[np.ndarray] = []

    for sid in tqdm(splits.train, desc="Fit (train)"):
        try:
            epochs, _ = load_subject(raw_dir, sid, file_format, cfg.dataset)
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: {e}")
            continue
        train_epochs_list.append(epochs)

    if not train_epochs_list:
        print("[preprocess] ERROR: No training subjects loaded.")
        sys.exit(1)

    # Fit on all training epochs concatenated.
    # PreprocessConfig with sfreq is required even for already-preprocessed data
    # because the epoch/filter methods won't be called — only fit/normalize.
    pcfg = PreprocessConfig(
        sfreq=cfg.dataset.sfreq,
        epoch_len_s=cfg.dataset.epoch_len_s,
    )
    preprocessor = EEGPreprocessor(pcfg)
    all_train = np.concatenate(train_epochs_list, axis=0)
    preprocessor.fit(all_train)
    preprocessor.save(preprocessor_path)
    print(f"[preprocess] Preprocessor saved → {preprocessor_path}")
    print(f"  Fitted on {all_train.shape[0]} training trials "
          f"({all_train.shape[1]} channels, {all_train.shape[2]} time points)")

    # ── Normalize and save all subjects ─────────────────────────────────── #
    all_subject_ids = splits.train + splits.val + splits.test + splits.fewshot
    print("[preprocess] Normalizing and saving all subjects ...")

    for sid in tqdm(all_subject_ids, desc="Normalize (all)"):
        try:
            epochs, labels = load_subject(raw_dir, sid, file_format, cfg.dataset)
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: {e}")
            continue

        normalized = preprocessor.normalize(epochs)
        np.save(str(processed_dir / f"{sid}_epochs.npy"), normalized.astype(np.float32))
        np.save(str(processed_dir / f"{sid}_labels.npy"), labels.astype(np.int64))

    print(f"[preprocess] Done. Output in {processed_dir}")
    print("[preprocess] Next: run scripts/train.py or scripts/pretrain.py (after explicit approval).")


if __name__ == "__main__":
    main()
