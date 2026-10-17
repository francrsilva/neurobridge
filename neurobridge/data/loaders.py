"""Format-specific EEG data loaders for the WBCIC-SHU dataset.

Each loader returns (epochs, labels) in a uniform shape:
  epochs : (N_trials, C, T) float32
  labels : (N_trials,) int64, 0-indexed

The callers in preprocess.py use these loaders and are therefore
independent of file format.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_wbcic_mat(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one WBCIC-SHU preprocessed .mat file.

    Two axis conventions appear in the dataset:

      (C, T, N) — Yang et al. (2025) published format, e.g. shape (58, 1000, 200).
                  Channels are the smallest axis (axis 0). Transposed to canonical
                  (N, C, T) on load.

      (N, C, T) — Thesis-derived recordings, e.g. shape (200, 58, 875).
                  Already in canonical PyTorch order; returned without transposing.

    Convention is detected by comparing axis sizes to labels.size:
    - (C, T, N): data.shape[2] == labels.size → transpose to canonical (N, C, T).
    - (N, C, T): data.shape[0] == labels.size → already canonical; returned as-is.
    This is robust for any n_trials size, including small test fixtures where
    n_trials < n_channels (which breaks argmin-based detection).

    n_times may be 1000 (published 4 s epoch) or 875 (trimmed 3.5 s epoch).
    The caller is responsible for enforcing n_times consistency across subjects
    within an experiment; this function imposes no constraint on T.

    Returns:
        epochs : (N_trials, C, T) float32
        labels : (N_trials,) int64, 0-indexed (label 1 → 0, label 2 → 1, ...)
    """
    import scipy.io as sio

    mat = sio.loadmat(str(path))

    if "data" not in mat:
        raise KeyError(
            f"Expected key 'data' in {path}. "
            f"Found keys: {[k for k in mat.keys() if not k.startswith('_')]}"
        )
    if "labels" not in mat:
        raise KeyError(
            f"Expected key 'labels' in {path}. "
            f"Found keys: {[k for k in mat.keys() if not k.startswith('_')]}"
        )

    data = mat["data"]
    labels_raw = mat["labels"].ravel()

    if data.ndim != 3:
        raise ValueError(
            f"Expected data.ndim == 3, got shape {data.shape} in {path}. "
            f"Supported conventions: (C,T,N) and (N,C,T)."
        )

    # Detect axis convention by matching the trials axis to labels.size.
    # (C, T, N): trials at axis 2 → data.shape[2] == labels.size → transpose to (N,C,T).
    # (N, C, T): trials at axis 0 → data.shape[0] == labels.size → already canonical.
    # Using label count is definitive and independent of relative axis magnitudes,
    # so it works for both real data (N≥195) and small synthetic test fixtures.
    n_labels = labels_raw.size
    if data.shape[2] == n_labels:
        epochs = np.transpose(data, (2, 0, 1)).astype(np.float32)
    elif data.shape[0] == n_labels:
        epochs = data.astype(np.float32)
    else:
        raise ValueError(
            f"Cannot determine axis convention in {path}: shape {data.shape}, "
            f"labels has {n_labels} entries. "
            f"For (C,T,N) data.shape[2] must equal labels count; "
            f"for (N,C,T) data.shape[0] must equal labels count."
        )

    labels = (labels_raw.astype(np.int64) - 1)

    if epochs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Trial count mismatch: data has {epochs.shape[0]} trials, "
            f"labels has {labels.shape[0]} entries in {path}"
        )

    return epochs, labels


def load_wbcic_bdf(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a raw WBCIC-SHU BDF recording and epoch it from annotations.

    This path is provided for completeness but the local dataset is expected
    to be the preprocessed .mat version. Requires MNE.

    NOTE: Raw BDF data (1000 Hz, 64ch) has NOT been bandpass-filtered,
    downsampled, or epoched. Full preprocessing must be applied.

    Returns:
        epochs : (N_trials, C, T) float32  — channels × 4 s × 1000 Hz = 4000 samples
        labels : (N_trials,) int64, 0-indexed
    """
    import mne

    mne.set_log_level("WARNING")

    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose=False)

    # Drop non-EEG channels (ECG ch60, EOG ch61-64).
    non_eeg = [ch for ch in raw.ch_names if "EOG" in ch or "ECG" in ch or "Status" in ch]
    if non_eeg:
        raw.drop_channels(non_eeg)

    sfreq = raw.info["sfreq"]
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    if not events.size:
        raise ValueError(f"No events found in {path}. Check annotation format.")

    # Standard WBCIC-SHU epoch: -0.5 to 4.0 s relative to cue. Adjust if needed.
    tmin, tmax = -0.5, 4.0
    epochs_mne = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=(-0.5, 0.0),
        preload=True, verbose=False,
    )
    data = epochs_mne.get_data().astype(np.float32) * 1e6  # V → µV
    labels = epochs_mne.events[:, 2].astype(np.int64) - 1  # 1-indexed → 0-indexed

    return data, labels


def load_npy_epochs(
    epochs_path: str | Path,
    labels_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load already-processed .npy files as produced by preprocess.py."""
    epochs = np.load(str(epochs_path), mmap_mode="r").astype(np.float32)
    labels = np.load(str(labels_path), mmap_mode="r").astype(np.int64)
    if epochs.ndim != 3:
        raise ValueError(f"Expected shape (N, C, T), got {epochs.shape}")
    return epochs, labels


def load_subject_sessions(
    raw_dir: str | Path,
    subject_id: str,
    file_format: str = "mat",
    n_sessions: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load all session files for one subject and concatenate trials.

    Uses raw_dir/subject_id/ as the canonical lookup: all .{file_format} files
    in that directory are loaded, regardless of filename. In WBCIC-SHU's
    2C_dataset_processeddata, folders sub-011 through sub-051 contain files
    numbered one higher (sub-012_*, sub-013_*, ...); folder name is the source
    of truth, not the filename.

    Falls back to BIDS filename-pattern glob for flat layouts (no per-subject
    subdirectories).

    Returns:
        epochs : (N_total, C, T) float32
        labels : (N_total,) int64
        session_indices : list of cumulative trial counts (for session-level indexing)
    """
    raw_dir = Path(raw_dir)

    subject_dir = raw_dir / subject_id
    if subject_dir.is_dir():
        session_files = sorted(subject_dir.glob(f"*.{file_format}"))
        if not session_files:
            session_files = sorted(subject_dir.glob(f"**/*.{file_format}"))
    else:
        # Fallback: filename-based BIDS glob for flat / alternative layouts
        pattern = f"{subject_id}_ses-*_task-motorimagery_eeg.{file_format}"
        session_files = sorted(raw_dir.glob(f"**/{pattern}"))

    if not session_files:
        raise FileNotFoundError(
            f"No {file_format} files found for {subject_id} under {raw_dir}.\n"
            f"  Checked folder: {subject_dir}\n"
            f"  Run scripts/inspect_dataset.py to see the actual file layout."
        )

    if n_sessions is not None:
        session_files = session_files[:n_sessions]

    all_epochs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    session_indices: list[int] = []
    cumulative = 0

    for sf in session_files:
        if file_format == "mat":
            epochs, labels = load_wbcic_mat(sf)
        elif file_format in ("bdf", "edf"):
            epochs, labels = load_wbcic_bdf(sf)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        all_epochs.append(epochs)
        all_labels.append(labels)
        cumulative += epochs.shape[0]
        session_indices.append(cumulative)

    return (
        np.concatenate(all_epochs, axis=0),
        np.concatenate(all_labels, axis=0),
        session_indices,
    )
