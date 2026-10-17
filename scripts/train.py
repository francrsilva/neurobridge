#!/usr/bin/env python
"""Supervised classification training (EEGNet baseline and supervised transformer).

Run AFTER scripts/preprocess.py has produced the processed .npy files and
the splits manifest at paths.splits.

Usage (on Vulpix):
    python scripts/train.py \
        model=eegnet \
        training=eegnet_baseline \
        paths.data_raw=datasets/2C_dataset_processeddata \
        mlflow.experiment_name=eegnet_baseline_2c

Output:
    experiments/checkpoints/eegnet_baseline/best.pt  — best val_bacc checkpoint
    experiments/mlruns/                              — MLflow run
"""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from neurobridge.data.dataset import EEGDataset, build_dataloader
from neurobridge.data.splits import load_splits
from neurobridge.evaluation.metrics import (
    balanced_accuracy,
    expected_calibration_error,
    risk_coverage_curve,
)
from neurobridge.models.eegnet import EEGNet
from neurobridge.training.trainer import Trainer, TrainerConfig


# ── Model factory ────────────────────────────────────────────────────────────

def _build_model(cfg: DictConfig) -> nn.Module:
    name = cfg.model.name
    if name == "eegnet":
        return EEGNet(
            n_channels=int(cfg.dataset.n_channels),
            n_times=int(cfg.dataset.n_times),
            n_classes=int(cfg.dataset.n_classes),
            sfreq=float(cfg.dataset.sfreq),
            F1=int(cfg.model.F1),
            D=int(cfg.model.D),
            F2=int(cfg.model.F2),
            dropout=float(cfg.model.dropout),
        )
    raise ValueError(
        f"Unknown model '{name}'. Supported in train.py: eegnet. "
        f"For transformer, use scripts/pretrain.py."
    )


# ── Optimizer and scheduler ───────────────────────────────────────────────────

def _build_optimizer_and_scheduler(
    model: nn.Module, cfg: DictConfig
) -> tuple:
    opt_name = cfg.training.optimizer.lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=float(cfg.training.weight_decay),
            betas=(float(cfg.training.beta1), float(cfg.training.beta2)),
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.training.lr),
            momentum=0.9,
            weight_decay=float(cfg.training.weight_decay),
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

    sched_name = cfg.training.scheduler.lower()
    if sched_name == "cosine_warmup":
        warmup = int(cfg.training.warmup_epochs)
        remaining = max(1, int(cfg.training.max_epochs) - warmup)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=remaining),
            ],
            milestones=[warmup],
        )
    elif sched_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {cfg.training.scheduler}")

    return optimizer, scheduler


# ── Git hash ─────────────────────────────────────────────────────────────────

def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).parent.parent),
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ── Post-training evaluation ──────────────────────────────────────────────────

def _evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Return balanced_accuracy, ECE, and risk-coverage on loader."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)   # (N, C)
    labels_arr = np.concatenate(all_labels, axis=0)  # (N,)
    preds = probs_arr.argmax(axis=1)

    rc = risk_coverage_curve(labels_arr, probs_arr)
    return {
        "balanced_accuracy": float(balanced_accuracy(labels_arr, preds)),
        "ece": float(expected_calibration_error(labels_arr, probs_arr)),
        "rc_100": float(rc.get(1.0, float("nan"))),
        "rc_90":  float(rc.get(0.9, float("nan"))),
        "rc_80":  float(rc.get(0.8, float("nan"))),
        "rc_70":  float(rc.get(0.7, float("nan"))),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import mlflow

    processed_dir = Path(cfg.paths.data_processed)
    splits_path = Path(cfg.paths.splits)
    ckpt_dir = Path(cfg.training.checkpoint_dir)

    # ── Seed ────────────────────────────────────────────────────────────── #
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Splits ──────────────────────────────────────────────────────────── #
    if not splits_path.exists():
        print(f"[train] ERROR: splits not found at {splits_path}.")
        print("  Run scripts/preprocess.py first.")
        sys.exit(1)

    splits = load_splits(splits_path)
    print(
        f"[train] Splits: train={len(splits.train)}, val={len(splits.val)}, "
        f"test={len(splits.test)}, fewshot={len(splits.fewshot)}"
    )

    # ── Datasets ────────────────────────────────────────────────────────── #
    try:
        train_ds = EEGDataset(processed_dir, splits.train)
        val_ds   = EEGDataset(processed_dir, splits.val)
    except FileNotFoundError as e:
        print(f"[train] ERROR: {e}")
        print("  Run scripts/preprocess.py first.")
        sys.exit(1)

    print(
        f"[train] Dataset: {len(train_ds)} train trials, "
        f"{len(val_ds)} val trials, {train_ds.n_classes} classes."
    )
    print(f"  Train class distribution: {train_ds.class_distribution()}")
    print(f"  Val   class distribution: {val_ds.class_distribution()}")

    # num_workers=4 on Linux/Vulpix; Windows multiprocessing requires 0
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = build_dataloader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # ── Model ────────────────────────────────────────────────────────────── #
    model = _build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model: {cfg.model.name}, {n_params:,} parameters.")

    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg)

    # ── MLflow ──────────────────────────────────────────────────────────── #
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        mlflow.set_tag("git_commit", _git_hash())
        mlflow.set_tag("dataset_version", cfg.dataset.version)
        mlflow.log_params({
            "seed":           seed,
            "model":          cfg.model.name,
            "n_channels":     cfg.dataset.n_channels,
            "n_times":        cfg.dataset.n_times,
            "n_classes":      train_ds.n_classes,
            "dataset_version": cfg.dataset.version,
            "max_epochs":     cfg.training.max_epochs,
            "patience":       cfg.training.patience,
            "lr":             cfg.training.lr,
            "weight_decay":   cfg.training.weight_decay,
            "batch_size":     cfg.training.batch_size,
            "grad_accum":     cfg.training.gradient_accumulation_steps,
            "optimizer":      cfg.training.optimizer,
            "scheduler":      cfg.training.scheduler,
            "monitor_metric": cfg.training.monitor_metric,
            "n_train_subj":   len(splits.train),
            "n_val_subj":     len(splits.val),
            "n_params":       n_params,
        })

        # ── Training ────────────────────────────────────────────────────── #
        trainer_cfg = TrainerConfig(
            max_epochs=int(cfg.training.max_epochs),
            gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
            grad_clip_norm=float(cfg.training.grad_clip_norm),
            use_amp=bool(cfg.training.use_amp),
            amp_dtype=str(cfg.training.amp_dtype),
            checkpoint_dir=str(ckpt_dir),
            save_every_n_epochs=int(cfg.training.save_every_n_epochs),
            gpu_safety_threshold_mib=float(cfg.training.gpu_safety_threshold_mib),
            seed=seed,
            patience=int(cfg.training.patience),
            monitor_metric=str(cfg.training.monitor_metric),
            monitor_mode=str(cfg.training.monitor_mode),
            track_bacc=bool(getattr(cfg.training, "track_bacc", True)),
        )

        trainer = Trainer(
            model=model,
            config=trainer_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=nn.CrossEntropyLoss(),
            mlflow_run=run,
        )

        print(f"[train] Training on {trainer.device}.")
        history = trainer.fit()

        final_epoch = len(history["train_loss"])
        print(
            f"[train] Finished after {final_epoch} epoch(s). "
            f"Best {trainer_cfg.monitor_metric}: {trainer._best_metric:.4f}"
        )

        # ── Evaluate best checkpoint ─────────────────────────────────────── #
        best_ckpt = ckpt_dir / "best.pt"
        if best_ckpt.exists():
            trainer.load_checkpoint(best_ckpt)
        else:
            print("[train] WARNING: no best.pt found; using last weights.")

        device = trainer.device
        val_metrics = _evaluate(model, val_loader, device)
        mlflow.log_metrics({f"final_{k}": v for k, v in val_metrics.items()})

        print(f"\n[train] Final val metrics (best checkpoint):")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\n[train] Run ID: {run.info.run_id}")
        print(f"  View: mlflow ui --backend-store-uri {cfg.mlflow.tracking_uri}")


if __name__ == "__main__":
    main()
