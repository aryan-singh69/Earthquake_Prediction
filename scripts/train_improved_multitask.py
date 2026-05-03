"""
Manual training entrypoint for MultiTaskCNNImproved.

Usage:
  python scripts/train_improved_multitask.py

Notes:
- This script does not auto-run unless executed directly.
- It saves to models/checkpoints/multitask_model_improved.pth by default.
- Existing checkpoints are not overwritten unless you set save_path to same file.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset import STEADDataset
from src.models.multitask_cnn import MultiTaskCNNImproved, load_pretrained_encoder
from src.training.multitask_utils import (
    prepare_targets_multitask,
    compute_multitask_loss,
    compute_multitask_metrics,
)


def _load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _freeze_encoder(model, freeze=True):
    for name, param in model.named_parameters():
        if name.startswith('conv1') or name.startswith('conv2') or name.startswith('conv3'):
            param.requires_grad = not freeze


def train_improved(config_path='configs/improved_multitask_config.yaml'):
    cfg = _load_config(_resolve_path(config_path))

    csv_path = _resolve_path(cfg.get('csv_path', 'data/raw/merge.csv'))
    hdf5_path = _resolve_path(cfg.get('hdf5_path', 'data/raw/merge.hdf5'))
    pretrained_path = _resolve_path(cfg.get('pretrained_path', 'models/checkpoints/best_model.pth'))
    save_path = _resolve_path(cfg.get('save_path', 'models/checkpoints/multitask_model_improved.pth'))

    batch_size = int(cfg.get('batch_size', 256))
    epochs = int(cfg.get('epochs', 20))
    head_lr = float(cfg.get('head_lr', 1e-3))
    finetune_lr = float(cfg.get('finetune_lr', 1e-4))
    freeze_encoder_epochs = int(cfg.get('freeze_encoder_epochs', 3))
    patience = int(cfg.get('patience', 5))
    num_workers = int(cfg.get('num_workers', 0))

    detection_loss_weight = float(cfg.get('detection_loss_weight', 1.0))
    phase_loss_weight = float(cfg.get('phase_loss_weight', 5.0))
    magnitude_loss_weight = float(cfg.get('magnitude_loss_weight', 3.0))
    location_loss_weight = float(cfg.get('location_loss_weight', 0.2))
    use_location_head = bool(cfg.get('use_location_head', False))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print('=' * 70)
    print('  TRAINING: MultiTaskCNNImproved')
    print('=' * 70)
    print(f'Config: {config_path}')
    print(f'Device: {device}')
    print(f'Save path: {save_path}')

    df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)
    eq_df = df[df['trace_category'] == 'earthquake_local'].sample(n=235426, random_state=42)
    noise_df = df[df['trace_category'] == 'noise']
    balanced_df = pd.concat([eq_df, noise_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    indices = list(range(len(balanced_df)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

    dataset = STEADDataset(
        csv_file=csv_path,
        hdf5_file=hdf5_path,
        task='multitask',
        dataframe=balanced_df,
        preload_indices=None,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MultiTaskCNNImproved(use_location_head=use_location_head).to(device)

    if os.path.exists(pretrained_path):
        model = load_pretrained_encoder(model, pretrained_path, device)
    else:
        print(f'Warning: pretrained encoder not found at {pretrained_path}, training from scratch.')

    encoder_params = [
        p for n, p in model.named_parameters()
        if n.startswith('conv1') or n.startswith('conv2') or n.startswith('conv3')
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if not (n.startswith('conv1') or n.startswith('conv2') or n.startswith('conv3'))
    ]

    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': finetune_lr},
        {'params': head_params, 'lr': head_lr},
    ], weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
    )

    detection_loss_fn = nn.BCEWithLogitsLoss()
    phase_loss_fn = nn.SmoothL1Loss()
    magnitude_loss_fn = nn.SmoothL1Loss()
    location_loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        freeze_now = epoch < freeze_encoder_epochs
        _freeze_encoder(model, freeze=freeze_now)

        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            features = batch['features'].to(device, non_blocking=True)
            targets = prepare_targets_multitask(batch, device)

            optimizer.zero_grad()
            outputs = model(features)
            losses = compute_multitask_loss(
                outputs=outputs,
                targets=targets,
                use_location_head=use_location_head,
                detection_loss_weight=detection_loss_weight,
                phase_loss_weight=phase_loss_weight,
                magnitude_loss_weight=magnitude_loss_weight,
                location_loss_weight=location_loss_weight,
                detection_loss_fn=detection_loss_fn,
                phase_loss_fn=phase_loss_fn,
                magnitude_loss_fn=magnitude_loss_fn,
                location_loss_fn=location_loss_fn,
            )
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += losses['total'].item() * features.size(0)

        train_loss = running_loss / max(1, len(train_idx))

        model.eval()
        val_running_loss = 0.0

        y_true_det = []
        y_pred_det = []
        p_true = []
        p_pred = []
        s_true = []
        s_pred = []
        m_true = []
        m_pred = []
        eq_mask_all = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                features = batch['features'].to(device, non_blocking=True)
                targets = prepare_targets_multitask(batch, device)
                outputs = model(features)

                losses = compute_multitask_loss(
                    outputs=outputs,
                    targets=targets,
                    use_location_head=use_location_head,
                    detection_loss_weight=detection_loss_weight,
                    phase_loss_weight=phase_loss_weight,
                    magnitude_loss_weight=magnitude_loss_weight,
                    location_loss_weight=location_loss_weight,
                    detection_loss_fn=detection_loss_fn,
                    phase_loss_fn=phase_loss_fn,
                    magnitude_loss_fn=magnitude_loss_fn,
                    location_loss_fn=location_loss_fn,
                )
                val_running_loss += losses['total'].item() * features.size(0)

                det_prob = torch.sigmoid(outputs['detection']).squeeze(1)
                det_pred = (det_prob >= 0.5).long()
                det_true = targets['det'].squeeze(1).long()

                y_true_det.extend(det_true.cpu().numpy().tolist())
                y_pred_det.extend(det_pred.cpu().numpy().tolist())

                p_true.extend(targets['p'].squeeze(1).cpu().numpy().tolist())
                s_true.extend(targets['s'].squeeze(1).cpu().numpy().tolist())
                m_true.extend(targets['mag'].squeeze(1).cpu().numpy().tolist())

                p_pred.extend(outputs['phase'][:, 0].cpu().numpy().tolist())
                s_pred.extend(outputs['phase'][:, 1].cpu().numpy().tolist())
                m_pred.extend(outputs['magnitude'].squeeze(1).cpu().numpy().tolist())

                eq_mask_all.extend((targets['det'].squeeze(1) > 0.5).cpu().numpy().tolist())

        val_loss = val_running_loss / max(1, len(val_idx))
        metrics = compute_multitask_metrics(
            y_true_det=y_true_det,
            y_pred_det=y_pred_det,
            p_true_norm=p_true,
            p_pred_norm=p_pred,
            s_true_norm=s_true,
            s_pred_norm=s_pred,
            mag_true_norm=m_true,
            mag_pred_norm=m_pred,
            eq_mask=eq_mask_all,
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"freeze_encoder={freeze_now} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"det_acc={metrics['detection_accuracy']:.4f} | "
            f"det_f1={metrics['detection_f1']:.4f} | "
            f"p_mae_sec={metrics['p_wave_mae_sec']} | "
            f"s_mae_sec={metrics['s_wave_mae_sec']} | "
            f"mag_mae={metrics['magnitude_mae']}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f'Saved improved checkpoint to: {save_path}')
        else:
            no_improve += 1
            print(f'No improvement: {no_improve}/{patience}')
            if no_improve >= patience:
                print('Early stopping triggered.')
                break

    summary = {
        'best_val_loss': best_val_loss,
        'checkpoint_path': save_path,
        'use_location_head': use_location_head,
    }
    summary_path = os.path.join(PROJECT_ROOT, 'models', 'metrics', 'improved_training_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\nTraining setup complete. Best model path:')
    print(save_path)


if __name__ == '__main__':
    train_improved()
