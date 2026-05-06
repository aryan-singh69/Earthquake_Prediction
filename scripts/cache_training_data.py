"""
Build a chunked tensor cache for fast multitask seismic training.

Usage:
  python scripts/cache_training_data.py

The cache preserves the same balanced dataframe and train/val split used by
scripts/train_improved_multitask.py, but reads merge.hdf5 only once up front.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import STEADDataset
from src.training.multitask_utils import (
    DEPTH_MAX,
    LAT_MAX,
    LON_MAX,
    MAG_MAX,
    P_S_MAX,
)


DEFAULT_CONFIG = PROJECT_ROOT / 'configs' / 'improved_multitask_config.yaml'
CACHE_PATTERNS = (
    'train_features_*.pt',
    'train_targets_*.pt',
    'val_features_*.pt',
    'val_targets_*.pt',
    'metadata.json',
)


def _load_config(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _parse_args():
    parser = argparse.ArgumentParser(description='Cache STEAD training tensors in chunks.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--csv-path', default=None)
    parser.add_argument('--hdf5-path', default=None)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--chunk-size', type=int, default=None)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    return parser.parse_args()


def _prepare_balanced_dataframe(csv_path):
    print(f'Reading metadata CSV: {csv_path}')
    df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)

    eq_df = df[df['trace_category'] == 'earthquake_local'].sample(n=235426, random_state=42)
    noise_df = df[df['trace_category'] == 'noise']
    balanced_df = pd.concat([eq_df, noise_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    balanced_df['label'] = (balanced_df['trace_category'] != 'noise').astype(np.float32)
    balanced_df['p_arrival_sample'] = balanced_df['p_arrival_sample'].fillna(-1.0)
    balanced_df['s_arrival_sample'] = balanced_df['s_arrival_sample'].fillna(-1.0)
    balanced_df['source_magnitude'] = balanced_df['source_magnitude'].fillna(0.0)
    balanced_df['source_latitude'] = balanced_df['source_latitude'].fillna(0.0)
    balanced_df['source_longitude'] = balanced_df['source_longitude'].fillna(0.0)
    balanced_df['source_depth_km'] = balanced_df['source_depth_km'].fillna(0.0)

    return balanced_df


def _clean_existing_cache(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for pattern in CACHE_PATTERNS:
        for path in cache_dir.glob(pattern):
            path.unlink()


def _validate_cache_dir(cache_dir, csv_path, hdf5_path):
    cache_dir_resolved = cache_dir.resolve()
    raw_dirs = {csv_path.resolve().parent, hdf5_path.resolve().parent}
    if cache_dir_resolved in raw_dirs:
        raise ValueError(f'Refusing to write cache into raw data directory: {cache_dir_resolved}')


def _waveform_to_tensor(waveform):
    data = np.asarray(waveform, dtype=np.float32)
    if data.shape == (6000, 3):
        data = data.T
    elif data.shape != (3, 6000):
        raise ValueError(f'Unexpected waveform shape: {data.shape}')

    data = STEADDataset.normalize_waveform(data).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(data))


def _target_tensors(rows):
    label = torch.as_tensor(rows['label'].to_numpy(dtype=np.float32))
    p_arrival = torch.as_tensor(rows['p_arrival_sample'].to_numpy(dtype=np.float32))
    s_arrival = torch.as_tensor(rows['s_arrival_sample'].to_numpy(dtype=np.float32))
    magnitude = torch.as_tensor(rows['source_magnitude'].to_numpy(dtype=np.float32))
    latitude = torch.as_tensor(rows['source_latitude'].to_numpy(dtype=np.float32))
    longitude = torch.as_tensor(rows['source_longitude'].to_numpy(dtype=np.float32))
    depth = torch.as_tensor(rows['source_depth_km'].to_numpy(dtype=np.float32))

    return {
        'label': label,
        'p_arrival': p_arrival,
        's_arrival': s_arrival,
        'magnitude': magnitude,
        'latitude': latitude,
        'longitude': longitude,
        'depth': depth,
        'p_arrival_norm': p_arrival / P_S_MAX,
        's_arrival_norm': s_arrival / P_S_MAX,
        'magnitude_norm': magnitude / MAG_MAX,
        'latitude_norm': latitude / LAT_MAX,
        'longitude_norm': longitude / LON_MAX,
        'depth_norm': depth / DEPTH_MAX,
    }


def _cache_split(split_name, split_df, h5_data, cache_dir, chunk_size):
    chunks = []
    total = len(split_df)
    progress = tqdm(total=total, desc=f'Caching {split_name}', unit='sample')

    for chunk_idx, start in enumerate(range(0, total, chunk_size)):
        end = min(start + chunk_size, total)
        rows = split_df.iloc[start:end].reset_index(drop=True)
        features = torch.empty((len(rows), 3, 6000), dtype=torch.float32)

        for local_idx, trace_name in enumerate(rows['trace_name'].astype(str).to_numpy()):
            features[local_idx] = _waveform_to_tensor(h5_data[trace_name][()])
            progress.update(1)

        targets = _target_tensors(rows)
        features_name = f'{split_name}_features_{chunk_idx:03d}.pt'
        targets_name = f'{split_name}_targets_{chunk_idx:03d}.pt'
        torch.save(features, cache_dir / features_name)
        torch.save(targets, cache_dir / targets_name)

        chunks.append({
            'features': features_name,
            'targets': targets_name,
            'num_samples': len(rows),
        })

        del features, targets

    progress.close()
    return chunks


def main():
    args = _parse_args()
    cfg = _load_config(args.config)

    csv_path = _resolve_path(args.csv_path or cfg.get('csv_path', 'data/raw/merge.csv'))
    hdf5_path = _resolve_path(args.hdf5_path or cfg.get('hdf5_path', 'data/raw/merge.hdf5'))
    cache_dir = _resolve_path(args.cache_dir or cfg.get('cache_dir', 'data/processed/cache'))
    chunk_size = int(args.chunk_size or cfg.get('chunk_size', 10000))
    max_train_samples = int(
        args.max_train_samples
        if args.max_train_samples is not None
        else cfg.get('max_train_samples', 0)
    )
    max_val_samples = int(
        args.max_val_samples
        if args.max_val_samples is not None
        else cfg.get('max_val_samples', 0)
    )

    if chunk_size <= 0:
        raise ValueError('chunk_size must be > 0')

    _validate_cache_dir(cache_dir, csv_path, hdf5_path)

    balanced_df = _prepare_balanced_dataframe(csv_path)
    indices = list(range(len(balanced_df)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

    train_df = balanced_df.iloc[train_idx].reset_index(drop=True)
    val_df = balanced_df.iloc[val_idx].reset_index(drop=True)

    full_train_count = len(train_df)
    full_val_count = len(val_df)
    if max_train_samples > 0:
        train_df = train_df.iloc[:max_train_samples].reset_index(drop=True)
    if max_val_samples > 0:
        val_df = val_df.iloc[:max_val_samples].reset_index(drop=True)

    print(f'Cache directory: {cache_dir}')
    print(f'Chunk size: {chunk_size}')
    print(f'Train samples: {len(train_df)} (full split: {full_train_count})')
    print(f'Val samples:   {len(val_df)} (full split: {full_val_count})')

    _clean_existing_cache(cache_dir)

    with h5py.File(hdf5_path, 'r', swmr=True) as h5_file:
        h5_data = h5_file['data']
        train_chunks = _cache_split('train', train_df, h5_data, cache_dir, chunk_size)
        val_chunks = _cache_split('val', val_df, h5_data, cache_dir, chunk_size)

    metadata = {
        'version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'csv_path': os.path.relpath(csv_path, PROJECT_ROOT),
        'hdf5_path': os.path.relpath(hdf5_path, PROJECT_ROOT),
        'chunk_size': chunk_size,
        'feature_shape': [3, 6000],
        'waveform_normalization': 'per-channel zero mean/unit std, same as STEADDataset',
        'target_storage': 'raw target keys are used by training; *_norm keys are cached for reference',
        'balanced_samples': len(balanced_df),
        'split_random_state': 42,
        'train_split_full_samples': full_train_count,
        'val_split_full_samples': full_val_count,
        'test_split_full_samples': len(test_idx),
        'splits': {
            'train': {
                'num_samples': len(train_df),
                'chunks': train_chunks,
            },
            'val': {
                'num_samples': len(val_df),
                'chunks': val_chunks,
            },
        },
    }

    metadata_path = cache_dir / 'metadata.json'
    with metadata_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f'Cache complete: {metadata_path}')


if __name__ == '__main__':
    main()
