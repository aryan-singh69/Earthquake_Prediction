import bisect
import json
import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler


TARGET_KEYS = (
    'label',
    'p_arrival',
    's_arrival',
    'magnitude',
    'latitude',
    'longitude',
    'depth',
)


def _torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


class CachedSeismicDataset(Dataset):
    """
    Chunked tensor dataset produced by scripts/cache_training_data.py.

    Features are already shaped as (3, 6000) and waveform-normalized to match
    STEADDataset. Targets are kept raw so prepare_targets_multitask preserves
    the current training behavior.
    """

    def __init__(self, cache_dir, split='train', max_samples=0,
                 chunks_in_memory=2, return_trace_name=False):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.chunks_in_memory = max(1, int(chunks_in_memory))
        self.return_trace_name = return_trace_name
        self._cache = OrderedDict()

        metadata_path = self.cache_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Cached data metadata not found: {metadata_path}. "
                "Run `python scripts/cache_training_data.py` first."
            )

        with metadata_path.open('r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        splits = self.metadata.get('splits', {})
        if split not in splits:
            available = ', '.join(sorted(splits)) or 'none'
            raise ValueError(f"Split '{split}' not found in cache. Available: {available}")

        self.chunks = splits[split].get('chunks', [])
        if not self.chunks:
            raise ValueError(f"No cached chunks found for split '{split}' in {metadata_path}")

        total_samples = int(splits[split].get('num_samples', 0))
        if max_samples and int(max_samples) > 0:
            total_samples = min(total_samples, int(max_samples))
        self.num_samples = total_samples

        self.chunk_offsets = []
        self.chunk_ends = []
        offset = 0
        for chunk in self.chunks:
            size = int(chunk['num_samples'])
            if offset >= self.num_samples:
                break
            usable = min(size, self.num_samples - offset)
            self.chunk_offsets.append(offset)
            self.chunk_ends.append(offset + usable)
            offset += usable

        self.num_chunks = len(self.chunk_offsets)

    def __len__(self):
        return self.num_samples

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache'] = OrderedDict()
        return state

    def chunk_range(self, chunk_idx):
        return self.chunk_offsets[chunk_idx], self.chunk_ends[chunk_idx]

    def _locate(self, idx):
        if idx < 0:
            idx += self.num_samples
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)

        chunk_idx = bisect.bisect_right(self.chunk_ends, idx)
        start = self.chunk_offsets[chunk_idx]
        return chunk_idx, idx - start

    def _load_chunk(self, chunk_idx):
        if chunk_idx in self._cache:
            self._cache.move_to_end(chunk_idx)
            return self._cache[chunk_idx]

        chunk = self.chunks[chunk_idx]
        features_path = self.cache_dir / chunk['features']
        targets_path = self.cache_dir / chunk['targets']

        features = _torch_load(features_path)
        targets = _torch_load(targets_path)

        self._cache[chunk_idx] = (features, targets)
        self._cache.move_to_end(chunk_idx)
        while len(self._cache) > self.chunks_in_memory:
            self._cache.popitem(last=False)

        return features, targets

    def __getitem__(self, idx):
        chunk_idx, local_idx = self._locate(int(idx))
        features, targets = self._load_chunk(chunk_idx)

        sample = {'features': features[local_idx]}
        for key in TARGET_KEYS:
            sample[key] = targets[key][local_idx]

        if self.return_trace_name and 'trace_name' in targets:
            sample['trace_name'] = targets['trace_name'][local_idx]

        return sample


class CachedChunkBatchSampler(Sampler):
    """
    Yields batches that stay within one cached chunk.

    This keeps random training order without forcing every worker to load a
    different large .pt file for almost every sample.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        chunks = list(range(self.dataset.num_chunks))
        if self.shuffle:
            rng.shuffle(chunks)

        for chunk_idx in chunks:
            start, end = self.dataset.chunk_range(chunk_idx)
            indices = list(range(start, end))
            if self.shuffle:
                rng.shuffle(indices)

            for batch_start in range(0, len(indices), self.batch_size):
                batch = indices[batch_start:batch_start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch

    def __len__(self):
        batches = 0
        for chunk_idx in range(self.dataset.num_chunks):
            start, end = self.dataset.chunk_range(chunk_idx)
            size = end - start
            if self.drop_last:
                batches += size // self.batch_size
            else:
                batches += (size + self.batch_size - 1) // self.batch_size
        return batches
