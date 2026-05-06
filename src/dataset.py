import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class STEADDataset(Dataset):
    def __init__(self, csv_file, hdf5_file, task="detection", transform=None,
                 preload_indices=None, dataframe=None):
        self.task           = task
        self.transform      = transform
        self.hdf5_file      = hdf5_file
        self.h5_file_handle = None
        self.cache          = {}

        # Dataframe ya CSV load karo
        if dataframe is not None:
            print("Using provided balanced dataframe...")
            self.metadata = dataframe.reset_index(drop=True)
        else:
            print("Loading metadata CSV...")
            self.metadata = pd.read_csv(csv_file, low_memory=False).reset_index(drop=True)

        # Labels banao
        self.metadata['label'] = (self.metadata['trace_category'] != 'noise').astype(int)

        # NaN values fill karo
        # p/s arrival: -1 = missing (noise samples mein hoga)
        self.metadata['p_arrival_sample'] = self.metadata['p_arrival_sample'].fillna(-1.0)
        self.metadata['s_arrival_sample'] = self.metadata['s_arrival_sample'].fillna(-1.0)
        self.metadata['source_magnitude'] = self.metadata['source_magnitude'].fillna(0.0)
        self.metadata['source_latitude']  = self.metadata['source_latitude'].fillna(0.0)
        self.metadata['source_longitude'] = self.metadata['source_longitude'].fillna(0.0)
        self.metadata['source_depth_km']  = self.metadata['source_depth_km'].fillna(0.0)

        # Cache metadata as NumPy arrays once. This avoids pandas iloc work in
        # every __getitem__ call while preserving the exact same sample content.
        self.trace_names = self.metadata['trace_name'].astype(str).to_numpy()
        self.labels = self.metadata['label'].to_numpy(dtype=np.float32)
        self.p_arrivals = self.metadata['p_arrival_sample'].to_numpy(dtype=np.float32)
        self.s_arrivals = self.metadata['s_arrival_sample'].to_numpy(dtype=np.float32)
        self.magnitudes = self.metadata['source_magnitude'].to_numpy(dtype=np.float32)
        self.latitudes = self.metadata['source_latitude'].to_numpy(dtype=np.float32)
        self.longitudes = self.metadata['source_longitude'].to_numpy(dtype=np.float32)
        self.depths = self.metadata['source_depth_km'].to_numpy(dtype=np.float32)
        self.num_samples = len(self.trace_names)

        if preload_indices is not None:
            print(f"Preloading {len(preload_indices)} samples into RAM...")
            with h5py.File(hdf5_file, 'r') as f:
                for i, idx in enumerate(preload_indices):
                    trace_name      = self.trace_names[idx]
                    self.cache[idx] = f['data'][trace_name][()]
                    if i % 10000 == 0:
                        print(f"  {i}/{len(preload_indices)} loaded...")
            print("Preload complete!")

    def _open_h5(self):
        """Har worker ka apna HDF5 handle hoga — parallel safe"""
        if self.h5_file_handle is None:
            self.h5_file_handle = h5py.File(
                self.hdf5_file, 'r',
                swmr=True
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['h5_file_handle'] = None
        state['metadata'] = None
        return state

    def __del__(self):
        handle = getattr(self, 'h5_file_handle', None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    @staticmethod
    def normalize_waveform(data: np.ndarray) -> np.ndarray:
        """
        Per-channel zero-mean, unit-std normalization.
        data shape: (3, 6000)
        Yeh zaroori hai — raw seismic amplitudes bahut badi hoti hain
        jo loss explode karti hain.
        """
        mean = data.mean(axis=1, keepdims=True)         # (3, 1)
        std  = data.std(axis=1,  keepdims=True) + 1e-8  # (3, 1) — div-by-zero se bachao
        return (data - mean) / std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        trace_name = self.trace_names[idx]

        # ── Waveform load karo ────────────────────────────────
        if idx in self.cache:
            data = self.cache[idx]
        else:
            self._open_h5()
            data = self.h5_file_handle['data'][trace_name][()]

        data     = np.asarray(data, dtype=np.float32).T  # (6000, 3) → (3, 6000)
        data     = self.normalize_waveform(data).astype(np.float32, copy=False)
        features = torch.from_numpy(np.ascontiguousarray(data))

        # ── Detection only ────────────────────────────────────
        if self.task == "detection":
            return {
                'features':   features,
                'label':      torch.tensor(self.labels[idx], dtype=torch.float32),
                'trace_name': trace_name
            }

        # ── Phase Picking only ────────────────────────────────
        elif self.task == "picking":
            return {
                'features':   features,
                'p_arrival':  torch.tensor(self.p_arrivals[idx], dtype=torch.float32),
                's_arrival':  torch.tensor(self.s_arrivals[idx], dtype=torch.float32),
                'trace_name': trace_name
            }

        # ── Multi-Task ────────────────────────────────────────
        elif self.task == "multitask":
            return {
                'features':   features,
                'label':      torch.tensor(self.labels[idx], dtype=torch.float32),
                'p_arrival':  torch.tensor(self.p_arrivals[idx], dtype=torch.float32),
                's_arrival':  torch.tensor(self.s_arrivals[idx], dtype=torch.float32),
                'magnitude':  torch.tensor(self.magnitudes[idx], dtype=torch.float32),
                'latitude':   torch.tensor(self.latitudes[idx], dtype=torch.float32),
                'longitude':  torch.tensor(self.longitudes[idx], dtype=torch.float32),
                'depth':      torch.tensor(self.depths[idx], dtype=torch.float32),
                'trace_name': trace_name
            }

        else:
            raise ValueError(f"Unknown task: {self.task}. Use 'detection', 'picking', or 'multitask'")


if __name__ == "__main__":
    import os
    base      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path  = os.path.join(base, "merge.csv")
    hdf5_path = os.path.join(base, "merge.hdf5")

    print("Testing detection mode...")
    ds = STEADDataset(csv_file=csv_path, hdf5_file=hdf5_path, task="detection")
    s  = ds[0]
    print(f"  Features: {s['features'].shape} | Label: {s['label']}")
    print(f"  Features min/max: {s['features'].min():.4f} / {s['features'].max():.4f}")

    print("\nTesting multitask mode...")
    ds2 = STEADDataset(csv_file=csv_path, hdf5_file=hdf5_path, task="multitask")
    s2  = ds2[0]
    print(f"  Features:  {s2['features'].shape}")
    print(f"  Label:     {s2['label']}")
    print(f"  P arrival: {s2['p_arrival']}")
    print(f"  S arrival: {s2['s_arrival']}")
    print(f"  Magnitude: {s2['magnitude']}")
    print(f"  Latitude:  {s2['latitude']}")
    print(f"  Longitude: {s2['longitude']}")
    print(f"  Depth:     {s2['depth']}")
    print("Done!")
