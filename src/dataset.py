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
        self.metadata['p_arrival_sample']  = self.metadata['p_arrival_sample'].fillna(-1.0)
        self.metadata['s_arrival_sample']  = self.metadata['s_arrival_sample'].fillna(-1.0)
        self.metadata['source_magnitude']  = self.metadata['source_magnitude'].fillna(0.0)
        self.metadata['source_latitude']   = self.metadata['source_latitude'].fillna(0.0)
        self.metadata['source_longitude']  = self.metadata['source_longitude'].fillna(0.0)
        self.metadata['source_depth_km']   = self.metadata['source_depth_km'].fillna(0.0)

        if preload_indices is not None:
            print(f"Preloading {len(preload_indices)} samples into RAM...")
            with h5py.File(hdf5_file, 'r') as f:
                for i, idx in enumerate(preload_indices):
                    trace_name = self.metadata.iloc[idx]['trace_name']
                    self.cache[idx] = f['data'][trace_name][()]
                    if i % 10000 == 0:
                        print(f"  {i}/{len(preload_indices)} loaded...")
            print("Preload complete!")

    def _open_h5(self):
        if self.h5_file_handle is None:
            self.h5_file_handle = h5py.File(self.hdf5_file, 'r')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row        = self.metadata.iloc[idx]
        trace_name = row['trace_name']

        # Data load karo
        if idx in self.cache:
            data = self.cache[idx]
        else:
            self._open_h5()
            data = self.h5_file_handle['data'][trace_name][()]

        data     = data.T  # (6000,3) → (3,6000)
        features = torch.tensor(data, dtype=torch.float32)

        # Detection only 
        if self.task == "detection":
            return {
                'features':   features,
                'label':      torch.tensor(row['label'],      dtype=torch.float32),
                'trace_name': trace_name
            }

        # Phase Picking only 
        elif self.task == "picking":
            return {
                'features':   features,
                'p_arrival':  torch.tensor(row['p_arrival_sample'], dtype=torch.float32),
                's_arrival':  torch.tensor(row['s_arrival_sample'], dtype=torch.float32),
                'trace_name': trace_name
            }

        # Multi-Task 
        elif self.task == "multitask":
            return {
                'features':   features,
                'label':      torch.tensor(row['label'],             dtype=torch.float32),
                'p_arrival':  torch.tensor(row['p_arrival_sample'],  dtype=torch.float32),
                's_arrival':  torch.tensor(row['s_arrival_sample'],  dtype=torch.float32),
                'magnitude':  torch.tensor(row['source_magnitude'],  dtype=torch.float32),
                'latitude':   torch.tensor(row['source_latitude'],   dtype=torch.float32),
                'longitude':  torch.tensor(row['source_longitude'],  dtype=torch.float32),
                'depth':      torch.tensor(row['source_depth_km'],   dtype=torch.float32),
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
    print("Done! ")