"""
HDF5 utilities — inspection and sample extraction from STEAD dataset.
"""

import os
import h5py
import numpy as np
import pandas as pd


def inspect_hdf5(hdf5_path, csv_path=None, num_samples=5):
    """
    Inspect HDF5 structure and optionally CSV metadata.

    Args:
        hdf5_path:   Path to STEAD HDF5 file.
        csv_path:    Optional path to STEAD metadata CSV.
        num_samples: Number of sample entries to display.
    """
    if csv_path and os.path.exists(csv_path):
        print("--- Inspecting CSV Metadata ---")
        df = pd.read_csv(csv_path, nrows=num_samples, low_memory=False)
        print("Columns:", df.columns.tolist())
        print(f"\nFirst {num_samples} rows:")
        print(df.head())

    print("\n--- Inspecting HDF5 File ---")
    with h5py.File(hdf5_path, "r") as f:
        print(f"Top-level keys: {list(f.keys())}")
        if "data" in f:
            print("'data' group found.")
            it = iter(f["data"].keys())
            print(f"First {num_samples} datasets inside 'data':")
            for _ in range(num_samples):
                key = next(it)
                ds  = f["data"][key]
                print(f"  - {key}: shape={ds.shape}, dtype={ds.dtype}")
        else:
            print("Warning: 'data' group NOT found!")


def extract_samples(csv_path, hdf5_path, output_dir, num_samples=5):
    """
    Extract sample .npy waveforms from HDF5 for testing.

    Args:
        csv_path:    Path to STEAD metadata CSV.
        hdf5_path:   Path to STEAD HDF5 file.
        output_dir:  Directory to save .npy files.
        num_samples: Number of samples per category.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, low_memory=False)

    with h5py.File(hdf5_path, "r") as h5:
        for category, prefix in [("earthquake_local", "earthquake_local"), ("noise", "noise")]:
            print(f"\nSaving {num_samples} {category} samples...")
            sub_df = df[df["trace_category"] == category].head(num_samples * 2)
            saved = 0
            for _, row in sub_df.iterrows():
                if saved >= num_samples:
                    break
                try:
                    data = h5["data"][row["trace_name"]][()]
                    path = os.path.join(output_dir, f"{prefix}_{saved}.npy")
                    np.save(path, data)
                    print(f"  Saved: {path} (shape: {data.shape})")
                    saved += 1
                except KeyError:
                    print(f"  Not found: {row['trace_name']}")
            print(f"  Total saved: {saved}/{num_samples}")

    print(f"\nDone! Files: {os.listdir(output_dir)}")
