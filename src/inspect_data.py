import h5py
import pandas as pd
import os

HDF5_PATH = "merge.hdf5"
CSV_PATH = "merge.csv"

def inspect_data():
    if not os.path.exists(HDF5_PATH) or not os.path.exists(CSV_PATH):
        print("Data files not found in the current directory.")
        return

    print("--- Inspecting CSV Metadata ---")
    # Read the first few rows of CSV to inspect metadata
    df = pd.read_csv(CSV_PATH, nrows=5, low_memory=False)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\n--- Inspecting HDF5 File ---")
    with h5py.File(HDF5_PATH, "r") as f:
        print(f"Top-level keys in HDF5: {list(f.keys())}")
        
        # Verify the structure inside 'data' without loading thousands of keys into memory.
        if 'data' in f:
            print(f"'data' group found.")
            # Get an interator so we don't block
            iterator = iter(f['data'].keys())
            print("First 5 datasets inside 'data':")
            for _ in range(5):
                key = next(iterator)
                dataset = f['data'][key]
                print(f" - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        else:
            print("Warning: 'data' group NOT found! (This might be causing KeyErrors)")

if __name__ == "__main__":
    inspect_data()


