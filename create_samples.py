import h5py
import numpy as np
import pandas as pd
import os

os.makedirs('samples', exist_ok=True)

print("Loading CSV and HDF5...")
df = pd.read_csv('merge.csv', low_memory=False)
h5 = h5py.File('merge.hdf5', 'r')

# HDF5 structure: h5['data'][trace_name] → shape (6000, 3)
def get_data_from_h5(trace_name):
    try:
        return h5['data'][trace_name][()]  # Direct access — no loop needed
    except KeyError:
        print(f"  Not found: {trace_name}")
        return None

def save_samples(category, prefix, num_samples=5):
    print(f"\nSaving {num_samples} {category} samples...")
    sub_df = df[df['trace_category'] == category].head(num_samples * 2)  # extra lelo agar kuch miss ho
    
    saved = 0
    for i, row in sub_df.iterrows():
        if saved >= num_samples:
            break
            
        trace_name = row['trace_name']
        data = get_data_from_h5(trace_name)
        
        if data is not None:
            # Shape already (6000, 3) — seedha save karo
            print(f"  Shape: {data.shape}")
            file_path = f'samples/{prefix}_{saved}.npy'
            np.save(file_path, data)
            print(f"  Saved: {file_path}")
            saved += 1

    print(f"  Total saved: {saved}/{num_samples}")

# 5-5 real samples save karo
save_samples('earthquake_local', 'earthquake_local', num_samples=10)
save_samples('noise', 'noise', num_samples=10)

h5.close()
print("\nDone! Samples saved in samples/ folder")
print("Files:", os.listdir('samples'))