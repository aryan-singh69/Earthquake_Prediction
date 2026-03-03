import h5py
import pandas as pd

df = pd.read_csv('merge.csv', low_memory=False)
print("Columns:", df.columns.tolist())
print("Categories:", df['trace_category'].unique())
print("Total rows:", len(df))

h5 = h5py.File('merge.hdf5', 'r')
print("\nHDF5 groups:", list(h5.keys()))
first_group = list(h5.keys())[0]
print("First group keys:", list(h5[first_group].keys())[:5])

# Pehli trace ka shape dekho
first_trace = list(h5[first_group].keys())[0]
print("First trace shape:", h5[first_group][first_trace][()].shape)
h5.close()

