import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
sns.set_theme(style="whitegrid")

def perform_eda(csv_path):
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    print("\n--- Basic Information ---")
    print(f"Total Traces: {len(df)}")
    print(df['trace_category'].value_counts())
    
    # 1. Class Imbalance Plot
    plt.figure(figsize=(8, 5))
    sns.countplot(x='trace_category', data=df)
    plt.title('Trace Categories in STEAD Dataset')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    print("Saved 'class_distribution.png'")
    plt.close()

    # 2. Magnitude Distribution (For structural Earthquakes)
    earthquakes = df[df['trace_category'] == 'earthquake_local'].copy()
    
    if not earthquakes.empty and 'source_magnitude' in earthquakes.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(earthquakes['source_magnitude'].dropna(), bins=50, kde=True, color='red')
        plt.title('Earthquake Magnitude Distribution')
        plt.xlabel('Magnitude')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('magnitude_distribution.png')
        print("Saved 'magnitude_distribution.png'")
        plt.close()
    
    # 3. Earthquake Depth Distribution
    if not earthquakes.empty and 'source_depth_km' in earthquakes.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(earthquakes['source_depth_km'].dropna(), bins=50, kde=True, color='blue')
        plt.title('Earthquake Source Depth (km)')
        plt.xlabel('Depth (km)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('depth_distribution.png')
        print("Saved 'depth_distribution.png'")
        plt.close()

    # 4. Source Distance Distribution
    if not earthquakes.empty and 'source_distance_km' in earthquakes.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(earthquakes['source_distance_km'].dropna(), bins=50, kde=True, color='green')
        plt.title('Source-Receiver Distance (km)')
        plt.xlabel('Distance (km)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('distance_distribution.png')
        print("Saved 'distance_distribution.png'")
        plt.close()

    print("\nEDA generated 4 plots in the current directory.")

if __name__ == "__main__":
    # Get the directory of this script (notebooks folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve the absolute path of merge.csv in the parent directory
    csv_path = os.path.abspath(os.path.join(script_dir, "..", "merge.csv"))
    
    # Change working directory so plots are saved strictly in notebooks folder
    os.chdir(script_dir)

    if os.path.exists(csv_path):
        perform_eda(csv_path)
    else:
        print(f"File {csv_path} not found.")
