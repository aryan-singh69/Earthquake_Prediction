import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import STEADDataset
from models import MultiTaskCNN, load_pretrained_encoder

def train():
    batch_size         = 256
    epochs             = 15
    learning_rate      = 1e-3
    csv_path           = "merge.csv"
    hdf5_path          = "merge.hdf5"
    patience           = 5
    pretrained_path    = "models/best_model.pth"  # purana model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #  Balanced dataframe 
    print("Loading metadata...")
    df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)

    eq_df       = df[df['trace_category'] == 'earthquake_local'].sample(n=235426, random_state=42)
    noise_df    = df[df['trace_category'] == 'noise']
    balanced_df = pd.concat([eq_df, noise_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset: {len(balanced_df)} samples")
    print(balanced_df['trace_category'].value_counts())

    indices = list(range(len(balanced_df)))

    # 80/10/10 split
    train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=42)
    val_idx,   test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    #  Dataset 
    full_dataset = STEADDataset(
        csv_file=csv_path,
        hdf5_file=hdf5_path,
        task="multitask",       # ← multitask mode
        dataframe=balanced_df,
        preload_indices=None
    )

    train_sub = torch.utils.data.Subset(full_dataset, train_idx)
    val_sub   = torch.utils.data.Subset(full_dataset, val_idx)
    test_sub  = torch.utils.data.Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_sub,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model — Transfer Learning 
    model = MultiTaskCNN().to(device)
    model = load_pretrained_encoder(model, pretrained_path, device)

    # Loss functions
    detection_loss_fn  = nn.BCEWithLogitsLoss()
    phase_loss_fn      = nn.MSELoss()
    magnitude_loss_fn  = nn.MSELoss()
    location_loss_fn   = nn.MSELoss()

    # Sirf unfreeze params train honge
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float('inf')
    no_improve    = 0

    for epoch in range(epochs):
        #  Training 
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            features  = batch['features'].to(device)
            det_label = batch['label'].to(device).unsqueeze(1).float()
            p_arr     = batch['p_arrival'].to(device).unsqueeze(1).float()
            s_arr     = batch['s_arrival'].to(device).unsqueeze(1).float()
            mag       = batch['magnitude'].to(device).unsqueeze(1).float()
            lat       = batch['latitude'].to(device).unsqueeze(1).float()
            lon       = batch['longitude'].to(device).unsqueeze(1).float()
            depth     = batch['depth'].to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(features)

            # Individual losses
            loss_det  = detection_loss_fn(outputs['detection'], det_label)
            loss_phase = phase_loss_fn(
                outputs['phase'],
                torch.cat([p_arr, s_arr], dim=1)
            )
            loss_mag  = magnitude_loss_fn(outputs['magnitude'], mag)
            loss_loc  = location_loss_fn(
                outputs['location'],
                torch.cat([lat, lon, depth], dim=1)
            )

            # Combined loss — weights
            loss = (1.0 * loss_det +
                    0.5 * loss_phase +
                    0.3 * loss_mag +
                    0.2 * loss_loc)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_sub)

        # Validation 
        model.eval()
        val_loss  = 0.0
        correct   = 0
        total     = 0

        with torch.no_grad():
            for batch in val_loader:
                features  = batch['features'].to(device)
                det_label = batch['label'].to(device).unsqueeze(1).float()
                p_arr     = batch['p_arrival'].to(device).unsqueeze(1).float()
                s_arr     = batch['s_arrival'].to(device).unsqueeze(1).float()
                mag       = batch['magnitude'].to(device).unsqueeze(1).float()
                lat       = batch['latitude'].to(device).unsqueeze(1).float()
                lon       = batch['longitude'].to(device).unsqueeze(1).float()
                depth     = batch['depth'].to(device).unsqueeze(1).float()

                outputs   = model(features)

                loss_det   = detection_loss_fn(outputs['detection'], det_label)
                loss_phase = phase_loss_fn(outputs['phase'], torch.cat([p_arr, s_arr], dim=1))
                loss_mag   = magnitude_loss_fn(outputs['magnitude'], mag)
                loss_loc   = location_loss_fn(outputs['location'], torch.cat([lat, lon, depth], dim=1))

                loss = (1.0 * loss_det +
                        0.5 * loss_phase +
                        0.3 * loss_mag +
                        0.2 * loss_loc)

                val_loss += loss.item() * features.size(0)

                # Detection accuracy
                preds   = torch.sigmoid(outputs['detection']) >= 0.5
                correct += (preds == det_label.bool()).sum().item()
                total   += det_label.size(0)

        val_loss   = val_loss / len(val_sub)
        val_acc    = correct / total
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), "models/multitask_model.pth")
            print(f"   Best model saved! Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        else:
            no_improve += 1
            print(f"   No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                print(f" Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    # Final Test 
    print("\nFinal Test Evaluation...")
    model.load_state_dict(torch.load("models/multitask_model.pth", weights_only=True))
    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in test_loader:
            features  = batch['features'].to(device)
            det_label = batch['label'].to(device).unsqueeze(1).float()
            outputs   = model(features)
            preds     = torch.sigmoid(outputs['detection']) >= 0.5
            correct  += (preds == det_label.bool()).sum().item()
            total    += det_label.size(0)

    test_acc = correct / total
    print(f"\n Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f" Best Val Loss:       {best_val_loss:.4f}")
    print("Training complete!")

if __name__ == "__main__":
    train()