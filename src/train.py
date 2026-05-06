import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import sys
import time
import numpy as np
import pandas as pd

from torch.cuda.amp import autocast, GradScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import STEADDataset
from models import MultiTaskCNN, load_pretrained_encoder

# ── Normalization constants ───────────────────────────────────
P_S_MAX   = 6000.0
MAG_MAX   = 9.0
LAT_MAX   = 90.0
LON_MAX   = 180.0
DEPTH_MAX = 700.0

# ── Loss weights ──────────────────────────────────────────────
W_DET   = 1.0
W_PHASE = 0.5
W_MAG   = 0.5
W_LOC   = 0.5


def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory,
                    persistent_workers=True, prefetch_factor=2):
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = persistent_workers
        kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def sync_if_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def prepare_targets(batch, device):
    """Batch se targets normalize karke return karo."""
    det_label = batch['label'].to(device, non_blocking=True).unsqueeze(1).float()

    p_arr = batch['p_arrival'].to(device, non_blocking=True).unsqueeze(1).float() / P_S_MAX
    s_arr = batch['s_arrival'].to(device, non_blocking=True).unsqueeze(1).float() / P_S_MAX
    mag   = batch['magnitude'].to(device, non_blocking=True).unsqueeze(1).float() / MAG_MAX
    lat   = batch['latitude'].to(device,  non_blocking=True).unsqueeze(1).float() / LAT_MAX
    lon   = batch['longitude'].to(device, non_blocking=True).unsqueeze(1).float() / LON_MAX
    depth = batch['depth'].to(device,     non_blocking=True).unsqueeze(1).float() / DEPTH_MAX

    p_arr = p_arr.clamp(0, 1)
    s_arr = s_arr.clamp(0, 1)
    lat   = lat.clamp(-1, 1)
    lon   = lon.clamp(-1, 1)
    depth = depth.clamp(0, 1)

    return det_label, p_arr, s_arr, mag, lat, lon, depth


def compute_losses(outputs, det_label, p_arr, s_arr, mag, lat, lon, depth,
                   detection_loss_fn, phase_loss_fn, magnitude_loss_fn, location_loss_fn,
                   device):
    """Har task ka loss compute karo."""
    loss_det = detection_loss_fn(outputs['detection'], det_label)

    eq_mask    = det_label.squeeze(1).bool()
    loss_phase = torch.tensor(0.0, device=device)
    loss_mag   = torch.tensor(0.0, device=device)
    loss_loc   = torch.tensor(0.0, device=device)

    if eq_mask.sum() > 0:
        p_eq = p_arr[eq_mask]
        s_eq = s_arr[eq_mask]

        # Sirf valid arrivals pe phase loss
        valid_mask = (p_eq.squeeze(1) > 0) & (s_eq.squeeze(1) > 0)
        if valid_mask.sum() > 0:
            phase_pred   = outputs['phase'][eq_mask][valid_mask]
            phase_target = torch.cat([p_eq[valid_mask], s_eq[valid_mask]], dim=1)
            loss_phase   = phase_loss_fn(phase_pred, phase_target)

        loss_mag = magnitude_loss_fn(
            outputs['magnitude'][eq_mask],
            mag[eq_mask]
        )

        loc_target = torch.cat([lat[eq_mask], lon[eq_mask], depth[eq_mask]], dim=1)
        loss_loc   = location_loss_fn(outputs['location'][eq_mask], loc_target)

    total = (W_DET   * loss_det   +
             W_PHASE * loss_phase  +
             W_MAG   * loss_mag    +
             W_LOC   * loss_loc)

    return total, loss_det, loss_phase, loss_mag, loss_loc


def train():
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    batch_size      = 256
    epochs          = 20
    csv_path        = "merge.csv"
    hdf5_path       = "merge.hdf5"
    patience        = 3
    pretrained_path = "models/best_model.pth"
    save_path       = "models/multitask_model.pth"
    max_grad_norm   = 1.0
    num_workers     = min(8, max(4, (os.cpu_count() or 8) // 2))
    prefetch_factor = 2
    perf_log_interval = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pin_memory = device.type == 'cuda'
    use_amp = device.type == 'cuda'
    print(
        f"DataLoader: batch_size={batch_size}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}, persistent_workers=True, "
        f"prefetch_factor={prefetch_factor}"
    )
    print(f"Mixed precision AMP: {'enabled' if use_amp else 'disabled'}")

    # ── Dataset ───────────────────────────────────────────────
    print("Loading metadata...")
    df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)

    eq_df       = df[df['trace_category'] == 'earthquake_local'].sample(n=235426, random_state=42)
    noise_df    = df[df['trace_category'] == 'noise']
    balanced_df = pd.concat([eq_df, noise_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset: {len(balanced_df)} samples")
    print(balanced_df['trace_category'].value_counts())

    indices = list(range(len(balanced_df)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=42)
    val_idx,   test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    full_dataset = STEADDataset(
        csv_file=csv_path,
        hdf5_file=hdf5_path,
        task="multitask",
        dataframe=balanced_df,
        preload_indices=None
    )

    train_loader = make_dataloader(
        torch.utils.data.Subset(full_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    val_loader = make_dataloader(
        torch.utils.data.Subset(full_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    test_loader = make_dataloader(
        torch.utils.data.Subset(full_dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    # ── Model ─────────────────────────────────────────────────
    model = MultiTaskCNN().to(device)
    model = load_pretrained_encoder(model, pretrained_path, device)

    # ── Differential LR ───────────────────────────────────────
    # Encoder (conv1/2/3) — pretrained hai, slow train karo
    # Heads + fc_shared   — naye hain, fast train karo
    encoder_params = [p for n, p in model.named_parameters()
                      if any(k in n for k in ['conv1', 'conv2', 'conv3'])]
    other_params   = [p for n, p in model.named_parameters()
                      if not any(k in n for k in ['conv1', 'conv2', 'conv3'])]

    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': 1e-4},   # Encoder — slow
        {'params': other_params,   'lr': 1e-3},   # Heads   — fast
    ], weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # ── Loss functions ────────────────────────────────────────
    detection_loss_fn = nn.BCEWithLogitsLoss()
    phase_loss_fn     = nn.MSELoss()
    magnitude_loss_fn = nn.MSELoss()
    location_loss_fn  = nn.MSELoss()
    scaler = GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    no_improve    = 0

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────
        model.train()
        running = {'loss': 0.0, 'det': 0.0, 'ph': 0.0, 'mag': 0.0, 'loc': 0.0}
        loop    = tqdm(train_loader, leave=True)
        perf = {'load': 0.0, 'compute': 0.0, 'samples': 0, 'batches': 0}
        data_wait_start = time.perf_counter()

        for step, batch in enumerate(loop, start=1):
            batch_ready = time.perf_counter()
            load_time = batch_ready - data_wait_start
            compute_start = time.perf_counter()

            features  = batch['features'].to(device, non_blocking=True)
            det_label, p_arr, s_arr, mag, lat, lon, depth = prepare_targets(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                outputs = model(features)

                loss, loss_det, loss_phase, loss_mag, loss_loc = compute_losses(
                    outputs, det_label, p_arr, s_arr, mag, lat, lon, depth,
                    detection_loss_fn, phase_loss_fn, magnitude_loss_fn, location_loss_fn,
                    device
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            sync_if_cuda(device)

            bs = features.size(0)
            loss_value = loss.item()
            loss_det_value = loss_det.item()
            loss_phase_value = loss_phase.item()
            loss_mag_value = loss_mag.item()
            loss_loc_value = loss_loc.item()

            running['loss'] += loss_value       * bs
            running['det']  += loss_det_value   * bs
            running['ph']   += loss_phase_value * bs
            running['mag']  += loss_mag_value   * bs
            running['loc']  += loss_loc_value   * bs

            compute_time = time.perf_counter() - compute_start
            perf['load'] += load_time
            perf['compute'] += compute_time
            perf['samples'] += bs
            perf['batches'] += 1

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(
                loss=f"{loss_value:.4f}",
                det=f"{loss_det_value:.3f}",
                ph=f"{loss_phase_value:.3f}",
                mag=f"{loss_mag_value:.3f}",
                loc=f"{loss_loc_value:.3f}"
            )

            if perf_log_interval > 0 and step % perf_log_interval == 0:
                elapsed = perf['load'] + perf['compute']
                tqdm.write(
                    f"[perf] epoch={epoch+1} step={step} "
                    f"iter={elapsed / max(1, perf['batches']):.3f}s "
                    f"load={perf['load'] / max(1, perf['batches']):.3f}s "
                    f"compute={perf['compute'] / max(1, perf['batches']):.3f}s "
                    f"samples/sec={perf['samples'] / max(elapsed, 1e-9):.1f}"
                )
                perf = {'load': 0.0, 'compute': 0.0, 'samples': 0, 'batches': 0}

            data_wait_start = time.perf_counter()

        train_loss = running['loss'] / len(train_idx)

        # ── Validation ────────────────────────────────────────
        model.eval()
        val_running = {'loss': 0.0, 'det': 0.0, 'ph': 0.0, 'mag': 0.0, 'loc': 0.0}
        correct = 0
        total   = 0

        with torch.no_grad():
            for batch in val_loader:
                features  = batch['features'].to(device, non_blocking=True)
                det_label, p_arr, s_arr, mag, lat, lon, depth = prepare_targets(batch, device)

                with autocast(enabled=use_amp):
                    outputs = model(features)

                    loss, loss_det, loss_phase, loss_mag, loss_loc = compute_losses(
                        outputs, det_label, p_arr, s_arr, mag, lat, lon, depth,
                        detection_loss_fn, phase_loss_fn, magnitude_loss_fn, location_loss_fn,
                        device
                    )

                bs = features.size(0)
                val_running['loss'] += loss.item()       * bs
                val_running['det']  += loss_det.item()   * bs
                val_running['ph']   += loss_phase.item() * bs
                val_running['mag']  += loss_mag.item()   * bs
                val_running['loc']  += loss_loc.item()   * bs

                preds   = torch.sigmoid(outputs['detection']) >= 0.5
                correct += (preds == det_label.bool()).sum().item()
                total   += det_label.size(0)

        val_loss = val_running['loss'] / len(val_idx)
        val_acc  = correct / total
        curr_lr  = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"LR: {curr_lr:.6f}"
        )
        print(
            f"         Val → "
            f"det={val_running['det']/len(val_idx):.4f}  "
            f"ph={val_running['ph']/len(val_idx):.4f}  "
            f"mag={val_running['mag']/len(val_idx):.4f}  "
            f"loc={val_running['loc']/len(val_idx):.4f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), save_path)
            print(f"   Saved! Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        else:
            no_improve += 1
            print(f"     No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                print(f" Early stopping at epoch {epoch+1}")
                break

    # ── Final Test ───────────────────────────────────────────
    print("\nFinal Test Evaluation...")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in test_loader:
            features  = batch['features'].to(device, non_blocking=True)
            det_label = batch['label'].to(device, non_blocking=True).unsqueeze(1).float()
            with autocast(enabled=use_amp):
                outputs = model(features)
            preds     = torch.sigmoid(outputs['detection']) >= 0.5
            correct  += (preds == det_label.bool()).sum().item()
            total    += det_label.size(0)

    test_acc = correct / total
    print(f"\n Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f" Best Val Loss:       {best_val_loss:.4f}")
    print(" Training complete!")


if __name__ == "__main__":
    train()
