import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


P_S_MAX = 6000.0
MAG_MAX = 9.0
LAT_MAX = 90.0
LON_MAX = 180.0
DEPTH_MAX = 700.0


def prepare_targets_multitask(batch, device):
    det = batch['label'].to(device, non_blocking=True).unsqueeze(1).float()
    p = batch['p_arrival'].to(device, non_blocking=True).unsqueeze(1).float() / P_S_MAX
    s = batch['s_arrival'].to(device, non_blocking=True).unsqueeze(1).float() / P_S_MAX
    mag = batch['magnitude'].to(device, non_blocking=True).unsqueeze(1).float() / MAG_MAX
    lat = batch['latitude'].to(device, non_blocking=True).unsqueeze(1).float() / LAT_MAX
    lon = batch['longitude'].to(device, non_blocking=True).unsqueeze(1).float() / LON_MAX
    depth = batch['depth'].to(device, non_blocking=True).unsqueeze(1).float() / DEPTH_MAX

    return {
        'det': det,
        'p': p,
        's': s,
        'mag': mag,
        'lat': lat,
        'lon': lon,
        'depth': depth,
    }


def compute_multitask_loss(
    outputs,
    targets,
    use_location_head=False,
    detection_loss_weight=1.0,
    phase_loss_weight=5.0,
    magnitude_loss_weight=3.0,
    location_loss_weight=0.2,
    detection_loss_fn=None,
    phase_loss_fn=None,
    magnitude_loss_fn=None,
    location_loss_fn=None,
):
    if detection_loss_fn is None:
        detection_loss_fn = nn.BCEWithLogitsLoss()
    if phase_loss_fn is None:
        phase_loss_fn = nn.SmoothL1Loss()
    if magnitude_loss_fn is None:
        magnitude_loss_fn = nn.SmoothL1Loss()
    if location_loss_fn is None:
        location_loss_fn = nn.MSELoss()

    det = targets['det']
    p = targets['p']
    s = targets['s']
    mag = targets['mag']
    lat = targets['lat']
    lon = targets['lon']
    depth = targets['depth']

    loss_det = detection_loss_fn(outputs['detection'], det)

    eq_mask = det.squeeze(1) > 0.5
    loss_phase = torch.zeros((), device=det.device)
    loss_mag = torch.zeros((), device=det.device)
    loss_loc = torch.zeros((), device=det.device)

    if eq_mask.any():
        p_eq = p[eq_mask]
        s_eq = s[eq_mask]

        # Ignore invalid phase labels: missing values are encoded as -1 in source data
        valid_p = torch.isfinite(p_eq.squeeze(1)) & (p_eq.squeeze(1) >= 0)
        valid_s = torch.isfinite(s_eq.squeeze(1)) & (s_eq.squeeze(1) >= 0)
        valid_phase = valid_p & valid_s

        if valid_phase.any():
            phase_target = torch.cat([p_eq[valid_phase], s_eq[valid_phase]], dim=1)
            phase_pred = outputs['phase'][eq_mask][valid_phase]
            loss_phase = phase_loss_fn(phase_pred, phase_target)

        mag_eq = mag[eq_mask]
        valid_mag = torch.isfinite(mag_eq.squeeze(1))
        if valid_mag.any():
            loss_mag = magnitude_loss_fn(outputs['magnitude'][eq_mask][valid_mag], mag_eq[valid_mag])

        if use_location_head and ('location' in outputs):
            loc_target = torch.cat([lat[eq_mask], lon[eq_mask], depth[eq_mask]], dim=1)
            valid_loc = torch.isfinite(loc_target).all(dim=1)
            if valid_loc.any():
                loss_loc = location_loss_fn(outputs['location'][eq_mask][valid_loc], loc_target[valid_loc])

    total = (
        detection_loss_weight * loss_det
        + phase_loss_weight * loss_phase
        + magnitude_loss_weight * loss_mag
        + (location_loss_weight * loss_loc if use_location_head else 0.0)
    )

    return {
        'total': total,
        'detection': loss_det,
        'phase': loss_phase,
        'magnitude': loss_mag,
        'location': loss_loc,
    }


def compute_multitask_metrics(
    y_true_det,
    y_pred_det,
    p_true_norm,
    p_pred_norm,
    s_true_norm,
    s_pred_norm,
    mag_true_norm,
    mag_pred_norm,
    eq_mask,
):
    y_true_det = np.asarray(y_true_det, dtype=int)
    y_pred_det = np.asarray(y_pred_det, dtype=int)

    p_true_norm = np.asarray(p_true_norm, dtype=float)
    p_pred_norm = np.asarray(p_pred_norm, dtype=float)
    s_true_norm = np.asarray(s_true_norm, dtype=float)
    s_pred_norm = np.asarray(s_pred_norm, dtype=float)
    mag_true_norm = np.asarray(mag_true_norm, dtype=float)
    mag_pred_norm = np.asarray(mag_pred_norm, dtype=float)
    eq_mask = np.asarray(eq_mask, dtype=bool)

    # As requested: seconds = normalized_value * 6000 / 100
    sec_scale = 6000.0 / 100.0

    p_valid = eq_mask & np.isfinite(p_true_norm) & np.isfinite(p_pred_norm) & (p_true_norm >= 0)
    s_valid = eq_mask & np.isfinite(s_true_norm) & np.isfinite(s_pred_norm) & (s_true_norm >= 0)
    m_valid = eq_mask & np.isfinite(mag_true_norm) & np.isfinite(mag_pred_norm)

    out = {
        'detection_accuracy': float(accuracy_score(y_true_det, y_pred_det)),
        'detection_f1': float(f1_score(y_true_det, y_pred_det, zero_division=0)),
        'p_wave_mae_sec': None,
        's_wave_mae_sec': None,
        'magnitude_mae': None,
    }

    if p_valid.any():
        out['p_wave_mae_sec'] = float(np.mean(np.abs(p_pred_norm[p_valid] - p_true_norm[p_valid]) * sec_scale))

    if s_valid.any():
        out['s_wave_mae_sec'] = float(np.mean(np.abs(s_pred_norm[s_valid] - s_true_norm[s_valid]) * sec_scale))

    if m_valid.any():
        # Convert normalized magnitude error back to original magnitude scale
        out['magnitude_mae'] = float(np.mean(np.abs(mag_pred_norm[m_valid] - mag_true_norm[m_valid]) * MAG_MAX))

    return out
