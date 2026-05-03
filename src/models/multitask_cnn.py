"""
MultiTaskCNN — Multi-output seismic analysis model.

Produces four outputs simultaneously:
  1. Detection     — Earthquake vs Noise (binary)
  2. Phase Picking — P-wave arrival, S-wave arrival
  3. Magnitude     — Earthquake magnitude
  4. Location      — Latitude, Longitude, Depth

Migrated from original src/models.py — all behavior preserved.
Checkpoint-compatible with: multitask_model.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskCNN(nn.Module):
    """
    Multi-Task CNN for seismic signal analysis.

    Input:  (batch, 3, 6000) — 3-component seismic waveform
    Output: dict with keys 'detection', 'phase', 'magnitude', 'location'

    Args:
        in_channels: Number of input channels (default: 3).
    """

    def __init__(self, in_channels=3):
        super(MultiTaskCNN, self).__init__()

        # ── Shared CNN Encoder ────────────────────────────────
        self.conv1 = nn.Conv1d(in_channels, 32,  kernel_size=21, stride=1, padding=10)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(32,  64,  kernel_size=15, stride=1, padding=7)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(64,  128, kernel_size=11, stride=1, padding=5)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Shared FC
        self.fc_shared = nn.Linear(11904, 512)
        self.dropout   = nn.Dropout(0.5)

        # ── Task Heads ────────────────────────────────────────

        # Head 1 — Detection (Earthquake vs Noise)
        self.detection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # Head 2 — Phase Picking (P-wave, S-wave)
        self.phase_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # Head 3 — Magnitude
        self.magnitude_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        # Head 4 — Location (Lat, Lon, Depth)
        self.location_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 6000).

        Returns:
            Dict with keys:
              - 'detection':  (batch, 1)
              - 'phase':      (batch, 2)
              - 'magnitude':  (batch, 1)
              - 'location':   (batch, 3)
        """
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        x = self.dropout(x)

        return {
            'detection': self.detection_head(x),   # (B, 1)
            'phase':     self.phase_head(x),        # (B, 2)
            'magnitude': self.magnitude_head(x),    # (B, 1)
            'location':  self.location_head(x),     # (B, 3)
        }


class MultiTaskCNNImproved(nn.Module):
    """
    Improved multitask model with stronger regression heads.

    - Detection head stays close to baseline for compatibility/stability.
    - Phase and magnitude heads are deepened.
    - Location head is optional (disabled by default).
    """

    def __init__(self, in_channels=3, use_location_head=False):
        super(MultiTaskCNNImproved, self).__init__()
        self.use_location_head = use_location_head

        # Shared encoder (kept same as baseline to preserve behavior)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=21, stride=1, padding=10)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.fc_shared = nn.Linear(11904, 512)
        self.dropout = nn.Dropout(0.5)

        # Detection head (stable / compatible)
        self.detection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # Improved phase head
        self.phase_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # Improved magnitude head
        self.magnitude_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        if self.use_location_head:
            self.location_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 3)
            )

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        x = self.dropout(x)

        outputs = {
            'detection': self.detection_head(x),
            'phase': self.phase_head(x),
            'magnitude': self.magnitude_head(x),
        }

        if self.use_location_head:
            outputs['location'] = self.location_head(x)

        return outputs


def load_pretrained_encoder(model, pretrained_path, device):
    """
    Load pretrained SimpleCNN encoder weights into a MultiTaskCNN.

    Only loads conv1/conv2/conv3 weights. All layers remain unfrozen
    for full training with differential learning rates.

    Args:
        model:           MultiTaskCNN instance.
        pretrained_path: Path to SimpleCNN checkpoint (.pth).
        device:          Torch device.

    Returns:
        Model with pretrained encoder weights loaded.
    """
    print("Loading pretrained encoder weights...")
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)

    encoder_keys = ['conv1.weight', 'conv1.bias',
                    'conv2.weight', 'conv2.bias',
                    'conv3.weight', 'conv3.bias']

    model_dict = model.state_dict()
    pretrained = {k: v for k, v in checkpoint.items() if k in encoder_keys}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    print(f"   Loaded {len(pretrained)} pretrained layers: {list(pretrained.keys())}")
    print("   All layers UNFROZEN — full model will train with differential LR!")

    return model
