"""
SimpleCNN — Binary earthquake detection model.

Architecture:
    3-layer Conv1d encoder → FC layers → single sigmoid output.

This model uses the same encoder architecture as MultiTaskCNN,
so pretrained encoder weights can be transferred between them.

Checkpoint-compatible with: best_model.pth, baseline_cnn.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple 1D CNN for binary earthquake detection.

    Input:  (batch, 3, 6000) — 3-component seismic waveform
    Output: (batch, 1)       — detection logit (apply sigmoid for probability)

    Args:
        in_channels: Number of input channels (default: 3 for 3-component seismogram).
        dropout:     Dropout probability (default: 0.5).
    """

    def __init__(self, in_channels=3, dropout=0.5):
        super(SimpleCNN, self).__init__()

        # ── Shared encoder (same as MultiTaskCNN) ─────────────
        self.conv1 = nn.Conv1d(in_channels, 32,  kernel_size=21, stride=1, padding=10)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(32,  64,  kernel_size=15, stride=1, padding=7)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(64,  128, kernel_size=11, stride=1, padding=5)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        # ── Classifier head ───────────────────────────────────
        # Flatten dim: 128 * (6000 / 4 / 4 / 4) = 128 * 93 = 11904
        self.fc1     = nn.Linear(11904, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(512, 128)
        self.out     = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 6000).

        Returns:
            Logit tensor of shape (batch, 1).
        """
        # Encoder
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.pool3(x)

        # Flatten + classify
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
