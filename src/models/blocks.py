"""
Reusable convolutional building blocks for seismic models.

Provides:
  - ConvBlock : Single Conv1d + ReLU + MaxPool layer
  - ConvEncoder : Full 3-layer encoder stack (for new models)

Note:
    SimpleCNN and MultiTaskCNN keep their original flat attribute names
    (conv1, conv2, conv3, pool1, pool2, pool3) for backward compatibility
    with existing checkpoints. Use ConvEncoder only for new architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Single 1D convolutional block: Conv1d → ReLU → MaxPool1d.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Convolution kernel size.
        padding:      Convolution padding.
        pool_kernel:  MaxPool kernel size.
        pool_stride:  MaxPool stride.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 pool_kernel=4, pool_stride=4):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=1, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        return self.pool(F.relu(self.conv(x)))


class ConvEncoder(nn.Module):
    """
    3-layer Conv1d encoder matching the STEAD seismic model architecture.

    Input:  (batch, 3, 6000)
    Output: (batch, 11904)  — flattened feature vector

    Note:
        This uses different state_dict keys than SimpleCNN/MultiTaskCNN
        (e.g., blocks.0.conv.weight vs conv1.weight), so it is NOT
        checkpoint-compatible with existing trained models. Use this for
        building new architectures only.
    """

    def __init__(self, in_channels=3):
        super(ConvEncoder, self).__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, 32,  kernel_size=21, padding=10),
            ConvBlock(32,          64,  kernel_size=15, padding=7),
            ConvBlock(64,          128, kernel_size=11, padding=5),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x.view(x.size(0), -1)  # Flatten to (batch, 11904)
