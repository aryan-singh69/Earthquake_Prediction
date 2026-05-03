"""
Plotting utilities — confusion matrix, waveform visualization, training curves.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_waveform(data, title="Seismic Waveform", output_path=None):
    """
    Plot a 3-component seismic waveform.

    Args:
        data:        Array of shape (3, 6000) or (6000, 3).
        title:       Plot title.
        output_path: If provided, save to this path instead of showing.
    """
    if data.shape == (6000, 3):
        data = data.T

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    labels = ["East-West", "North-South", "Vertical"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(data[i], linewidth=0.5, color=f"C{i}")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample (100 Hz)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses, output_path=None):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of per-epoch training losses.
        val_losses:   List of per-epoch validation losses.
        output_path:  If provided, save to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "o-", label="Train Loss")
    ax.plot(epochs, val_losses,   "o-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
