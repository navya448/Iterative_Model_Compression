# ============================================================
# OVERLAY COMPARISON PLOTS
# WITH LR DECAY (MultiStepLR) vs. NO LR DECAY (constant LR=0.1)
# ResNet-56 / CIFAR-10
# ============================================================

import os

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DECAY_PATH = os.path.join(PROJECT_DIR, "resnet56_cifar10_training_metrics.csv")
NO_DECAY_PATH = os.path.join(
    PROJECT_DIR, "resnet56_cifar10_no_lr_decay_training_metrics.csv"
)
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots_comparison")

os.makedirs(PLOTS_DIR, exist_ok=True)

decay = pd.read_csv(DECAY_PATH)
no_decay = pd.read_csv(NO_DECAY_PATH)

# Validated categorical palette (dataviz skill reference palette)
BLUE = "#2a78d6"    # With LR Decay
ORANGE = "#eb6834"  # No LR Decay

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "grid.color": "#e1e0d9",
    "font.size": 11,
})


def make_comparison_plot(filename, title, ylabel, column, logy=False):
    plt.figure(figsize=(10, 5))

    plt.plot(
        decay["epoch"], decay[column],
        label="With LR Decay", color=BLUE, linewidth=2,
    )
    plt.plot(
        no_decay["epoch"], no_decay[column],
        label="No LR Decay", color=ORANGE, linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.legend(frameon=False)
    plt.grid(True, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


plot_definitions = [
    (
        "val_loss_comparison.png",
        "Validation Loss vs Epoch: LR Decay vs No LR Decay",
        "Validation Loss",
        "val_loss",
        False,
    ),
    (
        "val_accuracy_comparison.png",
        "Validation Accuracy vs Epoch: LR Decay vs No LR Decay",
        "Validation Accuracy (%)",
        "val_accuracy",
        False,
    ),
    (
        "train_loss_comparison.png",
        "Train Loss vs Epoch: LR Decay vs No LR Decay",
        "Train Loss",
        "train_loss",
        True,
    ),
    (
        "train_accuracy_comparison.png",
        "Train Accuracy vs Epoch: LR Decay vs No LR Decay",
        "Train Accuracy (%)",
        "train_accuracy",
        False,
    ),
    (
        "gradient_norm_comparison.png",
        "Gradient Norm vs Epoch: LR Decay vs No LR Decay",
        "Gradient Norm",
        "gradient_norm",
        False,
    ),
    (
        "weight_norm_comparison.png",
        "Weight Norm vs Epoch: LR Decay vs No LR Decay",
        "Weight Norm",
        "weight_norm",
        False,
    ),
    (
        "weight_change_comparison.png",
        "Weight Change vs Epoch: LR Decay vs No LR Decay",
        "Weight Change",
        "weight_change",
        True,
    ),
]

for filename, title, ylabel, column, logy in plot_definitions:
    make_comparison_plot(filename, title, ylabel, column, logy)

print(f"Saved {len(plot_definitions)} comparison plots to: {os.path.abspath(PLOTS_DIR)}")
