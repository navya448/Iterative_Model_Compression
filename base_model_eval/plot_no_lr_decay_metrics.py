# ============================================================
# PLOTS FOR RESNET-56 / CIFAR-10 — NO LR DECAY TRAINING RUN
# Reads resnet56_cifar10_no_lr_decay_training_metrics.csv and
# saves plots to plots_no_lr_decay/
# ============================================================

import os

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(
    PROJECT_DIR,
    "resnet56_cifar10_no_lr_decay_training_metrics.csv",
)
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots_no_lr_decay")

os.makedirs(PLOTS_DIR, exist_ok=True)

metrics = pd.read_csv(METRICS_PATH)

# Validated categorical palette (dataviz skill reference palette)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

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


def make_plot(filename, title, ylabel, series, colors):
    plt.figure(figsize=(10, 5))

    for (column, label), color in zip(series, colors):
        plt.plot(
            metrics["epoch"],
            metrics[column],
            label=label,
            color=color,
            linewidth=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series) > 1:
        plt.legend(frameon=False)
    plt.grid(True, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


plot_definitions = [
    (
        "loss_vs_epoch.png",
        "Loss vs Epoch (No LR Decay)",
        "Loss",
        [("train_loss", "Train Loss"), ("val_loss", "Validation Loss")],
        [BLUE, ORANGE],
    ),
    (
        "accuracy_vs_epoch.png",
        "Accuracy vs Epoch (No LR Decay)",
        "Accuracy (%)",
        [("train_accuracy", "Train Accuracy"), ("val_accuracy", "Validation Accuracy")],
        [BLUE, ORANGE],
    ),
    (
        "gradient_norm_vs_epoch.png",
        "Gradient Norm vs Epoch (No LR Decay)",
        "Gradient Norm",
        [("gradient_norm", "Gradient Norm")],
        [BLUE],
    ),
    (
        "weight_norm_vs_epoch.png",
        "Weight Norm vs Epoch (No LR Decay)",
        "Weight Norm",
        [("weight_norm", "Weight Norm")],
        [AQUA],
    ),
    (
        "weight_change_vs_epoch.png",
        "Weight Change vs Epoch (No LR Decay)",
        "Weight Change",
        [("weight_change", "Weight Change")],
        [ORANGE],
    ),
]

for filename, title, ylabel, series, colors in plot_definitions:
    make_plot(filename, title, ylabel, series, colors)

print(f"Saved {len(plot_definitions)} plots to: {os.path.abspath(PLOTS_DIR)}")
