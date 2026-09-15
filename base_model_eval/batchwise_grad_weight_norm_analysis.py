import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(identity)
        return F.relu(out)


class ResNet56(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(9, 9, 9), num_classes=10):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def compute_grad_norm(model):
    total_norm_sq = 0.0
    layer_norms = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_norm = param.grad.detach().norm(2).item()
        total_norm_sq += grad_norm ** 2
        layer_norms[name] = grad_norm

    return total_norm_sq ** 0.5, layer_norms


def compute_weight_norm(model):
    total_norm_sq = 0.0
    layer_norms = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_norm = param.detach().norm(2).item()
            total_norm_sq += weight_norm ** 2
            layer_norms[name] = weight_norm

    return total_norm_sq ** 0.5, layer_norms


def build_dataloader(data_dir, batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform_train,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


def analyze_batches(checkpoint_path, data_dir, output_dir, batch_size=128, max_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet56(num_classes=10).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
    )

    loader = build_dataloader(data_dir, batch_size=batch_size)
    rows = []

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        grad_norm, layer_grad_norms = compute_grad_norm(model)
        weight_norm, layer_weight_norms = compute_weight_norm(model)

        row = {
            "batch_idx": batch_idx,
            "batch_size": images.size(0),
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "weight_norm": float(weight_norm),
        }

        for layer_name, value in layer_grad_norms.items():
            row[f"grad_{layer_name}_norm"] = float(value)

        for layer_name, value in layer_weight_norms.items():
            row[f"weight_{layer_name}_norm"] = float(value)

        rows.append(row)
        optimizer.step()

        if batch_idx % 10 == 0 or batch_idx == 1:
            print(
                f"Batch {batch_idx:03d}: "
                f"loss={loss.item():.4f}, "
                f"grad_norm={grad_norm:.4f}, "
                f"weight_norm={weight_norm:.4f}"
            )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "batchwise_grad_weight_norm.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved batchwise metrics to: {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["batch_idx"], df["grad_norm"], label="Batch Gradient Norm", linewidth=2)
    ax.plot(df["batch_idx"], df["weight_norm"], label="Weight Norm", linewidth=2)
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Norm")
    ax.set_title("Batchwise Gradient Norm and Weight Norm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "batchwise_grad_weight_norm.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {plot_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute batchwise gradient norm and weight norm for a CIFAR-10 ResNet-56 checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(
            "base_model_eval",
            "checkpoints",
            "resnet56_cifar10_final.pth",
        ),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join("data-"),
        help="Directory containing the CIFAR-10 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("base_model_eval", "analysis"),
        help="Directory where the CSV and plot will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size to use for the analysis.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on how many batches to process.",
    )

    args = parser.parse_args()

    analyze_batches(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
