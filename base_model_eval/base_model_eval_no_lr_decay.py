# ============================================================
# RESNET-56 CIFAR-10 BASELINE TRAINING
# WITHOUT LEARNING-RATE DECAY
# ============================================================

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt


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


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints_no_lr_decay")
DATA_DIR = os.path.join(PROJECT_DIR, "..", "data-")
METRICS_PATH = os.path.join(
    PROJECT_DIR,
    "resnet56_cifar10_no_lr_decay_training_metrics.csv",
)
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots_no_lr_decay")

# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)


# ============================================================
# 2. CREATE CHECKPOINT FOLDER
# ============================================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Checkpoint directory:")
print(os.path.abspath(CHECKPOINT_DIR))


# ============================================================
# 3. CIFAR-10 DATASET
# ============================================================

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform_test
)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 4. MODEL
# ============================================================

model = ResNet56(num_classes=10).to(device)


# ============================================================
# 5. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# ============================================================
# 6. METRIC FUNCTIONS
# ============================================================

def calculate_gradient_norm(model):

    total_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            param_norm = parameter.grad.detach().norm(2)

            total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


def calculate_weight_norm(model):

    total_norm = 0.0

    for parameter in model.parameters():

        param_norm = parameter.detach().norm(2)

        total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


def calculate_weight_change(model, previous_weights):

    total_change = 0.0

    for parameter in model.parameters():

        if parameter.requires_grad:

            previous = previous_weights[id(parameter)]

            change = (
                parameter.detach() - previous
            ).norm(2)

            total_change += change.item() ** 2

    return total_change ** 0.5


def save_weights(model):

    weights = {}

    for parameter in model.parameters():

        if parameter.requires_grad:

            weights[id(parameter)] = (
                parameter.detach().clone()
            )

    return weights


# ============================================================
# 7. CHECKPOINT FUNCTION
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    path
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict()
    }

    torch.save(checkpoint, path)

    print()
    print("==============================================")
    print(f"CHECKPOINT SAVED - EPOCH {epoch}")
    print("==============================================")
    print(os.path.abspath(path))
    print()


# ============================================================
# 8. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += (
            predicted.eq(labels).sum().item()
        )

    epoch_loss = running_loss / total

    epoch_accuracy = (
        100.0 * correct / total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# 9. VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += (
                predicted.eq(labels).sum().item()
            )

    loss = running_loss / total

    accuracy = (
        100.0 * correct / total
    )

    return loss, accuracy


# ============================================================
# 10. TRAINING
# ============================================================

NUM_EPOCHS = 200

previous_weights = save_weights(model)
history = []


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_loss, test_accuracy = validate(
        model,
        test_loader,
        criterion
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    gradient_norm = calculate_gradient_norm(
        model
    )

    weight_norm = calculate_weight_norm(
        model
    )

    weight_change = calculate_weight_change(
        model,
        previous_weights
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": test_loss,
        "val_accuracy": test_accuracy,
        "gradient_norm": gradient_norm,
        "weight_norm": weight_norm,
        "weight_change": weight_change
    })


    # Save current weights for next epoch
    previous_weights = save_weights(model)


    # --------------------------------------------------------
    # Current learning rate
    # --------------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    if epoch in [50, 100, 150, 200]:

        checkpoint_path = (
            os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth")
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            checkpoint_path
        )


    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Test Loss: {test_loss:.4f} "
        f"Test Acc: {test_accuracy:.2f}% "
        f"Grad Norm: {gradient_norm:.4f} "
        f"Weight Norm: {weight_norm:.4f} "
        f"Weight Change: {weight_change:.4f} "
        f"LR: {current_lr:.6f}"
    )


# ============================================================
# 11. FINAL CHECKPOINT
# ============================================================

final_checkpoint_path = (
    os.path.join(CHECKPOINT_DIR, "resnet56_cifar10_no_lr_decay_final.pth")
)

save_checkpoint(
    model,
    optimizer,
    NUM_EPOCHS,
    final_checkpoint_path
)


# ============================================================
# 12. FINAL RESULT
# ============================================================

print()
print("==============================================")
print("TRAINING COMPLETE")
print("==============================================")

print(
    f"Final Test Loss: {test_loss:.4f}"
)

print(
    f"Final Test Accuracy: {test_accuracy:.2f}%"
)

print()
print("Saved checkpoints:")

print(f"  {os.path.join(CHECKPOINT_DIR, 'epoch_050.pth')}")
print(f"  {os.path.join(CHECKPOINT_DIR, 'epoch_100.pth')}")
print(f"  {os.path.join(CHECKPOINT_DIR, 'epoch_150.pth')}")
print(f"  {os.path.join(CHECKPOINT_DIR, 'epoch_200.pth')}")
print(
    f"  {os.path.join(CHECKPOINT_DIR, 'resnet56_cifar10_no_lr_decay_final.pth')}"
)


# ============================================================
# 13. SAVE METRICS PLOTS
# ============================================================

pd.DataFrame(history).to_csv(METRICS_PATH, index=False)
metrics = pd.DataFrame(history)
os.makedirs(PLOTS_DIR, exist_ok=True)

plot_definitions = [
    (
        "loss_vs_epoch.png",
        "Loss vs Epoch",
        "Loss",
        [("train_loss", "Train Loss"), ("val_loss", "Validation Loss")]
    ),
    (
        "accuracy_vs_epoch.png",
        "Accuracy vs Epoch",
        "Accuracy (%)",
        [("train_accuracy", "Train Accuracy"), ("val_accuracy", "Validation Accuracy")]
    ),
    (
        "gradient_weight_norm.png",
        "Gradient and Weight Norm",
        "Norm",
        [("gradient_norm", "Gradient Norm"), ("weight_norm", "Weight Norm")]
    ),
    (
        "weight_change_vs_epoch.png",
        "Weight Change vs Epoch",
        "Weight Change",
        [("weight_change", "Weight Change")]
    )
]

for filename, title, ylabel, series in plot_definitions:
    plt.figure(figsize=(10, 5))

    for column, label in series:
        plt.plot(metrics["epoch"], metrics[column], label=label)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()

print(f"Metrics plots saved to: {os.path.abspath(PLOTS_DIR)}")