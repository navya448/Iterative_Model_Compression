import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

from torchvision.models import resnet18


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# ============================================================
# 2. CIFAR-10 DATASET
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
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
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
# 3. MODEL
# ============================================================

model = resnet18(num_classes=10)

# CIFAR-10 uses 32x32 images, so modify the first layer
model.conv1 = nn.Conv2d(
    3, 64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

# Remove ImageNet-style max pooling
model.maxpool = nn.Identity()

model = model.to(device)


# ============================================================
# 4. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[100, 150],
    gamma=0.1
)


# ============================================================
# 5. FUNCTIONS FOR TRACKING
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
            weights[id(parameter)] = parameter.detach().clone()

    return weights


# ============================================================
# 6. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 7. VALIDATION FUNCTION
# ============================================================

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = running_loss / total
    accuracy = 100.0 * correct / total

    return loss, accuracy


# ============================================================
# 8. TRAINING LOOP + LOGGING
# ============================================================

NUM_EPOCHS = 200

history = []

previous_weights = save_weights(model)


for epoch in range(1, NUM_EPOCHS + 1):

    # -----------------------------
    # Train
    # -----------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )


    # -----------------------------
    # Validation
    # -----------------------------

    val_loss, val_accuracy = validate(
        model,
        test_loader,
        criterion
    )


    # -----------------------------
    # Gradient norm
    # -----------------------------

    gradient_norm = calculate_gradient_norm(model)


    # -----------------------------
    # Weight norm
    # -----------------------------

    weight_norm = calculate_weight_norm(model)


    # -----------------------------
    # Weight change
    # -----------------------------

    weight_change = calculate_weight_change(
        model,
        previous_weights
    )


    # Save current weights for next epoch
    previous_weights = save_weights(model)


    # -----------------------------
    # Store everything
    # -----------------------------

    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "gradient_norm": gradient_norm,

        "weight_norm": weight_norm,

        "weight_change": weight_change

    })


    scheduler.step()


    print(
        f"Epoch [{epoch}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}% "
        f"Grad Norm: {gradient_norm:.4f} "
        f"Weight Norm: {weight_norm:.4f} "
        f"Weight Change: {weight_change:.4f}"
    )



# ============================================================
# 9. SAVE RESULTS
# ============================================================

df = pd.DataFrame(history)

# Save training metrics
df.to_csv(
    "resnet18_cifar10_training_metrics.csv",
    index=False
)

print("\nTraining complete.")
print(df.head())


# ============================================================
# 10. SAVE PLOTS
# ============================================================

# -----------------------------
# Plot 1: Loss
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df["epoch"],
    df["train_loss"],
    label="Train Loss"
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "loss_vs_epoch.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -----------------------------
# Plot 2: Accuracy
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df["epoch"],
    df["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    df["epoch"],
    df["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epoch")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "accuracy_vs_epoch.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -----------------------------
# Plot 3: Gradient and Weight Norm
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df["epoch"],
    df["gradient_norm"],
    label="Gradient Norm"
)

plt.plot(
    df["epoch"],
    df["weight_norm"],
    label="Weight Norm"
)

plt.xlabel("Epoch")
plt.ylabel("Norm")
plt.title("Gradient and Weight Norm")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "gradient_weight_norm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -----------------------------
# Plot 4: Weight Change
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df["epoch"],
    df["weight_change"],
    label="Weight Change"
)

plt.xlabel("Epoch")
plt.ylabel("Weight Change")
plt.title("Weight Change vs Epoch")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "weight_change_vs_epoch.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


print("\nPlots saved successfully:")
print("1. loss_vs_epoch.png")
print("2. accuracy_vs_epoch.png")
print("3. gradient_weight_norm.png")
print("4. weight_change_vs_epoch.png")
print("5. resnet18_cifar10_training_metrics.csv")