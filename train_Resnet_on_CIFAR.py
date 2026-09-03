# ============================================================
# RESNET-18 CIFAR-10 BASELINE TRAINING
# WITH CHECKPOINT SAVING
# ============================================================

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torchvision.models import resnet18


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)


# ============================================================
# 2. CREATE CHECKPOINT FOLDER
# ============================================================

os.makedirs("checkpoints", exist_ok=True)

print("Checkpoint directory:")
print(os.path.abspath("checkpoints"))


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
# 4. MODEL
# ============================================================

model = resnet18(num_classes=10)

# CIFAR-10: 32x32 images
model.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

# Remove ImageNet-style max pooling
model.maxpool = nn.Identity()

model = model.to(device)


# ============================================================
# 5. LOSS + OPTIMIZER
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
    scheduler,
    epoch,
    path
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict()
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


    # Save current weights for next epoch
    previous_weights = save_weights(model)


    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler.step()


    # --------------------------------------------------------
    # Current learning rate
    # --------------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    if epoch in [50, 100, 150, 200]:

        checkpoint_path = (
            f"checkpoints/epoch_{epoch:03d}.pth"
        )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
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
    "checkpoints/resnet18_cifar10_final.pth"
)

save_checkpoint(
    model,
    optimizer,
    scheduler,
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

print("  checkpoints/epoch_050.pth")
print("  checkpoints/epoch_100.pth")
print("  checkpoints/epoch_150.pth")
print("  checkpoints/epoch_200.pth")
print("  checkpoints/resnet18_cifar10_final.pth")