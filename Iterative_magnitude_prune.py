
# ============================================================
# ITERATIVE MAGNITUDE PRUNING (IMP)
# ResNet-18 on CIFAR-10
#
# Pruning levels:
# 20% -> 40% -> 60% -> 80% -> 90% total sparsity
#
# After each pruning step:
#   1. Evaluate immediately after pruning
#   2. Fine-tune for 50 epochs
#   3. Evaluate after fine-tuning
#   4. Save checkpoints
#   5. Record metrics
# ============================================================

import os
import copy
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision.models import resnet18


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10
BATCH_SIZE = 128

# Target TOTAL sparsity levels
TARGET_SPARSITIES = [0.20, 0.40, 0.60, 0.80, 0.90]

# Fine-tuning epochs after each pruning iteration
FINETUNE_EPOCHS = 50

# Fine-tuning learning rate
FINETUNE_LR = 0.001

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Dense model checkpoint
DENSE_CHECKPOINT = "checkpoints/epoch_200.pth"

# Output directory
OUTPUT_DIR = "iterative_magnitude_pruning"

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# 2. CIFAR-10 DATASET
# ============================================================

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
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
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ============================================================
# 3. MODEL
# ============================================================

def create_model():

    model = resnet18(num_classes=NUM_CLASSES)

    # CIFAR-10 modification
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # Remove ImageNet max pooling
    model.maxpool = nn.Identity()

    return model


# ============================================================
# 4. LOAD DENSE MODEL
# ============================================================

print("\nLoading dense model...")

model = create_model()

checkpoint = torch.load(
    DENSE_CHECKPOINT,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

else:

    model.load_state_dict(checkpoint)

model = model.to(DEVICE)


# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


def evaluate(model):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ============================================================
# 6. GET PRUNABLE PARAMETERS
# ============================================================

def get_prunable_parameters(model):

    parameters = []

    for name, param in model.named_parameters():

        # Only convolutional and linear weights
        if name.endswith(".weight") and param.dim() >= 2:

            parameters.append(param)

    return parameters


# ============================================================
# 7. COUNT PARAMETERS
# ============================================================

def count_parameters(model):

    total = 0
    nonzero = 0

    for param in get_prunable_parameters(model):

        total += param.numel()

        nonzero += torch.count_nonzero(param).item()

    zero = total - nonzero

    sparsity = zero / total * 100

    return total, nonzero, zero, sparsity


# ============================================================
# 8. CREATE / UPDATE ITERATIVE PRUNING MASK
# ============================================================

def create_iterative_masks(model, target_sparsity):

    """
    Global magnitude pruning.

    IMPORTANT:
    Previously pruned weights are already zero.
    They are automatically excluded because we only
    consider currently non-zero weights.

    target_sparsity = TOTAL desired sparsity.
    """

    parameters = get_prunable_parameters(model)

    # Collect currently surviving weights
    surviving_weights = []

    for param in parameters:

        nonzero_weights = param.data[
            param.data != 0
        ].abs()

        if nonzero_weights.numel() > 0:

            surviving_weights.append(
                nonzero_weights.flatten()
            )

    if len(surviving_weights) == 0:

        raise RuntimeError(
            "No surviving weights remain."
        )

    all_surviving_weights = torch.cat(
        surviving_weights
    )

    # Current sparsity
    total_params = sum(
        p.numel() for p in parameters
    )

    current_nonzero = sum(
        torch.count_nonzero(p).item()
        for p in parameters
    )

    current_sparsity = (
        1.0 -
        current_nonzero / total_params
    )

    # Number of weights that must remain
    desired_nonzero = int(
        total_params * (1.0 - target_sparsity)
    )

    # Number of additional weights to remove
    additional_to_prune = (
        current_nonzero -
        desired_nonzero
    )

    if additional_to_prune <= 0:

        print(
            "Target sparsity already reached."
        )

        return create_current_masks(model)

    if additional_to_prune >= all_surviving_weights.numel():

        raise RuntimeError(
            "Requested sparsity is too high."
        )

    # Find threshold among surviving weights
    threshold_index = (
        additional_to_prune - 1
    )

    sorted_weights, _ = torch.sort(
        all_surviving_weights
    )

    threshold = sorted_weights[
        threshold_index
    ]

    masks = {}

    for name, param in model.named_parameters():

        if name.endswith(".weight") and param.dim() >= 2:

            # Keep previously surviving weights
            # whose magnitude is above threshold
            mask = (
                (param.data.abs() > threshold)
                & (param.data != 0)
            )

            # Handle ties at threshold carefully
            num_to_keep_needed = (
                param.data[param.data != 0].numel()
            )

            masks[name] = mask.to(
                device=param.device,
                dtype=param.dtype
            )

    return masks


# ============================================================
# 9. CREATE CURRENT MASKS
# ============================================================

def create_current_masks(model):

    masks = {}

    for name, param in model.named_parameters():

        if name.endswith(".weight") and param.dim() >= 2:

            masks[name] = (
                (param.data != 0)
                .to(
                    device=param.device,
                    dtype=param.dtype
                )
            )

    return masks


# ============================================================
# 10. APPLY MASKS
# ============================================================

def apply_masks(model, masks):

    with torch.no_grad():

        for name, param in model.named_parameters():

            if name in masks:

                param.data.mul_(
                    masks[name]
                )


# ============================================================
# 11. SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    sparsity,
    path
):

    torch.save(
        {
            "epoch": epoch,
            "sparsity": sparsity,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict()
        },
        path
    )


# ============================================================
# 12. FINE-TUNING
# ============================================================

def fine_tune(model, masks, epochs):

    optimizer = optim.SGD(
        model.parameters(),
        lr=FINETUNE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            # CRITICAL:
            # Reapply pruning mask after every
            # optimizer update.
            apply_masks(
                model,
                masks
            )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += (
                predicted.eq(labels)
                .sum()
                .item()
            )

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        if (
            (epoch + 1) == 1
            or
            (epoch + 1) % 10 == 0
            or
            (epoch + 1) == epochs
        ):

            test_loss, test_acc = evaluate(model)

            print(
                f"    FT Epoch "
                f"{epoch + 1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}%"
            )

    return optimizer


# ============================================================
# 13. DENSE BASELINE
# ============================================================

print("\n" + "=" * 70)
print("DENSE BASELINE")
print("=" * 70)

dense_loss, dense_accuracy = evaluate(model)

total_params, nonzero_params, zero_params, dense_sparsity = (
    count_parameters(model)
)

print(
    f"Test Loss:       {dense_loss:.4f}"
)

print(
    f"Test Accuracy:   {dense_accuracy:.2f}%"
)

print(
    f"Prunable Params: {total_params:,}"
)

print(
    f"Non-zero Params: {nonzero_params:,}"
)

print(
    f"Sparsity:        {dense_sparsity:.2f}%"
)


# ============================================================
# 14. ITERATIVE MAGNITUDE PRUNING
# ============================================================

results = []

# Start from the original dense model
current_model = copy.deepcopy(model)

print("\n" + "=" * 70)
print("ITERATIVE MAGNITUDE PRUNING")
print("=" * 70)

for iteration, target_sparsity in enumerate(
    TARGET_SPARSITIES,
    start=1
):

    print("\n")
    print("=" * 70)

    print(
        f"ITERATION {iteration}"
    )

    print(
        f"Target Total Sparsity: "
        f"{target_sparsity * 100:.0f}%"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # PRUNE
    # --------------------------------------------------------

    masks = create_iterative_masks(
        current_model,
        target_sparsity
    )

    apply_masks(
        current_model,
        masks
    )


    # --------------------------------------------------------
    # COUNT PARAMETERS
    # --------------------------------------------------------

    total_params, nonzero_params, zero_params, actual_sparsity = (
        count_parameters(current_model)
    )


    print(
        f"\nActual Sparsity: "
        f"{actual_sparsity:.2f}%"
    )

    print(
        f"Non-zero Parameters: "
        f"{nonzero_params:,}"
    )

    print(
        f"Zero Parameters: "
        f"{zero_params:,}"
    )


    # --------------------------------------------------------
    # EVALUATE BEFORE FINE-TUNING
    # --------------------------------------------------------

    before_loss, before_accuracy = evaluate(
        current_model
    )

    print(
        f"\nAccuracy Before Fine-Tuning: "
        f"{before_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE PRUNED CHECKPOINT
    # --------------------------------------------------------

    iteration_dir = os.path.join(
        CHECKPOINT_DIR,
        f"iteration_{iteration}_{int(target_sparsity * 100)}"
    )

    os.makedirs(
        iteration_dir,
        exist_ok=True
    )

    pruned_checkpoint = os.path.join(
        iteration_dir,
        "pruned.pth"
    )

    torch.save(
        {
            "iteration": iteration,
            "target_sparsity":
                target_sparsity,
            "actual_sparsity":
                actual_sparsity / 100.0,
            "model_state_dict":
                current_model.state_dict(),
            "masks": masks
        },
        pruned_checkpoint
    )

    print(
        f"Pruned checkpoint saved: "
        f"{pruned_checkpoint}"
    )


    # --------------------------------------------------------
    # FINE-TUNE
    # --------------------------------------------------------

    print(
        f"\nFine-tuning for "
        f"{FINETUNE_EPOCHS} epochs..."
    )

    optimizer = fine_tune(
        current_model,
        masks,
        FINETUNE_EPOCHS
    )


    # --------------------------------------------------------
    # FINAL EVALUATION
    # --------------------------------------------------------

    after_loss, after_accuracy = evaluate(
        current_model
    )

    accuracy_recovery = (
        after_accuracy -
        before_accuracy
    )

    accuracy_vs_dense = (
        after_accuracy -
        dense_accuracy
    )


    # --------------------------------------------------------
    # SAVE FINE-TUNED CHECKPOINT
    # --------------------------------------------------------

    finetuned_checkpoint = os.path.join(
        iteration_dir,
        "finetuned.pth"
    )

    save_checkpoint(
        current_model,
        optimizer,
        FINETUNE_EPOCHS,
        actual_sparsity / 100.0,
        finetuned_checkpoint
    )


    print(
        f"\nFine-tuned Accuracy: "
        f"{after_accuracy:.2f}%"
    )

    print(
        f"Accuracy Recovery: "
        f"{accuracy_recovery:+.2f}%"
    )

    print(
        f"Accuracy vs Dense: "
        f"{accuracy_vs_dense:+.2f}%"
    )

    print(
        f"Fine-tuned checkpoint saved: "
        f"{finetuned_checkpoint}"
    )


    # --------------------------------------------------------
    # RECORD RESULTS
    # --------------------------------------------------------

    results.append(
        {
            "iteration":
                iteration,

            "target_sparsity":
                target_sparsity * 100,

            "actual_sparsity":
                actual_sparsity,

            "total_parameters":
                total_params,

            "nonzero_parameters":
                nonzero_params,

            "zero_parameters":
                zero_params,

            "accuracy_before_ft":
                before_accuracy,

            "accuracy_after_ft":
                after_accuracy,

            "accuracy_recovery":
                accuracy_recovery,

            "accuracy_vs_dense":
                accuracy_vs_dense
        }
    )


# ============================================================
# 15. SAVE RESULTS CSV
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

csv_path = os.path.join(
    OUTPUT_DIR,
    "iterative_pruning_results.csv"
)

fieldnames = [
    "iteration",
    "target_sparsity",
    "actual_sparsity",
    "total_parameters",
    "nonzero_parameters",
    "zero_parameters",
    "accuracy_before_ft",
    "accuracy_after_ft",
    "accuracy_recovery",
    "accuracy_vs_dense"
]


with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"{'Iter':<6}"
    f"{'Sparsity':<12}"
    f"{'Non-zero':<15}"
    f"{'Before FT':<13}"
    f"{'After FT':<13}"
    f"{'Recovery':<13}"
    f"{'vs Dense':<13}"
)

print("-" * 85)

for r in results:

    print(
        f"{r['iteration']:<6}"
        f"{r['actual_sparsity']:<12.2f}%"
        f"{r['nonzero_parameters']:<15,}"
        f"{r['accuracy_before_ft']:<13.2f}%"
        f"{r['accuracy_after_ft']:<13.2f}%"
        f"{r['accuracy_recovery']:+.2f}%"
        f"{r['accuracy_vs_dense']:+.2f}%"
    )

print("-" * 85)

print(
    f"\nResults CSV saved to:\n"
    f"{csv_path}"
)


# ============================================================
# 16. PLOTS
# ============================================================

sparsities = [
    r["actual_sparsity"]
    for r in results
]

before_ft = [
    r["accuracy_before_ft"]
    for r in results
]

after_ft = [
    r["accuracy_after_ft"]
    for r in results
]

recovery = [
    r["accuracy_recovery"]
    for r in results
]

nonzero = [
    r["nonzero_parameters"]
    for r in results
]


# ------------------------------------------------------------
# Accuracy vs Sparsity
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    sparsities,
    before_ft,
    marker="o",
    label="Before Fine-Tuning"
)

plt.plot(
    sparsities,
    after_ft,
    marker="o",
    label="After Fine-Tuning"
)

plt.axhline(
    dense_accuracy,
    linestyle="--",
    label="Dense Baseline"
)

plt.xlabel("Sparsity (%)")
plt.ylabel("Test Accuracy (%)")
plt.title(
    "Iterative Magnitude Pruning: Accuracy vs Sparsity"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_vs_sparsity.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Accuracy Recovery
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    sparsities,
    recovery,
    marker="o"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy Recovery (%)")
plt.title(
    "Accuracy Recovery After Fine-Tuning"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_recovery_vs_sparsity.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Non-zero Parameters
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    sparsities,
    nonzero,
    marker="o"
)

plt.xlabel("Sparsity (%)")
plt.ylabel("Non-zero Parameters")
plt.title(
    "Remaining Parameters vs Sparsity"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "nonzero_parameters_vs_sparsity.png"
    ),
    dpi=300
)

plt.close()


print("\nPlots saved.")

print(
    f"\nExperiment completed successfully."
)

print(
    f"Output directory:\n"
    f"{OUTPUT_DIR}"
)
