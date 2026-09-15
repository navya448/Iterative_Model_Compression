# ============================================================
# GLOBAL MAGNITUDE PRUNING + FINE-TUNING EXPERIMENT
# CIFAR-10 / RESNET-56
# ============================================================

import os
import copy
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


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
DATA_DIR = os.path.join(PROJECT_DIR, "..", "data-")

# ============================================================
# 1. SETTINGS
# ============================================================

# Trained dense model checkpoint
CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "..",
    "base_model_eval",
    "checkpoints",
    "resnet56_cifar10_final.pth"
)

# Sparsity levels to test
SPARSITY_LEVELS = [0.20, 0.40, 0.60, 0.80, 0.90]

# Number of fine-tuning epochs
FINETUNE_EPOCHS = 50

# Fine-tuning a deeper residual network is usually more stable
# with a smaller learning rate than the original ResNet-18 run.
FINETUNE_LR = 0.0005

# Batch size
BATCH_SIZE = 128

# Output directory
OUTPUT_DIR = PROJECT_DIR
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

os.makedirs(
    os.path.join(OUTPUT_DIR, "checkpoints"),
    exist_ok=True
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. CIFAR-10 DATASET
# ============================================================

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 4. MODEL CREATION
# ============================================================

def create_model():

    model = ResNet56(num_classes=10)
    return model.to(device)


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

def load_trained_model():

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{os.path.abspath(CHECKPOINT_PATH)}\n\n"
            f"Please check CHECKPOINT_PATH."
        )

    model = create_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    # Checkpoint created by the Stage 0 training script
    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        checkpoint_epoch = checkpoint.get(
            "epoch",
            "unknown"
        )

        print(
            f"Loaded checkpoint from epoch: "
            f"{checkpoint_epoch}"
        )

    else:

        # Also support a checkpoint containing
        # only the model state dictionary.
        model.load_state_dict(checkpoint)

        print(
            "Loaded model state dictionary."
        )

    return model


# ============================================================
# 6. EVALUATION
# ============================================================

criterion = nn.CrossEntropyLoss()


def evaluate(model):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

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

    average_loss = running_loss / total

    accuracy = (
        100.0 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# 7. IDENTIFY PRUNABLE WEIGHTS
# ============================================================

def get_prunable_parameters(model):

    parameters = []

    for name, parameter in model.named_parameters():

        # Prune only actual weight tensors.
        #
        # This includes:
        #   Conv2d weights
        #   Linear weights
        #
        # It excludes:
        #   biases
        #   BatchNorm parameters
        #
        if parameter.requires_grad and name.endswith(".weight"):

            # Do not accidentally include BatchNorm weights.
            if parameter.dim() >= 2:

                parameters.append(
                    (name, parameter)
                )

    return parameters


# ============================================================
# 8. CALCULATE CURRENT SPARSITY
# ============================================================

def calculate_sparsity(model):

    total_parameters = 0
    zero_parameters = 0

    for name, parameter in get_prunable_parameters(model):

        total_parameters += parameter.numel()

        zero_parameters += (
            parameter.detach() == 0
        ).sum().item()

    sparsity = (
        100.0 * zero_parameters / total_parameters
    )

    nonzero_parameters = (
        total_parameters - zero_parameters
    )

    return (
        total_parameters,
        nonzero_parameters,
        zero_parameters,
        sparsity
    )


# ============================================================
# 9. CREATE GLOBAL MAGNITUDE MASK
# ============================================================

def create_global_magnitude_masks(
    model,
    sparsity
):

    prunable_parameters = (
        get_prunable_parameters(model)
    )

    # Collect absolute magnitudes from all
    # prunable weights across the entire network.
    all_weights = []

    for name, parameter in prunable_parameters:

        all_weights.append(
            parameter.detach().abs().flatten()
        )

    all_weights = torch.cat(all_weights)

    # Number of weights that should become zero
    number_to_prune = int(
        sparsity * all_weights.numel()
    )

    if number_to_prune == 0:

        return {
            name: torch.ones_like(parameter)
            for name, parameter
            in prunable_parameters
        }

    # Find threshold.
    #
    # kth smallest magnitude becomes the threshold.
    sorted_weights, _ = torch.sort(
        all_weights
    )

    threshold = sorted_weights[
        number_to_prune - 1
    ]

    masks = {}

    for name, parameter in prunable_parameters:

        mask = (
            parameter.detach().abs() > threshold
        ).float()

        masks[name] = mask.to(device)

    return masks


# ============================================================
# 10. APPLY MASK
# ============================================================

def apply_masks(model, masks):

    for name, parameter in model.named_parameters():

        if name in masks:

            parameter.data.mul_(
                masks[name]
            )


# ============================================================
# 11. TRAINING FOR ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    masks
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

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

        # IMPORTANT:
        # Re-apply pruning mask after every
        # optimizer update.
        apply_masks(
            model,
            masks
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += (
            predicted.eq(labels).sum().item()
        )

    average_loss = running_loss / total

    accuracy = (
        100.0 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# 12. SAVE CHECKPOINT
# ============================================================

def save_experiment_checkpoint(
    model,
    masks,
    sparsity,
    epoch,
    accuracy,
    path
):

    checkpoint = {

        "sparsity": sparsity,

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "masks": masks,

        "test_accuracy": accuracy
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Saved checkpoint: "
        f"{os.path.abspath(path)}"
    )


# ============================================================
# 13. LOAD DENSE MODEL
# ============================================================

print()
print("==============================================")
print("LOADING TRAINED MODEL")
print("==============================================")

base_model = load_trained_model()

base_model.eval()


# ============================================================
# 14. VERIFY DENSE MODEL
# ============================================================

dense_loss, dense_accuracy = evaluate(
    base_model
)

(
    total_params,
    nonzero_params,
    zero_params,
    dense_sparsity
) = calculate_sparsity(
    base_model
)

print()
print("==============================================")
print("DENSE MODEL")
print("==============================================")

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
# 15. RESULTS LIST
# ============================================================

results = []


# Add dense baseline
results.append({

    "sparsity": 0.0,

    "total_parameters": total_params,

    "nonzero_parameters": nonzero_params,

    "zero_parameters": zero_params,

    "accuracy_before_finetuning": dense_accuracy,

    "accuracy_after_finetuning": dense_accuracy,

    "accuracy_recovered": 0.0
})


# ============================================================
# 16. RUN PRUNING EXPERIMENTS
# ============================================================

for sparsity in SPARSITY_LEVELS:

    sparsity_percent = int(
        sparsity * 100
    )

    print()
    print()
    print("==============================================")
    print(
        f"PRUNING EXPERIMENT: "
        f"{sparsity_percent}% SPARSITY"
    )
    print("==============================================")


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Every experiment starts from a fresh copy of the
    # ORIGINAL dense model.
    # --------------------------------------------------------

    model = copy.deepcopy(
        base_model
    ).to(device)


    # --------------------------------------------------------
    # Create global magnitude masks
    # --------------------------------------------------------

    masks = create_global_magnitude_masks(
        model,
        sparsity
    )


    # --------------------------------------------------------
    # Apply pruning
    # --------------------------------------------------------

    apply_masks(
        model,
        masks
    )


    # --------------------------------------------------------
    # Calculate actual sparsity
    # --------------------------------------------------------

    (
        total_params,
        nonzero_params,
        zero_params,
        actual_sparsity
    ) = calculate_sparsity(
        model
    )


    # --------------------------------------------------------
    # Evaluate immediately after pruning
    # --------------------------------------------------------

    pruned_loss, pruned_accuracy = evaluate(
        model
    )


    print()
    print(
        f"Target sparsity: "
        f"{sparsity_percent}%"
    )

    print(
        f"Actual sparsity: "
        f"{actual_sparsity:.2f}%"
    )

    print(
        f"Non-zero parameters: "
        f"{nonzero_params:,}"
    )

    print(
        f"Accuracy BEFORE fine-tuning: "
        f"{pruned_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # Save model immediately after pruning
    # --------------------------------------------------------

    pruning_directory = os.path.join(
        OUTPUT_DIR,
        "checkpoints",
        f"pruning_{sparsity_percent}"
    )

    os.makedirs(
        pruning_directory,
        exist_ok=True
    )


    pruned_checkpoint_path = os.path.join(
        pruning_directory,
        "pruned.pth"
    )


    save_experiment_checkpoint(
        model=model,
        masks=masks,
        sparsity=sparsity,
        epoch=0,
        accuracy=pruned_accuracy,
        path=pruned_checkpoint_path
    )


    # --------------------------------------------------------
    # Fine-tuning optimizer
    # --------------------------------------------------------

    optimizer = optim.SGD(
        model.parameters(),
        lr=FINETUNE_LR,
        momentum=0.9,
        weight_decay=5e-4
    )


    # --------------------------------------------------------
    # Fine-tuning
    # --------------------------------------------------------

    print()
    print(
        f"Starting {FINETUNE_EPOCHS}-epoch fine-tuning..."
    )


    for epoch in range(
        1,
        FINETUNE_EPOCHS + 1
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                optimizer,
                masks
            )
        )


        test_loss, test_accuracy = evaluate(
            model
        )


        print(
            f"Fine-tuning Epoch "
            f"[{epoch}/{FINETUNE_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_accuracy:.2f}%"
        )


    # --------------------------------------------------------
    # Final sparsity check
    # --------------------------------------------------------

    (
        total_params,
        nonzero_params,
        zero_params,
        final_sparsity
    ) = calculate_sparsity(
        model
    )


    # --------------------------------------------------------
    # Save final fine-tuned model
    # --------------------------------------------------------

    finetuned_checkpoint_path = os.path.join(
        pruning_directory,
        "finetuned.pth"
    )


    save_experiment_checkpoint(
        model=model,
        masks=masks,
        sparsity=sparsity,
        epoch=FINETUNE_EPOCHS,
        accuracy=test_accuracy,
        path=finetuned_checkpoint_path
    )


    # --------------------------------------------------------
    # Calculate recovery
    # --------------------------------------------------------

    accuracy_recovered = (
        test_accuracy - pruned_accuracy
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({

        "sparsity": sparsity * 100,

        "total_parameters": total_params,

        "nonzero_parameters": nonzero_params,

        "zero_parameters": zero_params,

        "accuracy_before_finetuning":
            pruned_accuracy,

        "accuracy_after_finetuning":
            test_accuracy,

        "accuracy_recovered":
            accuracy_recovered
    })


    # --------------------------------------------------------
    # Experiment summary
    # --------------------------------------------------------

    print()
    print("----------------------------------------------")
    print(
        f"{sparsity_percent}% PRUNING COMPLETE"
    )
    print("----------------------------------------------")

    print(
        f"Actual sparsity: "
        f"{final_sparsity:.2f}%"
    )

    print(
        f"Accuracy before FT: "
        f"{pruned_accuracy:.2f}%"
    )

    print(
        f"Accuracy after FT:  "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Accuracy recovered: "
        f"{accuracy_recovered:+.2f}%"
    )


# ============================================================
# 17. SAVE RESULTS TO CSV
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "magnitude_pruning_results.csv"
)


with open(
    results_file,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "sparsity",
            "total_parameters",
            "nonzero_parameters",
            "zero_parameters",
            "accuracy_before_finetuning",
            "accuracy_after_finetuning",
            "accuracy_recovered"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# 18. FINAL RESULTS
# ============================================================

print()
print()
print("============================================================")
print("MAGNITUDE PRUNING + FINE-TUNING RESULTS")
print("============================================================")

print(
    f"{'Sparsity':>10} | "
    f"{'Non-zero':>12} | "
    f"{'Before FT':>12} | "
    f"{'After FT':>12} | "
    f"{'Recovered':>12}"
)

print("-" * 70)


for result in results:

    print(
        f"{result['sparsity']:>9.0f}% | "
        f"{result['nonzero_parameters']:>12,} | "
        f"{result['accuracy_before_finetuning']:>11.2f}% | "
        f"{result['accuracy_after_finetuning']:>11.2f}% | "
        f"{result['accuracy_recovered']:>+11.2f}%"
    )


print()
print(
    f"Results saved to:\n"
    f"{os.path.abspath(results_file)}"
)

print()
print("Experiment complete.")

# ============================================================
# 19. SAVE PLOTS
# ============================================================

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Extract results
# ------------------------------------------------------------

sparsities = [
    result["sparsity"]
    for result in results
]

accuracy_before = [
    result["accuracy_before_finetuning"]
    for result in results
]

accuracy_after = [
    result["accuracy_after_finetuning"]
    for result in results
]

accuracy_recovered = [
    result["accuracy_recovered"]
    for result in results
]

nonzero_parameters = [
    result["nonzero_parameters"]
    for result in results
]


# ============================================================
# PLOT 1: ACCURACY VS SPARSITY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    sparsities,
    accuracy_before,
    marker="o",
    label="Before Fine-Tuning"
)

plt.plot(
    sparsities,
    accuracy_after,
    marker="o",
    label="After Fine-Tuning"
)

plt.xlabel("Sparsity (%)")
plt.ylabel("Test Accuracy (%)")

plt.title(
    "Test Accuracy vs. Model Sparsity"
)

plt.xticks(sparsities)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "accuracy_vs_sparsity.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 2: ACCURACY RECOVERY VS SPARSITY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    sparsities,
    accuracy_recovered,
    marker="o"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Sparsity (%)")

plt.ylabel(
    "Accuracy Change After Fine-Tuning (%)"
)

plt.title(
    "Accuracy Recovery from Fine-Tuning"
)

plt.xticks(sparsities)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "accuracy_recovery_vs_sparsity.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 3: NON-ZERO PARAMETERS VS SPARSITY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    sparsities,
    nonzero_parameters,
    marker="o"
)

plt.xlabel("Sparsity (%)")

plt.ylabel("Number of Non-Zero Parameters")

plt.title(
    "Remaining Parameters vs. Model Sparsity"
)

plt.xticks(sparsities)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "nonzero_parameters_vs_sparsity.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. PRINT PLOT LOCATIONS
# ============================================================

print()
print("============================================================")
print("PLOTS SAVED")
print("============================================================")

print(
    os.path.abspath(
        os.path.join(
            PLOTS_DIR,
            "accuracy_vs_sparsity.png"
        )
    )
)

print(
    os.path.abspath(
        os.path.join(
            PLOTS_DIR,
            "accuracy_recovery_vs_sparsity.png"
        )
    )
)

print(
    os.path.abspath(
        os.path.join(
            PLOTS_DIR,
            "nonzero_parameters_vs_sparsity.png"
        )
    )
)