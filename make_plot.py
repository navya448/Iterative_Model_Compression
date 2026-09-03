import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. FILE LOCATIONS
# ============================================================

csv_file = r"C:\Users\nj255\major project\resnet18_cifar10_training_metrics.csv"

output_dir = r"C:\Users\nj255\major project\plots"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. LOAD CSV
# ============================================================

df = pd.read_csv(csv_file)

print("CSV loaded successfully.")
print(f"Number of epochs: {len(df)}")
print(df.head())


# ============================================================
# 3. LOSS PLOT
# ============================================================

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

loss_path = os.path.join(
    output_dir,
    "loss_vs_epoch.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. ACCURACY PLOT
# ============================================================

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

accuracy_path = os.path.join(
    output_dir,
    "accuracy_vs_epoch.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. GRADIENT AND WEIGHT NORM PLOT
# ============================================================

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

norm_path = os.path.join(
    output_dir,
    "gradient_weight_norm.png"
)

plt.savefig(
    norm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. WEIGHT CHANGE PLOT
# ============================================================

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

weight_change_path = os.path.join(
    output_dir,
    "weight_change_vs_epoch.png"
)

plt.savefig(
    weight_change_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. DONE
# ============================================================

print("\nPlots created successfully!")

print("\nSaved files:")
print(loss_path)
print(accuracy_path)
print(norm_path)
print(weight_change_path)