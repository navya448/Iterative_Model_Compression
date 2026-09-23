# Iterative Model Compression

Experiments on CIFAR-10 that look at how a ResNet behaves during dense training and after magnitude pruning. The project covers:

1. **Dense baseline training.** ResNet-56 trained for 200 epochs, logging loss, accuracy, gradient norm, weight norm, and weight change every epoch.
2. **Learning-rate schedule ablation.** The same ResNet-56 run with a constant learning rate, compared against the step-decay baseline.
3. **Batchwise norm analysis.** Per-mini-batch gradient and weight norms (global and per layer) for a trained checkpoint.
4. **Global magnitude pruning with fine-tuning.** The dense checkpoint pruned to increasing sparsity levels, then fine-tuned to recover accuracy.

## Project Structure

```text
data-/
  cifar-10-batches-py/                          CIFAR-10 dataset (python version)

base_model_eval/
  base_model_eval.py                            ResNet-56 training with MultiStepLR decay
  base_model_eval_no_lr_decay.py                ResNet-56 training with a constant LR of 0.1
  batchwise_grad_weight_norm_analysis.py        Per-batch gradient/weight norm analysis (CLI)
  plot_no_lr_decay_metrics.py                   Plots for the constant-LR run
  plot_lr_comparison.py                         Overlay plots: LR decay vs. constant LR

  resnet56_cifar10_training_metrics.csv         Per-epoch metrics, LR decay run
  resnet56_cifar10_no_lr_decay_training_metrics.csv   Per-epoch metrics, constant-LR run
  resnet18_cifar10_training_metrics.csv         Per-epoch metrics from the earlier ResNet-18 run
  output_from_training.txt                      Console log, LR decay run
  output_no_lr_training.txt                     Console log, constant-LR run

  plots/                                        LR decay run plots
  plots_no_lr_decay/                            Constant-LR run plots
  plots_comparison/                             LR decay vs. constant-LR overlays

imp_finetuning/
  imp_finetuning.py                             Global magnitude pruning + fine-tuning
  magnitude_pruning_results.csv                 Summary by sparsity level
  magPruning.txt                                Detailed fine-tuning run log
```

Generated but not tracked by Git: `base_model_eval/checkpoints/`, `base_model_eval/checkpoints_no_lr_decay/`, `base_model_eval/analysis/`, `imp_finetuning/checkpoints/`, and `imp_finetuning/plots/`.

## Setup

Install the required Python packages:

```powershell
python -m pip install torch torchvision pandas matplotlib
```

The scripts expect the CIFAR-10 python batches in `data-/cifar-10-batches-py/` and do not download the dataset themselves. All paths are resolved relative to each script, so you can run the scripts from any working directory.

A CUDA GPU is used automatically when available; otherwise the scripts fall back to CPU.

## Experiments

### 1. Dense baseline (`base_model_eval.py`)

Trains a CIFAR-10 ResNet-56 (three stages of 9 basic blocks, 16/32/64 channels).

| Setting | Value |
| --- | --- |
| Epochs | 200 |
| Batch size | 128 |
| Optimizer | SGD, lr 0.1, momentum 0.9, weight decay 1e-4 |
| LR schedule | MultiStepLR, ×0.1 at epochs 150 and 175 |
| Augmentation | Random crop (padding 4), random horizontal flip |

Each epoch it records training and validation loss, training and validation accuracy, gradient norm, weight norm, and weight change from the previous epoch. It writes:

- `resnet56_cifar10_training_metrics.csv`
- `plots/`: `loss_vs_epoch.png`, `accuracy_vs_epoch.png`, `gradient_weight_norm.png`, `weight_change_vs_epoch.png`
- `checkpoints/epoch_050.pth`, `epoch_100.pth`, `epoch_150.pth`, `epoch_200.pth`, and `resnet56_cifar10_final.pth`

### 2. Constant-LR ablation (`base_model_eval_no_lr_decay.py`)

This run uses the same model, data, optimizer, and metrics as the baseline, but keeps the learning rate at 0.1 for all 200 epochs. Metrics go to `resnet56_cifar10_no_lr_decay_training_metrics.csv` and checkpoints go to `checkpoints_no_lr_decay/`.

After both runs have finished, two helper scripts build plots from the saved CSVs. Neither needs a GPU or a checkpoint.

- `plot_no_lr_decay_metrics.py` writes loss, accuracy, gradient norm, weight norm, and weight change plots for the constant-LR run to `plots_no_lr_decay/`.
- `plot_lr_comparison.py` overlays both runs for train/val loss, train/val accuracy, gradient norm, weight norm, and weight change, and saves the plots to `plots_comparison/`.

### 3. Batchwise norm analysis (`batchwise_grad_weight_norm_analysis.py`)

Loads a trained ResNet-56 checkpoint and makes one pass over the CIFAR-10 training set. Before each optimizer step (SGD, lr 0.001), it records the batch loss plus the global and per-layer gradient norms and weight norms.

```powershell
python base_model_eval/batchwise_grad_weight_norm_analysis.py [--checkpoint PATH] [--data-dir DIR] [--output-dir DIR] [--batch-size N] [--max-batches N]
```

| Option | Default |
| --- | --- |
| `--checkpoint` | `base_model_eval/checkpoints/resnet56_cifar10_final.pth`, or the newest `epoch_*.pth` in that folder if the final checkpoint is missing |
| `--data-dir` | `data-/` |
| `--output-dir` | `base_model_eval/analysis/` |
| `--batch-size` | 128 |
| `--max-batches` | no limit (full epoch) |

The script writes `batchwise_grad_weight_norm.csv` and `batchwise_grad_weight_norm.png` to the output directory.

### 4. Magnitude pruning with fine-tuning (`imp_finetuning.py`)

Loads `base_model_eval/checkpoints/resnet56_cifar10_final.pth` and applies global magnitude pruning to the dense model at 20%, 40%, 60%, 80%, and 90% sparsity. Each pruned model is fine-tuned for 50 epochs (lr 0.0005, batch size 128) with the pruning mask kept fixed.

The script records the parameter counts and the test accuracy before and after fine-tuning. It saves the results summary, plots, and per-sparsity checkpoints under `imp_finetuning/`.

## Running

Run the steps in this order. Pruning and the batchwise analysis both need the dense checkpoint from step 1.

```powershell
# 1. Dense baseline (creates base_model_eval/checkpoints/)
python base_model_eval/base_model_eval.py

# 2. Constant-LR ablation and its plots
python base_model_eval/base_model_eval_no_lr_decay.py
python base_model_eval/plot_no_lr_decay_metrics.py
python base_model_eval/plot_lr_comparison.py

# 3. Batchwise norm analysis
python base_model_eval/batchwise_grad_weight_norm_analysis.py

# 4. Pruning and fine-tuning
python imp_finetuning/imp_finetuning.py
```

> **Windows note:** if a run ends with `OMP: Error #15: Initializing libiomp5md.dll`, you have more than one OpenMP runtime loaded (typically from mixing conda and pip packages). This happened at the very end of the saved runs, after the metrics and checkpoints had already been written. Setting `$env:KMP_DUPLICATE_LIB_OK="TRUE"` is a common workaround, but the proper fix is to install PyTorch and NumPy from a single package source.

## Recorded Results

### ResNet-56: LR decay vs. constant LR (epoch 200)

| Run | Train acc. | Val acc. | Val loss | Weight norm |
| --- | --- | --- | --- | --- |
| MultiStepLR (0.1 → 0.01 → 0.001) | 99.92% | **93.79%** | 0.268 | 66.15 |
| Constant LR 0.1 | 95.17% | 88.69% | 0.413 | 73.78 |

With learning-rate decay, final validation accuracy is about 5.1 percentage points higher than with a constant learning rate. The constant-LR run never settles: at epoch 200 its epoch-to-epoch weight change is still around 13.9, compared with 0.06 for the decay run.

### Magnitude pruning with fine-tuning

> These numbers come from the **earlier ResNet-18 run** (11.16M parameters). The dense ResNet-18 reached 95.07% validation accuracy; see `resnet18_cifar10_training_metrics.csv`. The pruning script now targets ResNet-56, and its results have not been regenerated yet.

| Sparsity | Non-zero params | Acc. before FT | Acc. after FT | Recovered |
| --- | --- | --- | --- | --- |
| 0% (dense) | 11,164,352 | 95.18% | 95.18% | – |
| 20% | 8,931,482 | 95.21% | 95.38% | +0.17 |
| 40% | 6,698,612 | 95.21% | 95.36% | +0.15 |
| 60% | 4,465,741 | 94.97% | 95.32% | +0.35 |
| 80% | 2,232,871 | 89.66% | 95.06% | +5.40 |
| 90% | 1,116,436 | 42.35% | 94.83% | +52.48 |

Up to 60% sparsity, the pruned models lose almost no accuracy even before fine-tuning. At 90% sparsity, accuracy drops to 42.35% right after pruning, and 50 fine-tuning epochs bring it back to 94.83%, only 0.35 points below the dense model.
