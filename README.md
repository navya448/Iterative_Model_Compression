# Iterative Model Compression

Experiments analyzing ResNet-18 behavior on CIFAR-10 during normal training and after iterative magnitude pruning with fine-tuning.

## Project Structure

```text
data-/
  cifar-10-batches-py/              CIFAR-10 dataset

base_model_eval/
  base_model_eval.py                Dense ResNet-18 training
  resnet18_cifar10_training_metrics.csv
  plots/                            Training behavior plots

imp_finetuning/
  imp_finetuning.py                 Magnitude pruning and fine-tuning
  magnitude_pruning_results.csv     Summary by sparsity level
  magPruning.txt                    Detailed fine-tuning run log
  plots/                            Pruning result plots
```

## Experiments

### Base model evaluation

The base model script trains a CIFAR-10 adapted ResNet-18 for 200 epochs and records:

- Training and validation loss
- Training and validation accuracy
- Gradient norm
- Weight norm
- Weight change between epochs

It saves the metrics CSV, plots, and checkpoints under `base_model_eval/`.

### IMP with fine-tuning

The IMP experiment loads the 200-epoch dense checkpoint, applies global magnitude pruning at 20%, 40%, 60%, 80%, and 90% sparsity, and fine-tunes each pruned model for 50 epochs.

It records accuracy before and after fine-tuning, parameter counts, accuracy recovery, plots, and the detailed run log.

## Running

Install the required Python packages:

```powershell
python -m pip install torch torchvision pandas matplotlib
```

Run the base experiment first so that the dense checkpoint is available:

```powershell
python base_model_eval/base_model_eval.py
```

Then run pruning and fine-tuning:

```powershell
python imp_finetuning/imp_finetuning.py
```

The CIFAR-10 files are expected in `data-/cifar-10-batches-py/`. Model checkpoint files are ignored by Git and are generated locally under `base_model_eval/checkpoints/` and `imp_finetuning/checkpoints/`.

## Recorded Result

In the saved run, 90% pruning reduced immediate test accuracy to 42.35%. After 50 fine-tuning epochs, accuracy recovered to 94.83%, a recovery of 52.48 percentage points.
