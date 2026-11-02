# Experiment 01: Iterative Model Compression with Dynamic Learning-Rate Retraining

## Purpose

This experiment investigates whether an iterative magnitude-pruning pipeline can
recover the accuracy lost after pruning **without rewinding the model to an
earlier training checkpoint**.

The proposed method starts with a normally trained dense ResNet-20 and
repeatedly removes a fraction of the smallest surviving convolution weights.
After each pruning step, the model continues training from its current
parameters. A dynamic learning-rate controller monitors validation progress and
reduces the learning rate when the current training stage reaches a plateau.

The intention is to compare this approach with the standard
**Iterative Magnitude Pruning (IMP) with rewinding** approach implemented in
`imp_rewind/`. The experiment is designed to provide evidence for a go/no-go
decision on using adaptive learning-rate retraining as an alternative to
rewinding.

## Research question

> After each pruning step, can a model recover its validation accuracy by
> continuing from its pruned weights and adapting the learning rate according
> to observed training progress, rather than rewinding to an earlier dense
> checkpoint?

## Hypothesis

Dynamic learning-rate retraining should recover most or all of the dense
model's validation accuracy at moderate sparsity levels. It may also reduce
unnecessary retraining because:

- the learning rate is lowered only after loss and accuracy improvements stall;
- retraining stops as soon as the recovery target is reached;
- the best validation-accuracy checkpoint is retained if the target is not
  reached.

The experiment may show that this method is insufficient at high sparsity. That
result is still useful: it identifies the sparsity range where rewinding or a
different recovery strategy is required.

## Experimental setup

| Component | Configuration |
| --- | --- |
| Dataset | CIFAR-10 |
| Data split | 45,000 training / 5,000 validation / 10,000 test images |
| Model | ResNet-20 for CIFAR-10 |
| Dense training | 80 epochs |
| Optimizer | SGD, initial learning rate 0.1, momentum 0.9, weight decay 1e-4 |
| Dense LR schedule | Learning rate divided by 10 at epochs 40 and 60 |
| Pruning method | Global magnitude pruning |
| Pruning amount | 20% of currently surviving convolution weights per round |
| Prunable parameters | All convolution weights |
| Kept dense | Final fully connected layer, batch-normalization parameters, and biases |
| Number of pruning rounds | 10 by default |
| Recovery target | Dense validation accuracy minus 0.5 percentage points |
| Retraining | Starts from the pruned model; no weight rewinding |

Because 20% of the surviving weights are removed in each round, the expected
convolution-weight sparsity after round \(r\) is:

```text
1 - 0.8^r
```

For example, the expected sparsity is approximately 20%, 36%, 49%, and 59%
after rounds 1 through 4, respectively.

## Method

### 1. Train the dense baseline

The script trains a dense ResNet-20 on CIFAR-10 and records the validation
accuracy. This accuracy defines the recovery target for every pruning round:

```text
recovery target = dense validation accuracy - tolerance
```

The default tolerance is 0.5 percentage points.

### 2. Prune globally by weight magnitude

At the start of each IMP round, the script removes the smallest-magnitude
weights among the currently surviving convolution weights. A binary mask is
maintained so that removed weights remain zero during retraining.

### 3. Recalibrate batch normalization and measure pruning damage

Batch-normalization statistics are recalibrated after pruning before measuring
the immediate effect of pruning. The script records:

- validation-loss increase;
- validation-accuracy decrease;
- accuracy gap relative to the dense model;
- relative norm of the removed weights.

### 4. Retrain with the dynamic learning-rate controller

If the pruned model is already at the recovery target, the round requires no
retraining. Otherwise, retraining uses:

1. A fresh SGD optimizer for the round.
2. Linear warmup to the initial learning rate.
3. Exponential moving averages of validation loss and validation accuracy.
4. Improvement rates measured over a configurable sliding window.
5. A plateau detector based on both loss and accuracy improvement.
6. Learning-rate reduction by `lr_factor` when the plateau persists.
7. Early stopping after the target is reached for `confirm` consecutive epochs.

The round ends when one of the following occurs:

- **Recovered:** the validation target is reached;
- **Plateau at minimum LR:** another learning-rate reduction would go below
  `lr_min`;
- **Maximum epochs:** the retraining budget is exhausted.

If recovery does not occur, the model state with the best validation accuracy
seen during that round is retained.

### 5. Continue the pruning chain

The retrained model becomes the starting point for the next pruning round.
This makes the experiment an actual iterative compression chain rather than a
set of independent pruning tests.

## Tunable parameters and current values

The following values are currently defined in `exp_01.py`. They can be changed
from the command line without modifying the source code.

### Experiment and pruning parameters

| Argument | Current value | Meaning |
| --- | ---: | --- |
| `--seed` | `1` | Random seed |
| `--run_dir` | `dynamic lr/runs/seed<seed>` | Output directory when not supplied |
| `--data_dir` | `../imp_rewind/data` | CIFAR-10 data directory |
| `--dense_epochs` | `80` | Dense-model training epochs |
| `--rounds` | `10` | Number of iterative pruning rounds |
| `--prune_frac` | `0.2` | Fraction of surviving convolution weights removed per round |
| `--tol` | `0.5` | Allowed validation-accuracy drop from the dense model, in percentage points |
| `--bn_batches` | `100` | Batches used for batch-normalization recalibration |
| `--stop_when_unrecovered` | Off | Stop the pruning chain after the first unsuccessful recovery |

### Dynamic learning-rate controller parameters

| Argument | Current value | Meaning |
| --- | ---: | --- |
| `--lr0` | `0.1` | Initial retraining learning rate |
| `--lr_factor` | `0.1` | Multiplier applied after a plateau |
| `--lr_min` | `0.001` | Minimum learning-rate stage |
| `--warmup` | `2` | Linear warmup epochs |
| `--ema` | `0.5` | Weight of the newest validation observation |
| `--window` | `3` | Window used to estimate improvement rates |
| `--loss_eps` | `0.005` | Minimum validation-loss improvement per epoch |
| `--acc_eps` | `0.1` | Minimum validation-accuracy improvement per epoch |
| `--patience` | `2` | Consecutive stalled epochs before a learning-rate drop |
| `--max_stage_epochs` | `30` | Maximum epochs at one learning-rate stage |
| `--confirm` | `2` | Consecutive target-reaching epochs required for recovery |
| `--max_epochs` | `80` | Maximum retraining epochs per pruning round |

The shared training configuration currently used by the experiment is batch
size `128`, SGD momentum `0.9`, weight decay `1e-4`, and dense-training
learning-rate milestones at epochs `40` and `60` with a `0.1` multiplier.

## Requirements

The experiment requires Python and:

```powershell
python -m pip install torch torchvision numpy
```

The script imports shared utilities from
[`imp_rewind/common.py`](../imp_rewind/common.py). Run the commands from this
directory or provide paths explicitly. CUDA is used automatically when
available; otherwise PyTorch falls back to CPU, although a CUDA-capable GPU is
strongly recommended for the full experiment.

The CIFAR-10 dataset is downloaded by `torchvision` into the configured data
directory if it is not already present.

## How to run

From the `dynamic lr` directory:

```powershell
cd "D:\computer science\ML Major Proj\Iterative_Model_Compression\dynamic lr"
```

### Smoke test

Use a small run to verify the environment and output structure:

```powershell
python exp_01.py --seed 99 --dense_epochs 2 --rounds 2 --max_epochs 4 --run_dir runs\smoke
```

### Full experiment

The default run trains for 80 dense epochs and performs 10 pruning rounds:

```powershell
python exp_01.py --seed 1
```

For a second independent seed:

```powershell
python exp_01.py --seed 2
```

To compare controller settings, use a separate output directory:

```powershell
python exp_01.py --seed 1 --lr_factor 0.2 --patience 3 --run_dir runs\ablation_lr_factor
```

The script resumes completed dense training and completed IMP rounds
automatically. Re-run the same command after an interruption.

## Outputs

For a run with `--run_dir runs\seed1`, the main outputs are:

```text
runs/seed1/
  config.json
  dense/
    dense_log.csv
    final_model.pt
    final_metrics.json
  imp/
    rounds.csv
    masks/
      round_01.pt
      ...
    round_01/
      curve.csv
      final_model.pt
      summary.json
    ...
```

### Important result files

- `dense/final_metrics.json`: dense baseline loss and accuracy.
- `imp/rounds.csv`: one summary row per pruning round.
- `imp/round_XX/curve.csv`: epoch-level retraining behavior, including the
  learning rate, EMA values, improvement rates, and controller events.
- `imp/round_XX/summary.json`: pruning damage, retraining schedule, stopping
  reason, and final metrics for one round.
- `imp/masks/round_XX.pt`: binary pruning mask for the round.

## Evidence needed for approval

The experiment should be considered successful enough to proceed if the
following evidence is available:

1. The dense baseline trains successfully and produces reproducible metrics.
2. The dynamic controller lowers the learning rate only after measurable
   plateaus, rather than at arbitrary fixed epochs.
3. Several pruning rounds recover to within the chosen tolerance of the dense
   validation accuracy.
4. The retained model remains masked, so pruned weights do not return during
   retraining.
5. Results are consistent across at least two random seeds.
6. `rounds.csv` shows the trade-off between sparsity, retraining epochs,
   learning-rate drops, validation accuracy, and test accuracy.

The most important comparison is not only final accuracy. It is also the
retraining cost required to recover accuracy at each sparsity level.

## Suggested comparison with IMP rewinding

Run the corresponding rewinding experiment in `imp_rewind/` using comparable
seeds, pruning fractions, and recovery tolerance. Compare:

- final validation and test accuracy at each sparsity level;
- number of retraining epochs;
- number and timing of learning-rate changes;
- number of rounds that fail to recover;
- variation across random seeds.

This comparison will show whether dynamic continuation is a practical
alternative to rewinding, and where each method is preferable.

## Limitations and discussion points

- A validation target based on the dense model does not prove that the pruned
  model is optimal; it measures recovery relative to the baseline.
- The plateau detector depends on thresholds such as `loss_eps`, `acc_eps`,
  and `patience`.
- The experiment uses one architecture and one dataset, so conclusions should
  not be generalized without additional experiments.
- The current method restarts SGD momentum at every pruning round. This is
  intentional, but it should be kept the same in comparisons.
- The default controller is designed for investigation, not claimed as an
  optimized production schedule.

## Expected decision after the experiment

At the end of the run, the results should support one of these decisions:

1. **Proceed:** dynamic learning-rate retraining recovers accuracy reliably and
   uses acceptable retraining cost.
2. **Proceed with limits:** the method works up to a specific sparsity level,
   after which rewinding or another recovery method is required.
3. **Do not proceed:** recovery is unreliable or substantially more expensive
   than the rewinding baseline.
