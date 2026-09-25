# IMP rewind-point experiments

Finds t*, the latest epoch you can rewind to after pruning and still recover accuracy, for every IMP round, and tests which candidate rule predicts it.

Setup (from the plan): ResNet-20 on CIFAR-10, 45k train / 5k val / 10k test, T = 80 epochs, SGD 0.1 (÷10 at epochs 40 and 60), global magnitude pruning of 20% of the remaining conv weights per round, 10 rounds, recovered = final val acc ≥ dense val acc − 0.5%.

## Files

| File | What it does |
|---|---|
| `common.py` | Data, ResNet-20, training, evaluation, pruning, BN recalibration |
| `rules.py` | The candidate rules (loss match, accuracy match, weight-movement match, specialization point, baselines) |
| `train_dense.py` | Experiment A: dense run, checkpoint + metrics every epoch |
| `run_imp.py` | Experiment B: IMP rounds with binary search for t* |
| `analyze.py` | Experiment C: tables and plots to find the formula |

## How to run

```bash
# CUDA build of PyTorch (plain `pip install torch` on Windows gives a CPU-only build).
# Pick the cuXXX that matches your driver: see https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pandas matplotlib
# check: should print True and your GPU name
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 0. Smoke test first (a few minutes): tiny T, 2 rounds
python train_dense.py --seed 99 --run_dir runs/smoke --epochs 4
python run_imp.py     --seed 99 --run_dir runs/smoke --rounds 2
python analyze.py     --runs runs/smoke --out analysis_smoke

# 1. Real runs, seeds 1 and 2 (to find the formula)
python train_dense.py --seed 1 --run_dir runs/seed1
python run_imp.py     --seed 1 --run_dir runs/seed1
python train_dense.py --seed 2 --run_dir runs/seed2
python run_imp.py     --seed 2 --run_dir runs/seed2
python analyze.py --runs runs/seed1 runs/seed2 --out analysis

# 2. Held-out seed 3 (only after you have settled on a rule)
python train_dense.py --seed 3 --run_dir runs/seed3
python run_imp.py     --seed 3 --run_dir runs/seed3
python analyze.py --runs runs/seed1 runs/seed2 --test_runs runs/seed3 --out analysis
```

Both training scripts **resume automatically** if interrupted: just rerun the same command. On Colab or Kaggle, put `runs/` on Google Drive or in the persistent output folder so it survives a disconnect.

Useful options for `run_imp.py`: `--tol 0.5` (recovery threshold), `--resolution 2` (binary-search precision in epochs), `--t_min 2` (earliest rewind tried), `--prune_frac 0.2`, `--rounds 10`.

## What gets recorded

`runs/seedN/dense/dense_log.csv`, one row per epoch 0..80:
`lr, run_train_loss, run_train_acc, val_loss, val_acc, train_loss, train_acc, test_loss, test_acc, grad_norm, weight_change, weight_norm`.
`train_*` is measured in eval mode on a fixed, un-augmented 10k training subset. `weight_change` = ‖W_e − W_{e−1}‖ over conv weights.

`runs/seedN/imp/rounds.csv`, one row per IMP round:
- `sparsity`, `before_*` / `after_*` metrics (after = pruned model with BN recalibrated, no retraining)
- damage: `D_loss_val`, `D_loss_train`, `D_acc_val`, `D_acc_train`, `D_loss_val_vs_dense`, `weight_perturbation` (‖removed‖ / ‖W‖), plus the same without BN recalibration for comparison
- `pred_*`: every rule's predicted t, **computed before the search** so it can't be tuned to the answer
- `t_star`, `retrain_epochs`, `t_star_frac` (t*/T), `lr_remaining_frac`, `n_trials`, `final_*` metrics

`runs/seedN/imp/rewind_trials.csv`: every rewind attempt with its final val/train/test loss and accuracy and whether it recovered. Per-epoch curves of each attempt are in `imp/round_XX/curve_tXXX.csv`.

## What `analyze.py` gives you

- `rule_scores.csv`: per rule, mean error vs t*, how often it was too late (would fail) and how many epochs it wasted when too early. The rule has to beat the baselines to be interesting.
- `feature_corr.csv`: how strongly t* follows each damage feature. The **within-round** column compares seeds at the same sparsity, so it separates "damage decides t*" from "both simply grow with sparsity". It needs at least 2 seeds.
- `figures/`: dense curves with LR drops and the specialization point, final accuracy vs rewind epoch for every round, t* vs each feature, predicted vs actual for every rule, and t* in epochs vs in share of the LR budget.

## Design choices to know about

- **LR schedule is rewound with the weights.** Retraining from t follows the original schedule from epoch t on. A fresh SGD optimizer is used at each rewind (momentum restarts), as in the standard lottery-ticket code.
- **BN is recalibrated before measuring damage.** Otherwise stale BN statistics inflate the damage.
- **The chain continues from the model retrained from t\*.**
- **Binary search assumes later rewinds recover less.** Check the `rewind_curves_*.png` plots: if a round's accuracy jumps up and down with t, that assumption is broken for that round.
- If the pruned model already meets the target with no retraining, t* = 80 and no trials are run. If even t = 2 fails, the chain stops there.
- t* depends on `--tol`. To compare thresholds, rerun `run_imp.py` into a different `--run_dir` that reuses the same dense run (copy the `dense/` folder).
