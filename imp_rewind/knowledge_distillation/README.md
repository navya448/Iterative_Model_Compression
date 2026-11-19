# Knowledge distillation and the rewind formula

These experiments pair IMP with knowledge distillation (KD). The dense model is the **teacher**, and each pruned model is retrained as the **student**. The aim is to estimate the rewind formula's parameters for a new model and dataset from their structure, without running the full t\* search.

The work follows [Part 3](../data_order_experiments/formula_eval/report.md), which gave the revised formula:

```
A(t) − A_dense = a0 − a1·ln(1/d) + a2·(before_val_acc − A_dense)
                 − (k0 + k1·ln(1 + D_acc)) · ln(B(t_min)/B(t))
```

Everything in this folder is self-contained. It does not import or change the older experiments.

## Run everything

On the lab GPU, from this folder:

```bash
# 1. Smoke test of every model: tiny runs, ~10-20 min on the GPU.
python run_all.py --smoke

# 2. The real experiments. One command runs every step in order.
nohup python run_all.py > run_all.out 2>&1 &      # Linux: keeps running after you log out
# or inside tmux / screen:  python run_all.py
```

On Windows, open a terminal and run `python run_all.py`. Keep that window open.

- **Rerunning picks up where it stopped.** After a crash, a reboot or Ctrl-C, run the same command again. Finished epochs, trials, rounds and runs are kept; at most the last unfinished epoch is lost.
- **Failures are contained.** A failing job is retried twice, one minute apart. If it still fails, only the jobs that depend on it are skipped, and the rest of the plan continues.
- **Results update as runs finish.** The analysis is refreshed after every finished IMP chain. Open [`analysis/summary.md`](analysis/summary.md) at any time to see the results so far.

**To check progress:**

| File | What it shows |
|---|---|
| `runs/status.json` | state of every job (done, running, failed, skipped), attempts and minutes |
| `runs/logs/run_all.log` | the timeline |
| `runs/logs/<job>.log` | the full output of one job |

**Useful options:**

| Option | Effect |
|---|---|
| `--dry_run` | lists the jobs and which ones are already done |
| `--only resnet56` | runs only jobs whose name contains the text, plus the jobs they need |
| `--gpu 1` | uses GPU 1 |
| `--gpu_fraction 0.5` | uses about half of a shared GPU |
| `--quiet` | doesn't echo job output to the console; the logs still get it |

**Requirements:** `torch`, `numpy`, `pandas` and `matplotlib`. `torchvision` is not needed. Everything uses the CIFAR-10 copy already in `../data/cifar-10-batches-py/`, with the same 45k / 5k / 10k split as the old experiments. Nothing is downloaded.

## What runs, in order ([`plan.json`](plan.json))

All runs use CIFAR-10. What changes between settings is the **model**, so the transfer test asks whether the formula's parameters can be predicted from the architecture.

| Stage | Models | Seeds | Methods | Answers |
|---|---|---|---|---|
| `step1_kd_vs_ce_resnet20` | ResNet-20 | 1, 2, 3 | `kd`, `ce` | What does KD change in the formula? Does a2 drop to 0? |
| `step3_models_seed1` | ResNet-32, ResNet-44, VGG-11, ResNet-56 | 1 | `kd` | Settings for the transfer test |
| `step3_models_seed2` | same | 2 | `kd` | Second seed, for CIs and leave-one-run-out |

| Model | Prunable conv weights | Conv layers | z_overparam = ln(weights / 45k) |
|---|---|---|---|
| ResNet-20 | 0.27 M | 21 | 1.79 |
| ResNet-32 | 0.46 M | 33 | 2.33 |
| ResNet-44 | 0.66 M | 45 | 2.68 |
| ResNet-56 | 0.85 M | 57 | 2.94 |
| VGG-11 | 9.2 M | 8 | 5.32 |

The four ResNets differ only in depth. VGG-11 is a different family: much wider, shallower and without skip connections.

Steps 2 and 4 need no extra training. They are the refit in normalized units and the leave-one-setting-out test, and both run in `analyze_kd.py`. The models run from cheapest to most expensive, so a first version of the transfer test is available early.

### Retraining methods

| Method | Student loss during the retrain from t to T |
|---|---|
| `ce` | Cross-entropy only. This is the control, the same as the old experiments. |
| `kd` | `0.1·CE + 0.9·T²·KL(teacher_T ‖ student_T)` with T = 4. The teacher is the **dense model at epoch T**, fixed for the whole chain. This is the professor's setup. |
| `kd_parent` | Same loss, but the teacher is the chain model before this round's pruning. Not in the default plan. |

The teacher sees the same augmented batch as the student. CE and KD trials use the same data order, so their round 1 results are an exact pair.

### Changes from the old protocol, and why

The old protocol is otherwise unchanged: 45k / 5k / 10k split, T = 80, SGD 0.1 with ÷10 at epochs 40 and 60, 20% global magnitude pruning per round, 10 rounds, t_min = 2 and resolution 2.

- **Relative recovery target.** A round recovers when `val_err ≤ dense_val_err · (1 + 0.055)`. For ResNet-20 that is about 0.5 pp, the old α. A stronger model with lower error gets a proportionally tighter target, so "recovered" means the same thing for every model.
- **A probe rewind each round.** After the t = t_min trial, the search runs one short late rewind at t = 66 (82% of T), then bisects whichever bracket is left. The probe is part of the search, so it costs about one extra trial per round. In return every round gets a κ measurement, which the Part 3 report could only check on some rounds.
- **New measurements before retraining.** Each round also records:
  - KL(dense ‖ pruned) on val, before and after pruning
  - three structure-aware densities (below)
  - per-layer densities, in `round_XX/layer_density.csv`
- **VGG-11 uses peak LR 0.05**, because 0.1 is unstable for it. The rewind clock is a ratio in units of the peak LR, so this does not change its scale.
- **Trial weights are deleted once a round is done.** Each trial's JSON, curve and metrics are kept. Pass `--keep_trial_models` to `kd_job.py` if you need the weights.

## The formula variants ([`kd_formula.py`](kd_formula.py))

Every variant has the same shape: a "badness" that grows with the rewind clock.

```
bad(t) = c0 + c_density·ln(1/x) + c_parent·parent + (k0 + k_damage·damage) · ln(B(t_min)/B(t))
```

The variants differ in three choices: the scale, the density measure x, and the damage measure. All of them are computed before any retraining.

**Scale**

| Scale | Badness | Parent term |
|---|---|---|
| `pp` | `A_dense − A(t)`, in pp. This is exactly the revised formula, with a0 = −c0, a1 = c_density, a2 = c_parent. | `A_dense − before_val_acc` |
| `ln` | `ln(err(t) / err_dense)`, in e-folds of relative error. Error becomes a power law in density and in budget, which is the form expected to carry across datasets. | `ln(err_before / err_dense)` |

**Density x**

| Measure | Meaning |
|---|---|
| `d_global` | kept fraction of conv weights (the old d) |
| `d_flops` | kept fraction of conv multiply-accumulates |
| `d_min` | density of the most-pruned layer |
| `d_lgeo` | geometric mean of per-layer densities |

**Damage**

| Measure | Formula |
|---|---|
| `acc` | `ln(1 + D_acc)` (the old one) |
| `kl` | `ln(1 + D_KL / 0.01)` |
| `relerr` | `ln(err_after / err_before)` |

D_KL is how much the KL to the dense teacher rose when this round pruned.

The analysis compares the old formula (`pp/d_global/acc`) with all 12 `ln` variants. `ln/d_flops/kl` was **chosen in advance** as the main variant, so that the transfer result is not tuned to the data.

**Meta-model.** Each parameter is predicted from at most one structure descriptor. The descriptors were fixed in advance; see `META_DESCRIPTORS` in [`kd_formula.py`](kd_formula.py).

| Parameter | Descriptor | Meaning |
|---|---|---|
| c_density | `ln(n_prunable_weights / n_train)` | over-parameterization |
| c0, c_parent, k0, k_damage | none | the same for every setting (the mean over training settings) |

k0 should depend on how hard the task is. With CIFAR-10 only, that hardly changes between models, so k0 is shared. If you add other datasets later, set `"k0": "z_difficulty"` in `META_DESCRIPTORS`.

Every dense run saves these in `dense/descriptors.json`, along with the parameter count, depth, conv MACs, number of classes, and the teacher's entropy and NLL.

## What the analysis reports ([`analyze_kd.py`](analyze_kd.py) → `analysis/`)

| Section | Files | Question |
|---|---|---|
| Overview | `runs_overview.csv`, `figures/tstar_by_round.png` | What has finished? Does recovery decrease monotonically in t, which the binary search assumes? |
| 1. CE vs KD | `ce_vs_kd_rounds.csv`, `ce_vs_kd_params.csv`, `figures/ce_vs_kd_params_*.png` | Paired Δt\* and Δ capacity. Parameters with 95% run-bootstrap CIs. Is c_parent (a2) about 0 under KD? |
| 2. Parameters | `params_by_group.csv` | Fitted parameters per setting and method |
| 3. Variants | `variant_scores.csv` | Which scale, density and damage fit best inside a setting (t\* MAE, leave one run out; accuracy RMSE in pp) |
| 4. Transfer | `transfer_mae_kd.csv`, `transfer_predictions_kd.csv`, `figures/transfer_kd.png` | Leave one setting out. (a) global fit, (b) parameters from structure, (c) structure plus the probe's κ, (d) the setting's own fit (upper bound). `gap_closed_by_b` = how much of the (a)-to-(d) gap structure alone closes |
| 5. Structure | `params_vs_descriptors.csv`, `kappa_per_round.csv`, `figures/params_vs_descriptors.png`, `figures/kappa_per_round.png` | Do the parameters line up with the descriptors? How does κ change across rounds? |

`summary.md` collects the tables from every section. `all_rounds.csv` and `all_trials.csv` hold every number, with all features attached, for your own analysis.

You can run the analysis on its own at any time, including while experiments are still running:

```bash
python analyze_kd.py --runs_dir runs --out analysis
```

## Time and disk (estimates; check them against your first logs)

Your earlier ResNet-20 chains used about 1,300–1,800 retraining epochs each. On the lab's RTX A4000, rough guesses are:

| Job | Approx. time |
|---|---|
| Dense ResNet-20 (80 epochs) | 10–15 min |
| IMP chain, ResNet-20 / CIFAR, CE | about 3–4 h |
| IMP chain, ResNet-20, KD | about 4–5 h (the teacher's forward pass adds about 30%) |
| ResNet-32 / 44 / 56 | about 1.6× / 2.2× / 2.8× ResNet-20 |
| VGG-11 | about 2× ResNet-20 |

| Stage | Approx. total |
|---|---|
| Step 1 (3 seeds × KD and CE) | about 1 day |
| Each model seed (4 models, KD) | about 1.5–2 days |
| **Whole plan** | **about 4–5 days** |

To get usable results sooner:
- **Trim the plan.** Delete models or seeds in `plan.json`; nothing else needs changing. Dropping ResNet-44 saves the most time while keeping the spread from 0.27 M to 9.2 M weights.
- **Read the per-epoch time** from the first dense log and scale the table above by it.

**Disk:** every dense run keeps all 81 epoch checkpoints, because rewinds need them.

| Model | Per seed |
|---|---|
| ResNet-20 / 32 / 44 / 56 | about 25 / 40 / 55 / 70 MB |
| VGG-11 | about 3 GB |

Plan on about 7 GB in total.

## Files

| File | What it does |
|---|---|
| [`run_all.py`](run_all.py) | Runs the whole plan, unattended and resumable, and refreshes the analysis |
| [`plan.json`](plan.json) | Which settings, seeds and methods to run, plus shared hyperparameters |
| [`kd_job.py`](kd_job.py) | One job: `prepare` (download a dataset), `dense` (train the teacher), `imp` (IMP chain with t\* search, CE or KD) |
| [`kd_common.py`](kd_common.py) | CIFAR-10 data, models (ResNet-20/32/44/56, VGG-11), KD loss, training, pruning, density and KL measures. Loaders for CIFAR-100 and SVHN are included but not used by the plan. |
| [`kd_formula.py`](kd_formula.py) | Formula variants, fitting and t\* prediction |
| [`analyze_kd.py`](analyze_kd.py) | All analysis, tables, figures and `summary.md` |

### Output layout

```
runs/cifar10_<model>/seed<k>/
  dense/  checkpoints/epoch_000..080.pt, dense_log.csv, final_metrics.json, descriptors.json
  kd/     config.json, rounds.csv, rewind_trials.csv, masks/, DONE.json,
          round_XX/ summary.json, final_model.pt, layer_density.csv,
                    trial_tXXX.json, curve_tXXX.csv (per-epoch val/train acc, KL to teacher)
  ce/     same as kd/
runs/status.json, runs/logs/
```

### Running a single job by hand

```bash
python kd_job.py dense --dataset cifar10 --model resnet32 --seed 1
python kd_job.py imp   --dataset cifar10 --model resnet32 --seed 1 --method kd
```

## Caveats to keep in mind

- **This tests transfer across models, not datasets.** All five settings are CIFAR-10. Whether the parameters also carry over to another dataset is a separate question that this plan cannot answer.
- **With five settings, the meta-model is only a first check.** It has one slope (c_density against over-parameterization), fitted from four training settings. VGG-11 sits far from the ResNets, so a lot rests on it. Read `gap_closed_by_b` together with the per-setting parameters, not on its own.
- **The step-3 models have only 2 seeds each.** CIs per model are wide. The leave-one-run-out "own fit" uses the other seed.
- **CE and KD runs differ only in the retraining loss.** After round 1 the two chains continue from different parents, which is how IMP works. Only round 1 is an exact pair.
- **The comparison depends on the KD hyperparameters.** T = 4 and λ = 0.9 are fixed for every setting, so they do not turn into extra parameters. Changing them needs a fresh `runs_dir`, and the jobs refuse to mix configs.
