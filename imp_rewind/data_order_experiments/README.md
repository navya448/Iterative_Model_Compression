# Data-order IMP experiment

Goal: measure how much of the t* noise comes from data order alone.

## Changes from `repeated_experiments`

| | repeated_experiments | data_order_experiments |
|---|---|---|
| Runs | 10 seeds | 1 seed, 10 data orders |
| Model init | different per seed | same for all runs (`--seed`) |
| Augmentation (crop/flip) | different per seed | same for all runs: image i gets the same crop/flip in epoch e in every run |
| Data order | different per seed | different per run (`--order_seed`), in dense training, every retraining trial and BN recalibration |
| alpha (`--tol`) | 0.5 | 2.5 |
| Recovery target | dense val acc(T) - alpha | same, fixed for every round of a run |

Each run trains its own dense model, so each run has its own target, e.g.
dense val acc 91.0 means every round of that run needs 88.5.

## Run

From `imp_rewind\`:

```powershell
# quick check first (2 epochs, 1 round)
python data_order_experiments\run_data_order.py --orders 1 2 --epochs 2 --rounds 1

# full experiment
python data_order_experiments\run_data_order.py
```

Every launch starts from scratch: a new folder `runs\<date_time>\` is made,
each run trains its own dense model from epoch 0, then runs IMP. Nothing from
earlier runs (including `repeated_experiments`) is loaded; `train_dense.py`
and `run_imp.py` get `--fresh` and refuse to start if their folder already
has files.

If the experiment is interrupted, continue it in the same folder:

```powershell
python data_order_experiments\run_data_order.py --runs_dir data_order_experiments\runs\20261102_153000 --resume
```

## Sharing the GPU

`--gpu_fraction 0.5` (the default) keeps this experiment at about half the
GPU:

- compute: after every batch the process sleeps as long as the batch took, so
  the GPU is busy with it about 50% of the time and free for the other
  project the rest
- memory: PyTorch's allocator is capped at 50% of GPU memory

The cost is that the experiment takes about 2x as long as at full speed. Use
`--gpu_fraction 1` when the machine is free. The same limit works for the
other scripts through an environment variable:

```powershell
set IMP_GPU_FRACTION=0.5
python train_dense.py ...
```

## Output

All inside `runs\<date_time>\`:

- `orderNN\` per-run dense + IMP results (same layout as before)
- `orderNN.log` training log per run
- `experiment_manifest.json` parameters
- `analysis\t_star_by_round.csv` t* mean / std / min / max across the 10 orders per round
- `analysis\t_star_by_round_wide.csv` round x run table of t*
- `analysis\` the usual `analyze.py` outputs

Options: `--seed`, `--orders`, `--tol`, `--rounds`, `--epochs`,
`--gpu_fraction`, `--runs_dir`, `--resume`, `--continue_on_error`,
`--skip_analysis`, `--dry_run`.

## Note on the old runs

In `repeated_experiments\runs`, seed1 and seed2 round 1 were left over from
the 2-epoch smoke test (target 63.36 / 54.66, t* = 2) and got reused by the
full run, so their later rounds continued from a 2-epoch model. Exclude
seed1 and seed2, or rerun them in a fresh folder.
