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
# quick check first (2 epochs, 1 round, separate folder)
python data_order_experiments\run_data_order.py --orders 1 2 --epochs 2 --rounds 1 --runs_dir data_order_experiments\runs_smoke

# full experiment
python data_order_experiments\run_data_order.py
```

Do not point the full run at the smoke folder. `run_imp.py` now refuses to
reuse a round or trial whose target / T / tol / seeds don't match, instead of
silently continuing from it.

## Output

- `runs\orderNN\` per-run dense + IMP results (same layout as before)
- `runs\experiment_manifest.json` parameters
- `analysis\t_star_by_round.csv` t* mean / std / min / max across the 10 orders per round
- `analysis\t_star_by_round_wide.csv` round x run table of t*
- `analysis\` the usual `analyze.py` outputs

Options: `--seed`, `--orders`, `--tol`, `--rounds`, `--epochs`,
`--continue_on_error`, `--skip_analysis`, `--dry_run`.

## Note on the old runs

In `repeated_experiments\runs`, seed1 and seed2 round 1 were left over from
the 2-epoch smoke test (target 63.36 / 54.66, t* = 2) and got reused by the
full run, so their later rounds continued from a 2-epoch model. Exclude
seed1 and seed2, or rerun them in a fresh folder.
