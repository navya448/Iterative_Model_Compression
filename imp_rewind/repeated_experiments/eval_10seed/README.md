# eval_10seed

Audit and evaluation of the 10-seed IMP rewind-point runs (ResNet-20, CIFAR-10), and of the formula for the rewind point t\*.

## Reports

1. [Part 1: 10-seed evaluation](reports/01_ten_seed_evaluation.md): audit of the runs, the formula fit, the held-out comparison against the baseline rules, and the model checks (κ vs ρ, capacity form).
2. [Part 2: redefined t\*](reports/02_redefined_tstar.md): common target, t\* from a curve through all trials, the comparison re-run, the 90%-safe formula and its calibration.

## Layout

| path | contents |
|---|---|
| `reports/` | the two reports, with figures embedded |
| `figures/` | fig1–fig10, PNG and PDF |
| `results/` | result tables (CSV); `report_tables.md` holds every number quoted in Part 1 |
| `audit/` | run checks: `AUDIT.md`, `checks.csv`, `rounds_status.csv`, `run_manifest.json` |
| `archive/first_formula_fit/` | the first formula fit on the two discovery runs, kept for reference |
| `01_*.py` … `08_*.py` | the pipeline, in order (below) |
| `common_eval.py`, `models.py` | shared loading, B(t), B⁻¹, predictors |

## Scripts

| file | step |
|---|---|
| `01_audit_runs.py` | checks every run directory; writes `audit/` |
| `02_original_method.py` | runs `imp_rewind/analyze.py` unchanged on the included runs |
| `03_formula_eval.py` | fits the formula, 5-train / 5-test seed splits, all baselines, paired tests |
| `04_model_checks.py` | κ(ρ), capacity forms, per-seed fits, safety/cost sweep |
| `05_figures.py` | fig1–fig7 |
| `06_report_tables.py` | `results/report_tables.md`, the source of every number in Part 1 |
| `07_redefine_tstar.py` | common target, t\* from a curve through all trials of a round, fig8 |
| `08_reevaluate.py` | comparison re-run against the redefined t\*, 90%-safe formula, calibration, fig9–fig10 |
| `run_all.py` | runs 01–08 |

## Reproduce

```
cd imp_rewind/repeated_experiments/eval_10seed
python run_all.py
```

Needs numpy, pandas, scipy and matplotlib. Takes a few minutes on CPU.

The per-split prediction files (`results/*split_predictions*.csv`, `results/split_scores_all.csv`, about 18 MB) are not committed. `run_all.py` regenerates them; `05_figures.py` needs `split_predictions_strict.csv`, so run `03_formula_eval.py` before it.

Inputs (read-only): `imp_rewind/runs/seed1`, `imp_rewind/runs/seed2` (discovery runs) and `imp_rewind/repeated_experiments/runs/seed1..seed10`. Repeated seeds 1 and 2 are excluded (see `audit/AUDIT.md`).
