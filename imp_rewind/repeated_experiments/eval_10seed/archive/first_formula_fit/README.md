# First formula fit

The first version of the t\* formula, fitted on the two discovery runs (`imp_rewind/runs/seed1`, `seed2`) before the 10-seed experiments. It used to live at `imp_rewind/tstar_formula.py` and `imp_rewind/analysis_formula/`. The scripts in `eval_10seed/` supersede it; it is kept because the margins of 0 and 0.1 pp evaluated there were chosen from this fit.

- `tstar_formula.py`: fits a0, a1 and κ on all rewind trials and makes leave-one-run-out predictions
- `output/params.json`: the fitted parameters (a0 = 0.839, a1 = 0.676, κ = 0.114, RMSE 0.215 pp)
- `output/loo_predictions_margin0.00.csv`, `output/loo_predictions_margin0.10.csv`: leave-one-run-out predictions
- `output/tstar_formula.png`: the figure

To re-run it (from `imp_rewind/`):

```
python repeated_experiments/eval_10seed/archive/first_formula_fit/tstar_formula.py --runs runs/seed1 runs/seed2 --out <output dir>
```
