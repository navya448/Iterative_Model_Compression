# formula_eval

Analysis of the data-order experiment (`../runs/20261102_165000`) and the revised t\* formula. The findings are in [report.md](report.md).

| file | contents |
|---|---|
| `formula.py` | the formula: `original()` (from `eval_10seed/models.py`, branch `eval-10seed`) and `revised()` |
| `evaluate.py` | the full analysis: noise from data order, the chain, the original formula out of sample, κ per round, the revised fit, the leave-one-run-out comparison, the probe check, figures |
| `results/` | CSV tables and `numbers.txt` (every number quoted in the report) |
| `figures/` | fig1–fig4 |

```
cd imp_rewind/data_order_experiments/formula_eval
python evaluate.py                       # about a minute on CPU
python evaluate.py --data_order_run <folder under ../runs>
```

Needs numpy, pandas, scipy and matplotlib. Inputs are read only: `imp_rewind/runs/seed1`, `seed2`, `imp_rewind/repeated_experiments/runs/seed3..seed10` and the data-order run folder.
