# Repeated IMP experiments

This folder runs the existing dense-training and IMP-rewind experiment
independently for multiple random seeds. It does not duplicate or modify the
experiment implementation in the parent folder.

## Full run

From `imp_rewind\`:

```powershell
python repeated_experiments\run_repeated.py
```

The default is seeds `1` through `10`. Each seed is isolated under
`repeated_experiments\runs\seedN`, and the existing scripts resume if a run is
interrupted. The combined output is written to
`repeated_experiments\analysis`.

Each seed also has a log file and the run folder contains
`experiment_manifest.json`, which records the parameters and completion
status.

## Small validation run

Before starting the full 80-epoch, 10-round experiment, verify the wrapper
with a short run:

```powershell
python repeated_experiments\run_repeated.py --seeds 1 2 --epochs 2 --rounds 1
```

Use `--dry_run` to inspect commands without training:

```powershell
python repeated_experiments\run_repeated.py --dry_run
```

Useful options include `--seeds 11 12 13`, `--continue_on_error`,
`--skip_analysis`, `--tol`, `--resolution`, and `--t_min`.
