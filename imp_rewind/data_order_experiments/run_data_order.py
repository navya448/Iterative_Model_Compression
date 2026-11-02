"""Data-order experiment: one seed, 10 runs that differ only in data order.

What is fixed across runs (all from --seed):
  - model initialization
  - per-image augmentation (crop offsets + flips) for every epoch
  - train/val split (split_seed in common.py, same as before)

What changes per run (--orders):
  - shuffle order of the training data, in dense training, in every IMP
    retraining trial, and in the BN recalibration batches

Recovery condition: each run trains its own dense model, and every IMP round
of that run must reach  dense_val_acc(T) - tol . With tol = 2.5 and a dense
val acc of 91.0, the target is 88.5 for all rounds of that run.

Run from imp_rewind/:
    python data_order_experiments/run_data_order.py

Small validation run first:
    python data_order_experiments/run_data_order.py --orders 1 2 --epochs 2 --rounds 1 --runs_dir data_order_experiments/runs_smoke
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
DEFAULT_RUNS_DIR = os.path.join(HERE, "runs")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1,
                        help="fixes model init and augmentation for every run (default: 1)")
    parser.add_argument("--orders", nargs="+", type=int, default=list(range(1, 11)),
                        help="data-order seeds, one run each (default: 1 2 ... 10)")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--runs_dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--prune_frac", type=float, default=0.2)
    parser.add_argument("--tol", type=float, default=2.5,
                        help="alpha: recovered if val acc >= dense val acc - tol (default: 2.5)")
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--t_min", type=int, default=2)
    parser.add_argument("--bn_batches", type=int, default=100)
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def run_command(command, log_path, dry_run):
    print("  " + subprocess.list2cmdline(command))
    if dry_run:
        return
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=PROJECT_DIR, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed with exit code {result.returncode}; see {log_path}")


def summarize_t_star(run_dirs, out_dir):
    """t* per round across runs: the spread here is the data-order noise."""
    frames = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "imp", "rounds.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["run"] = os.path.basename(os.path.normpath(run_dir))
        frames.append(df)
    if not frames:
        return
    rounds = pd.concat(frames, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)

    wide = rounds.pivot(index="round", columns="run", values="t_star")
    wide.to_csv(os.path.join(out_dir, "t_star_by_round_wide.csv"))

    rows = []
    for rnd, g in rounds.groupby("round"):
        t = g["t_star"].dropna()
        rows.append(dict(
            round=rnd,
            sparsity=g["sparsity"].mean(),
            n_runs=len(g),
            n_recovered=len(t),
            t_star_mean=t.mean() if len(t) else None,
            t_star_std=t.std(ddof=1) if len(t) > 1 else None,
            t_star_min=t.min() if len(t) else None,
            t_star_median=t.median() if len(t) else None,
            t_star_max=t.max() if len(t) else None,
            target_min=g["target_val_acc"].min(),
            target_max=g["target_val_acc"].max(),
        ))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "t_star_by_round.csv"), index=False)
    print("\n=== t* across data orders ===")
    print(summary[["round", "n_recovered", "t_star_mean", "t_star_std",
                   "t_star_min", "t_star_max"]].to_string(index=False))


def main():
    args = parse_args()
    orders = sorted(set(args.orders))
    if not orders or any(o < 0 for o in orders):
        raise SystemExit("--orders must contain at least one non-negative integer")

    os.makedirs(args.runs_dir, exist_ok=True)
    manifest_path = os.path.join(args.runs_dir, "experiment_manifest.json")
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        design="fixed init + fixed augmentation (seed), data order varies per run",
        seed=args.seed,
        aug_seed=args.seed,
        orders=orders,
        epochs=args.epochs,
        rounds=args.rounds,
        prune_frac=args.prune_frac,
        tol=args.tol,
        recovery_rule="final val acc >= this run's dense val acc at T - tol, same for every round",
        resolution=args.resolution,
        t_min=args.t_min,
        bn_batches=args.bn_batches,
    )
    if not args.dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    seed_args = ["--seed", str(args.seed), "--aug_seed", str(args.seed)]
    completed, failed = [], []
    for i, order in enumerate(orders, 1):
        name = f"order{order:02d}"
        run_dir = os.path.join(args.runs_dir, name)
        log_path = os.path.join(args.runs_dir, f"{name}.log")
        common = seed_args + ["--order_seed", str(order), "--run_dir", run_dir,
                              "--data_dir", args.data_dir]

        dense_cmd = [sys.executable, os.path.join(PROJECT_DIR, "train_dense.py"), *common]
        if args.epochs is not None:
            dense_cmd += ["--epochs", str(args.epochs)]
        imp_cmd = [sys.executable, os.path.join(PROJECT_DIR, "run_imp.py"), *common,
                   "--rounds", str(args.rounds),
                   "--prune_frac", str(args.prune_frac),
                   "--tol", str(args.tol),
                   "--resolution", str(args.resolution),
                   "--t_min", str(args.t_min),
                   "--bn_batches", str(args.bn_batches)]

        print(f"\nRun {name} ({i}/{len(orders)})")
        try:
            run_command(dense_cmd, log_path, args.dry_run)
            run_command(imp_cmd, log_path, args.dry_run)
            completed.append(run_dir)
        except RuntimeError as error:
            failed.append(dict(order=order, error=str(error)))
            print(f"  FAILED: {error}")
            if not args.continue_on_error:
                raise

    analysis_dir = os.path.join(HERE, "analysis") if args.runs_dir == DEFAULT_RUNS_DIR \
        else os.path.join(args.runs_dir, "analysis")
    if not args.dry_run and completed:
        summarize_t_star(completed, analysis_dir)
        if not args.skip_analysis:
            run_command([sys.executable, os.path.join(PROJECT_DIR, "analyze.py"),
                         "--runs", *completed, "--out", analysis_dir],
                        os.path.join(args.runs_dir, "analysis.log"), args.dry_run)

    if not args.dry_run:
        manifest["completed_orders"] = [int(os.path.basename(p)[5:]) for p in completed]
        manifest["failed"] = failed
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    print(f"\nFinished: {len(completed)}/{len(orders)} runs completed"
          + (f", {len(failed)} failed" if failed else ""))


if __name__ == "__main__":
    main()
