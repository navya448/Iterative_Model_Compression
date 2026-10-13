"""Run the IMP rewind experiment for multiple independent random seeds.

The existing train_dense.py and run_imp.py scripts remain the source of truth
for one experiment. This wrapper gives every seed its own run directory, so
the experiment can be resumed safely after an interruption.

Run from imp_rewind/:
    python repeated_experiments/run_repeated.py

By default this runs seeds 1 through 10 and then combines them with
analyze.py. Use --epochs and --rounds for a smaller validation run.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
DEFAULT_RUNS_DIR = os.path.join(HERE, "runs")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="independent seeds to run (default: 1 2 ... 10)",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--runs_dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--prune_frac", type=float, default=0.2)
    parser.add_argument("--tol", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--t_min", type=int, default=2)
    parser.add_argument("--bn_batches", type=int, default=100)
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="run the per-seed experiments but do not run analyze.py",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="continue with later seeds if one seed fails",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print commands without starting training",
    )
    return parser.parse_args()


def command_text(command):
    return subprocess.list2cmdline(command)


def run_command(command, log_path, dry_run):
    print("  " + command_text(command))
    if dry_run:
        return
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + command_text(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}"
        )


def main():
    args = parse_args()
    seeds = sorted(set(args.seeds))
    if not seeds or any(seed < 0 for seed in seeds):
        raise SystemExit("--seeds must contain at least one non-negative integer")

    os.makedirs(args.runs_dir, exist_ok=True)
    manifest_path = os.path.join(args.runs_dir, "experiment_manifest.json")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "epochs": args.epochs,
        "rounds": args.rounds,
        "prune_frac": args.prune_frac,
        "tol": args.tol,
        "resolution": args.resolution,
        "t_min": args.t_min,
        "bn_batches": args.bn_batches,
    }
    if not args.dry_run:
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

    completed_runs = []
    failed = []
    for seed in seeds:
        run_dir = os.path.join(args.runs_dir, f"seed{seed}")
        log_path = os.path.join(args.runs_dir, f"seed{seed}.log")
        dense_command = [
            sys.executable,
            os.path.join(PROJECT_DIR, "train_dense.py"),
            "--seed",
            str(seed),
            "--run_dir",
            run_dir,
            "--data_dir",
            args.data_dir,
        ]
        if args.epochs is not None:
            dense_command.extend(["--epochs", str(args.epochs)])

        imp_command = [
            sys.executable,
            os.path.join(PROJECT_DIR, "run_imp.py"),
            "--seed",
            str(seed),
            "--run_dir",
            run_dir,
            "--data_dir",
            args.data_dir,
            "--rounds",
            str(args.rounds),
            "--prune_frac",
            str(args.prune_frac),
            "--tol",
            str(args.tol),
            "--resolution",
            str(args.resolution),
            "--t_min",
            str(args.t_min),
            "--bn_batches",
            str(args.bn_batches),
        ]

        print(f"\nSeed {seed} ({len(completed_runs) + 1}/{len(seeds)})")
        try:
            run_command(dense_command, log_path, args.dry_run)
            run_command(imp_command, log_path, args.dry_run)
            completed_runs.append(run_dir)
        except RuntimeError as error:
            failed.append({"seed": seed, "error": str(error)})
            print(f"  FAILED: {error}")
            if not args.continue_on_error:
                raise

    if not args.skip_analysis:
        if failed and not args.continue_on_error:
            raise RuntimeError("analysis skipped because a seed failed")
        if not args.dry_run and completed_runs:
            analysis_dir = os.path.join(HERE, "analysis")
            analysis_command = [
                sys.executable,
                os.path.join(PROJECT_DIR, "analyze.py"),
                "--runs",
                *completed_runs,
                "--out",
                analysis_dir,
            ]
            run_command(
                analysis_command,
                os.path.join(args.runs_dir, "analysis.log"),
                args.dry_run,
            )

    if not args.dry_run:
        manifest["completed_seeds"] = [
            int(os.path.basename(path).replace("seed", ""))
            for path in completed_runs
        ]
        manifest["failed"] = failed
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
    print(
        f"\nFinished: {len(completed_runs)}/{len(seeds)} seeds completed"
        + (f", {len(failed)} failed" if failed else "")
    )


if __name__ == "__main__":
    main()
