"""Shared loading and helper code for the 10-seed evaluation.

Nothing here modifies raw results. Every script in this folder reads the run
directories, writes only into this folder's output directories, and can be
re-run from scratch.

Run registry
------------
* repeated_experiments/runs/seed1 .. seed10   (the new 10-seed runs)
* runs/seed1, runs/seed2                      (the two original discovery runs,
                                               labelled seed1_orig / seed2_orig)
The original runs were used to design the t* formula, so they may appear in
training sets but are never used as test seeds (see 03_formula_eval.py).
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IMP_REWIND = os.path.abspath(os.path.join(HERE, "..", ".."))
REPEATED_RUNS = os.path.join(IMP_REWIND, "repeated_experiments", "runs")
ORIGINAL_RUNS = os.path.join(IMP_REWIND, "runs")
OUT = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
AUDIT = os.path.join(HERE, "audit")
MANIFEST = os.path.join(AUDIT, "run_manifest.json")

# Protocol every run is expected to follow (imp_rewind/README.md, common.DEFAULTS).
REF_CFG = dict(epochs=80, milestones=[40, 60], gamma=0.1, lr=0.1, momentum=0.9,
               weight_decay=1e-4, batch_size=128, tol=0.5, prune_frac=0.2, t_min=2,
               resolution=2, val_size=5000, split_seed=0, bn_batches=100,
               train_eval_size=10000, rounds=10)

DISCOVERY_LABELS = ("seed1_orig", "seed2_orig")


def registry():
    """(label, path, group) for every run directory considered."""
    out = []
    for p in [os.path.join(ORIGINAL_RUNS, "seed1"), os.path.join(ORIGINAL_RUNS, "seed2")]:
        if os.path.isdir(p):
            out.append((os.path.basename(p) + "_orig", p, "original"))
    for p in sorted(glob.glob(os.path.join(REPEATED_RUNS, "seed*")),
                    key=lambda s: int(re.sub(r"\D", "", os.path.basename(s)) or 0)):
        if os.path.isdir(p):
            out.append((os.path.basename(p), p, "repeated"))
    return out


def lr_at_epoch(epoch, cfg):
    """LR used DURING 0-indexed epoch `epoch` (same as common.lr_at_epoch)."""
    lr = cfg["lr"]
    for m in cfg["milestones"]:
        if epoch >= m:
            lr *= cfg["gamma"]
    return lr


def budget_table(dense):
    """B[t] = sum of the per-epoch LR over epochs t..T-1, in units of the peak LR.

    Row e of dense_log.csv holds the LR used for epoch e-1 -> e, so the epochs
    run after rewinding to t are rows t+1..T."""
    lr = dense["lr"].fillna(0).to_numpy(dtype=float)
    ep = dense["epoch"].to_numpy(dtype=int)
    T = int(ep.max())
    peak = lr.max()
    return np.array([lr[ep > t].sum() / peak for t in range(T + 1)])


def b_inverse(B, b_req, t_min):
    """Latest integer t in [t_min, T] whose remaining budget B[t] >= b_req."""
    T = len(B) - 1
    if b_req <= 0:
        return T
    ok = np.where(B[t_min:] >= b_req - 1e-12)[0]
    return int(t_min + ok.max()) if len(ok) else t_min


def read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def load_run(label, path):
    imp = os.path.join(path, "imp")
    dense_dir = os.path.join(path, "dense")
    run = dict(label=label, path=path)
    run["cfg"] = json.load(open(os.path.join(imp, "config.json"))) if os.path.exists(
        os.path.join(imp, "config.json")) else {}
    run["dense_final"] = json.load(open(os.path.join(dense_dir, "final_metrics.json"))) if os.path.exists(
        os.path.join(dense_dir, "final_metrics.json")) else {}
    run["dense_raw"] = read_csv(os.path.join(dense_dir, "dense_log.csv"))
    d = run["dense_raw"]
    run["dense"] = (d.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)
                    if len(d) else d)
    run["rounds"] = read_csv(os.path.join(imp, "rounds.csv"))
    run["trials_csv"] = read_csv(os.path.join(imp, "rewind_trials.csv"))
    trials = []
    for f in sorted(glob.glob(os.path.join(imp, "round_*", "trial_*.json"))):
        m = json.load(open(f))
        m["file"] = os.path.relpath(f, path)
        trials.append(m)
    run["trials_json"] = pd.DataFrame(trials)
    run["summaries"] = {}
    for f in sorted(glob.glob(os.path.join(imp, "round_*", "summary.json"))):
        rnd = int(re.search(r"round_(\d+)", f).group(1))
        run["summaries"][rnd] = json.load(open(f))
    run["round_dirs"] = sorted(int(re.search(r"round_(\d+)", p).group(1))
                               for p in glob.glob(os.path.join(imp, "round_*")) if os.path.isdir(p))
    run["curves"] = sorted(glob.glob(os.path.join(imp, "round_*", "curve_t*.csv")))
    log = os.path.join(os.path.dirname(path), os.path.basename(path) + ".log")
    run["log"] = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else None
    return run


def load_manifest():
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"{MANIFEST} not found: run 01_audit_runs.py first")
    return json.load(open(MANIFEST))


def build_tables(manifest):
    """Rounds and trials of every INCLUDED round of every INCLUDED run, plus the
    per-run constants the predictors need. Excluded runs/rounds are listed in the
    manifest with a reason; nothing is dropped silently."""
    rounds, trials, runs = [], [], {}
    for entry in manifest["runs"]:
        if not entry["include"]:
            continue
        run = load_run(entry["label"], entry["path"])
        T = int(run["dense"]["epoch"].max())
        B = budget_table(run["dense"])
        a_dense = float(run["dense_final"]["val_acc"])
        tol = float(run["cfg"]["tol"])
        runs[entry["label"]] = dict(B=B, T=T, a_dense=a_dense, tol=tol,
                                    target=float(run["cfg"]["target_val_acc"]),
                                    t_min=int(run["cfg"]["t_min"]), dense=run["dense"],
                                    cfg=run["cfg"], group=entry["group"])
        status = {int(k): v for k, v in entry["rounds"].items()}
        r = run["rounds"].copy()
        r["run"] = entry["label"]
        r["group"] = entry["group"]
        r["round_status"] = r["round"].map(status)
        r = r[r["round_status"].isin(["no_retrain", "searched", "unrecoverable"])]
        rounds.append(r)
        # sparsity of incomplete rounds (needed only for their trials)
        sp = dict(zip(run["rounds"]["round"], run["rounds"]["sparsity"]))
        rho = dict(zip(run["rounds"]["round"], run["rounds"]["weight_perturbation"]))
        tj = run["trials_json"].copy()
        if len(tj):
            tj["run"] = entry["label"]
            tj["round_status"] = tj["round"].map(status)
            tj = tj[tj["round_status"].isin(["searched", "unrecoverable", "incomplete"])]
            prune_frac = float(run["cfg"]["prune_frac"])
            tj["sparsity"] = [sp.get(k, 1 - (1 - prune_frac) ** k) for k in tj["round"]]
            tj["weight_perturbation"] = [rho.get(k, np.nan) for k in tj["round"]]
            tj["a_dense"] = a_dense
            tj["target_val_acc"] = runs[entry["label"]]["target"]
            tj["y"] = tj["final_val_acc"] - a_dense
            tj["ln_inv_d"] = -np.log(1 - tj["sparsity"])
            tj["B"] = [B[int(t)] for t in tj["t"]]
            tj["clock"] = np.log(B[runs[entry["label"]]["t_min"]] / tj["B"])
            trials.append(tj)
    rounds = pd.concat(rounds, ignore_index=True)
    rounds["ln_inv_d"] = -np.log(1 - rounds["sparsity"])
    rounds["needs_retrain"] = rounds["after_val_acc"] < rounds["target_val_acc"]
    trials = pd.concat(trials, ignore_index=True)
    return rounds, trials, runs
