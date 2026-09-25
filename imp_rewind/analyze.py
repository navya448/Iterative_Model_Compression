"""Experiment C: look for the formula.

Reads the dense log and IMP results of one or more seeds and produces:
  all_rounds.csv       every IMP round of every seed in one table
  rule_scores.csv      how well each candidate rule predicts t*
  feature_corr.csv     how strongly t* follows each damage feature
                       (overall, and WITHIN a round across seeds, which removes
                        the "both just grow with sparsity" confound)
  figures/*.png        the plots from the plan

Usage:
  python analyze.py --runs runs/seed1 runs/seed2 --out analysis
  # later, the held-out test: score the rules on seed 3 only
  python analyze.py --runs runs/seed1 runs/seed2 --test_runs runs/seed3 --out analysis
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rules import load_dense_log, specialization_point

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"

RULES = {
    "pred_loss_match_val": "Loss match (val)",
    "pred_loss_match_train": "Loss match (train)",
    "pred_loss_match_val_vs_dense": "Loss match (val, vs dense)",
    "pred_acc_match_val": "Accuracy match (val)",
    "pred_movement_match": "Weight-movement match",
    "pred_specialization": "Specialization point",
    "pred_baseline_5pct": "Baseline: 5% of T",
    "pred_baseline_before_first_lr_drop": "Baseline: before 1st LR drop",
    "pred_baseline_before_last_lr_drop": "Baseline: before last LR drop",
}
FEATURES = ["D_loss_val", "D_loss_train", "D_acc_val", "D_loss_val_vs_dense",
            "weight_perturbation", "sparsity"]


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def load_run(run_dir):
    dense = load_dense_log(os.path.join(run_dir, "dense", "dense_log.csv"))
    rounds_path = os.path.join(run_dir, "imp", "rounds.csv")
    trials_path = os.path.join(run_dir, "imp", "rewind_trials.csv")
    rounds = pd.read_csv(rounds_path) if os.path.exists(rounds_path) else pd.DataFrame()
    trials = pd.read_csv(trials_path) if os.path.exists(trials_path) else pd.DataFrame()
    cfg_path = os.path.join(run_dir, "imp", "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"milestones": [40, 60]}
    name = os.path.basename(os.path.normpath(run_dir))
    for df in (rounds, trials):
        if len(df):
            df["run"] = name
    return dict(name=name, dense=dense, rounds=rounds, trials=trials, cfg=cfg)


def score_rules(rounds):
    """Error of each rule vs the real t*. 'Too late' predictions would fail to
    recover (assuming later rewinds recover less); 'too early' ones waste epochs."""
    r = rounds.dropna(subset=["t_star"])
    rows = []
    for col, label in RULES.items():
        if col not in r or r[col].isna().all():
            continue
        d = r.dropna(subset=[col])
        err = d[col] - d["t_star"]
        rho = d[col].corr(d["t_star"], method="spearman") if d[col].nunique() > 1 else np.nan
        rows.append(dict(
            rule=label, n=len(d),
            mean_abs_error_epochs=err.abs().mean(),
            mean_signed_error=err.mean(),
            n_too_late_would_fail=int((err > 0).sum()),
            mean_wasted_epochs_when_early=(-err[err <= 0]).mean() if (err <= 0).any() else 0.0,
            spearman_with_t_star=rho,
        ))
    return pd.DataFrame(rows).sort_values("mean_abs_error_epochs")


def feature_correlations(rounds):
    r = rounds.dropna(subset=["t_star"]).copy()
    rows = []
    for f in FEATURES:
        if f not in r:
            continue
        overall = r[f].corr(r["t_star"], method="spearman")
        # within-round: subtract each round's mean across seeds, then correlate
        g = r.groupby("round")
        sizes = g["t_star"].transform("size")
        w = r[sizes > 1]
        if len(w) >= 3:
            dt = w["t_star"] - w.groupby("round")["t_star"].transform("mean")
            df = w[f] - w.groupby("round")[f].transform("mean")
            within = df.corr(dt) if dt.std() > 0 and df.std() > 0 else np.nan
        else:
            within = np.nan
        rows.append(dict(feature=f, spearman_overall=overall,
                         pearson_within_round_across_seeds=within))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def fig_dense(run, out):
    d, ms = run["dense"], run["cfg"].get("milestones", [40, 60])
    ts = specialization_point(d)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, unit in [(axes[0], "loss", "cross-entropy loss"), (axes[1], "acc", "accuracy (%)")]:
        for i, split in enumerate(["train", "val"]):
            ax.plot(d["epoch"], d[f"{split}_{metric}"], color=COLORS[i], linewidth=2,
                    label="train subset" if split == "train" else "validation")
        for m in ms:
            ax.axvline(m, color=MUTED, linestyle=":", linewidth=1)
        if ts is not None:
            ax.axvline(ts, color=COLORS[2], linestyle="--", linewidth=1.5, label=f"t_s = {ts}")
        style(ax, f"Dense {metric} ({run['name']})", "epoch", unit)
        ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Dotted lines: LR drops." + ("" if ts is not None else
             "  No specialization point detected."), color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, f"dense_curves_{run['name']}.png"), dpi=150)
    plt.close(fig)


def fig_rewind_curves(run, out):
    """Final val acc vs rewind epoch, one panel per round, target marked."""
    tr = run["trials"]
    if not len(tr):
        return
    rounds = sorted(tr["round"].unique())
    ncol = min(5, len(rounds))
    nrow = int(np.ceil(len(rounds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.6 * nrow), squeeze=False)
    for ax, rnd in zip(axes.flat, rounds):
        d = tr[tr["round"] == rnd].sort_values("t")
        ax.plot(d["t"], d["final_val_acc"], color=COLORS[0], linewidth=2, marker="o", markersize=5)
        ax.axhline(d["target_val_acc"].iloc[0], color=COLORS[1], linestyle="--", linewidth=1.2)
        rr = run["rounds"][run["rounds"]["round"] == rnd]
        if len(rr) and pd.notna(rr["t_star"].iloc[0]):
            ax.axvline(rr["t_star"].iloc[0], color=COLORS[2], linewidth=1.2)
        style(ax, f"Round {rnd}", "rewind epoch t", "final val acc (%)")
    for ax in list(axes.flat)[len(rounds):]:
        ax.axis("off")
    fig.suptitle(f"Recovery vs rewind point ({run['name']}): dashed = target, solid = t*",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"rewind_curves_{run['name']}.png"), dpi=150)
    plt.close(fig)


def fig_tstar_vs_features(rounds, out):
    r = rounds.dropna(subset=["t_star"])
    feats = [f for f in ["D_loss_val", "D_acc_val", "weight_perturbation", "sparsity"] if f in r]
    fig, axes = plt.subplots(1, len(feats), figsize=(3.4 * len(feats), 3.3), squeeze=False)
    for ax, f in zip(axes[0], feats):
        for i, (name, d) in enumerate(r.groupby("run")):
            ax.scatter(d[f], d["t_star"], s=40, color=COLORS[i % 6], marker=MARKERS[i % 6],
                       edgecolor="white", linewidth=1, label=name)
        style(ax, f"t* vs {f}", f, "t* (epoch)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "t_star_vs_features.png"), dpi=150)
    plt.close(fig)


def fig_pred_vs_actual(rounds, out, fname, title):
    r = rounds.dropna(subset=["t_star"])
    cols = [c for c in RULES if c in r and r[c].notna().any()]
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow), squeeze=False)
    hi = max(r["t_star"].max(), r[cols].max().max()) + 3
    for ax, c in zip(axes.flat, cols):
        ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1, linestyle="--")
        for i, (name, d) in enumerate(r.groupby("run")):
            ax.scatter(d["t_star"], d[c], s=40, color=COLORS[i % 6], marker=MARKERS[i % 6],
                       edgecolor="white", linewidth=1, label=name)
        mae = (r[c] - r["t_star"]).abs().mean()
        style(ax, f"{RULES[c]}\nMAE {mae:.1f} epochs", "actual t*", "predicted t")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title + "  (above the diagonal = too late, would not recover)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=150)
    plt.close(fig)


def fig_tstar_by_round(rounds, out):
    r = rounds.dropna(subset=["t_star"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (name, d) in enumerate(r.groupby("run")):
        axes[0].plot(d["sparsity"] * 100, d["t_star"], color=COLORS[i % 6], marker=MARKERS[i % 6],
                     linewidth=2, markersize=6, label=name)
        if "lr_remaining_frac" in d:
            axes[1].plot(d["sparsity"] * 100, d["lr_remaining_frac"], color=COLORS[i % 6],
                         marker=MARKERS[i % 6], linewidth=2, markersize=6, label=name)
    style(axes[0], "t* in epochs", "sparsity (%)", "t* (epoch)")
    style(axes[1], "Same, measured as share of LR budget left", "sparsity (%)",
          "fraction of total LR still ahead")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "t_star_by_sparsity.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="seeds used to FIND the formula")
    p.add_argument("--test_runs", nargs="*", default=[], help="held-out seeds, only scored")
    p.add_argument("--out", default="analysis")
    args = p.parse_args()
    figs = os.path.join(args.out, "figures")
    os.makedirs(figs, exist_ok=True)

    runs = [load_run(r) for r in args.runs]
    tests = [load_run(r) for r in args.test_runs]
    for run in runs + tests:
        fig_dense(run, figs)
        fig_rewind_curves(run, figs)

    rounds = pd.concat([r["rounds"] for r in runs if len(r["rounds"])], ignore_index=True)
    rounds.to_csv(os.path.join(args.out, "all_rounds.csv"), index=False)

    scores = score_rules(rounds)
    corr = feature_correlations(rounds)
    scores.to_csv(os.path.join(args.out, "rule_scores.csv"), index=False)
    corr.to_csv(os.path.join(args.out, "feature_corr.csv"), index=False)
    fig_tstar_vs_features(rounds, figs)
    fig_pred_vs_actual(rounds, figs, "predicted_vs_actual.png", "Discovery seeds")
    fig_tstar_by_round(rounds, figs)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== t* per round ===")
    show = ["run", "round", "sparsity", "D_loss_val", "D_acc_val", "weight_perturbation",
            "t_star", "retrain_epochs", "pred_loss_match_val"]
    print(rounds[[c for c in show if c in rounds]].round(4).to_string(index=False))
    print("\n=== How well each rule predicts t* (discovery seeds) ===")
    print(scores.round(2).to_string(index=False))
    print("\n=== How strongly t* follows each feature ===")
    print(corr.round(3).to_string(index=False))

    if tests:
        test_rounds = pd.concat([r["rounds"] for r in tests if len(r["rounds"])], ignore_index=True)
        test_rounds.to_csv(os.path.join(args.out, "test_rounds.csv"), index=False)
        tscores = score_rules(test_rounds)
        tscores.to_csv(os.path.join(args.out, "rule_scores_test.csv"), index=False)
        fig_pred_vs_actual(test_rounds, figs, "predicted_vs_actual_test.png", "Held-out seed(s)")
        print("\n=== Held-out test: how well each rule predicts t* ===")
        print(tscores.round(2).to_string(index=False))

    print(f"\nTables in {args.out}/, figures in {figs}/")


if __name__ == "__main__":
    main()
