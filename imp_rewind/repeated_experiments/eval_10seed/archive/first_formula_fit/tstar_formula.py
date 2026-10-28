"""Budget-matching formula for the rewind point t*.

Model (fitted on every rewind trial, not just on t*):

    A(t) - A_dense = a0 - a1 * ln(1/d) - kappa * ln( B(t_min) / B(t) )

    A(t)   final val acc after rewinding to epoch t and retraining to T
    d      density (1 - sparsity) of the round's mask
    B(t)   remaining LR budget = sum of the per-epoch LR from t to T-1,
           divided by the peak LR (so it is in "epochs at peak LR")

Setting A(t) = A_dense - tol and solving for t gives the rule

    t_hat = T                                   if the pruned model already meets the target
    t_hat = B^-1( B(t_min) * exp(-S / kappa) )  otherwise,  S = tol + a0 - a1*ln(1/d)
    t_hat = t_min, flagged "at capacity limit"  if S <= 0 (no rewind is expected to recover)

S is the accuracy slack the sparse network has at full budget, kappa is how much
accuracy each e-fold of lost LR budget costs. Everything the rule needs at
prediction time (d, the pruned model's val acc, the LR schedule) is known before
any retraining.

Usage (from imp_rewind/):
    python tstar_formula.py --runs runs/seed1 runs/seed2 --out analysis_formula
    python tstar_formula.py --runs repeated_experiments/runs/seed* --out repeated_experiments/analysis_formula
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- loading

def load_run(run_dir):
    imp = os.path.join(run_dir, "imp")
    cfg = json.load(open(os.path.join(imp, "config.json")))
    dense = pd.read_csv(os.path.join(run_dir, "dense", "dense_log.csv"))
    dense = dense.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)
    rounds = pd.read_csv(os.path.join(imp, "rounds.csv"))
    trials_path = os.path.join(imp, "rewind_trials.csv")
    trials = pd.read_csv(trials_path) if os.path.exists(trials_path) else pd.DataFrame()
    name = os.path.basename(os.path.normpath(run_dir))
    rounds["run"] = name
    if len(trials):
        trials["run"] = name
    tol = cfg.get("tol", 0.5)
    target = cfg["target_val_acc"]
    return dict(name=name, cfg=cfg, dense=dense, rounds=rounds, trials=trials,
                tol=tol, target=target, dense_acc=target + tol,
                T=int(dense["epoch"].max()), t_min=int(cfg.get("t_min", 2)))


def budget_fn(dense):
    """B(t) = sum of LR over epochs t..T-1, in units of peak LR.
    Row e of the dense log holds the LR used for epoch e-1 -> e."""
    lr = dense["lr"].fillna(0).to_numpy()
    ep = dense["epoch"].to_numpy()
    peak = lr.max()
    T = int(ep.max())
    table = np.array([lr[ep > t].sum() / peak for t in range(T + 1)])
    return table  # table[t] = B(t)


# ---------------------------------------------------------------- model

def trial_frame(runs):
    rows = []
    for r in runs:
        if not len(r["trials"]):
            continue
        Bt = budget_fn(r["dense"])
        tr = r["trials"].merge(r["rounds"][["round", "sparsity", "after_val_acc"]], on="round")
        tr["y"] = tr["final_val_acc"] - r["dense_acc"]
        tr["ln_inv_d"] = -np.log(1 - tr["sparsity"])
        tr["clock"] = np.log(Bt[r["t_min"]] / np.maximum(Bt[tr["t"].astype(int)], 1e-12))
        rows.append(tr)
    return pd.concat(rows, ignore_index=True)


def fit(trials):
    X = np.column_stack([np.ones(len(trials)), -trials["ln_inv_d"], -trials["clock"]])
    w, *_ = np.linalg.lstsq(X, trials["y"].to_numpy(), rcond=None)
    resid = X @ w - trials["y"].to_numpy()
    return dict(a0=w[0], a1=w[1], kappa=w[2], rmse=float(np.sqrt(np.mean(resid ** 2))))


def predict_t(p, sparsity, after_val_acc, target, tol, Bt, t_min, margin=0.0):
    """Returns (t_hat, flag). flag in {"no_retrain", "ok", "capacity_limit"}."""
    T = len(Bt) - 1
    if after_val_acc >= target:
        return T, "no_retrain"
    S = tol + p["a0"] - p["a1"] * np.log(1 / (1 - sparsity)) - margin
    if S <= 0:
        return t_min, "capacity_limit"
    B_req = Bt[t_min] * np.exp(-S / p["kappa"])
    ok = [t for t in range(t_min, T + 1) if Bt[t] >= B_req - 1e-12]
    return (max(ok) if ok else t_min), "ok"


def predicted_acc(p, sparsity, t, Bt, t_min):
    return p["a0"] - p["a1"] * np.log(1 / (1 - sparsity)) - p["kappa"] * np.log(Bt[t_min] / max(Bt[t], 1e-12))


# ---------------------------------------------------------------- evaluation

def evaluate(runs, margin):
    """Leave-one-run-out: fit on all other runs' trials, predict this run's t*."""
    out = []
    for r in runs:
        others = [o for o in runs if o["name"] != r["name"]]
        if not others:
            continue
        p = fit(trial_frame(others))
        Bt = budget_fn(r["dense"])
        # baseline from the mid-term report: least-squares line on sparsity, fitted on the other runs
        base = pd.concat([o["rounds"] for o in others])
        base = base[base["t_star"].notna()]
        b1, b0 = np.polyfit(base["sparsity"], base["t_star"], 1)
        for _, x in r["rounds"].iterrows():
            t_hat, flag = predict_t(p, x["sparsity"], x["after_val_acc"], r["target"], r["tol"], Bt, r["t_min"], margin)
            out.append(dict(run=r["name"], round=int(x["round"]), sparsity=x["sparsity"], t_star=x["t_star"],
                            t_hat=t_hat, flag=flag,
                            t_base=int(np.clip(round(b0 + b1 * x["sparsity"]), r["t_min"], r["dense"]["epoch"].max()))))
    return pd.DataFrame(out)


def summarize(ev, col):
    e = ev[ev["t_star"].notna()]
    err = e[col] - e["t_star"]
    return dict(n=len(e), mae=err.abs().mean(), too_late=int((err > 0).sum()),
                wasted_when_early=(-err[err < 0]).mean() if (err < 0).any() else 0.0)


# ---------------------------------------------------------------- figure

def make_figure(runs, p, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": surf, "axes.facecolor": surf})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) collapse: remove the capacity term, every trial should fall on one line of slope -kappa
    tr = trial_frame(runs)
    tr["collapsed"] = tr["y"] + p["a1"] * tr["ln_inv_d"] - p["a0"]
    xs = np.linspace(0, tr["clock"].max() * 1.02, 50)
    for i, (name, g) in enumerate(tr.groupby("run")):
        ax1.scatter(g["clock"], g["collapsed"], s=26, color=colors[i % 8], edgecolor=surf, linewidth=1.2,
                    label=name, zorder=3)
    ax1.plot(xs, -p["kappa"] * xs, color=ink, lw=2, zorder=2)
    ax1.text(xs[-1] * 0.62, -p["kappa"] * xs[-1] * 0.62 + 0.42, "slope = -%.3f pp\nper e-fold of budget" % p["kappa"],
             ha="left", va="bottom", color=ink, fontsize=9)
    Bt0 = budget_fn(runs[0]["dense"])
    t0 = runs[0]["t_min"]
    for t in list(runs[0]["cfg"].get("milestones", [])) + [len(Bt0) - 3]:
        x = np.log(Bt0[t0] / Bt0[t])
        ax1.axvline(x, color=ink2, lw=0.8, ls=":")
        ax1.text(x + 0.06, ax1.get_ylim()[1] - 0.02, "t=%d" % t, color=ink2, fontsize=8, va="top")
    ax1.set_xlabel("ln( B(t_min) / B(t) )   (e-folds of LR budget given up by rewinding later)")
    ax1.set_ylabel("val acc after retrain, minus capacity term (pp)")
    ax1.set_title("(a) Every rewind trial vs remaining LR budget", loc="left", color=ink, fontsize=11)
    ax1.grid(color=grid, lw=0.8)
    ax1.legend(frameon=False, fontsize=9)

    # (b) predicted t_hat(s) vs measured t*
    r0 = runs[0]
    Bt = budget_fn(r0["dense"])
    T = len(Bt) - 1
    s_grid = np.linspace(0.40, 0.95, 400)
    t_line = []
    for s in s_grid:
        t, flag = predict_t(p, s, -np.inf, r0["target"], r0["tol"], Bt, r0["t_min"])
        t_line.append(np.nan if flag == "capacity_limit" else t)
    s_lim = 1 - np.exp(-(r0["tol"] + p["a0"]) / p["a1"])
    ax2.axvspan(s_lim * 100, 95, color=grid, alpha=0.7, lw=0)
    ax2.text(s_lim * 100 + 0.4, 74, "predicted\ncapacity limit\n(s > %.1f%%)" % (s_lim * 100), color=ink2, fontsize=8.5, va="top")
    for m in r0["cfg"].get("milestones", []):
        ax2.axhline(m, color=ink2, lw=0.8, ls=":")
        ax2.text(41, m + 0.8, "LR drop (epoch %d)" % m, color=ink2, fontsize=8)
    ax2.plot(s_grid * 100, t_line, color=ink, lw=2, label="formula (fit on all runs)", zorder=2)
    for i, r in enumerate(runs):
        rr = r["rounds"]
        need = rr[rr["after_val_acc"] < r["target"]]
        ok = need[need["t_star"].notna()]
        ax2.scatter(ok["sparsity"] * 100, ok["t_star"], s=40, color=colors[i % 8], edgecolor=surf, linewidth=1.5,
                    zorder=3, label="%s measured t*" % r["name"])
        fail = need[need["t_star"].isna()]
        if len(fail):
            ax2.scatter(fail["sparsity"] * 100 - 0.9, [r["t_min"]] * len(fail), marker="x", s=50, color=colors[i % 8],
                        zorder=4, label="%s: even t=%d failed" % (r["name"], r["t_min"]))
    ax2.set_xlim(40, 95)
    ax2.set_ylim(0, T + 2)
    ax2.set_xlabel("sparsity (%)")
    ax2.set_ylabel("rewind epoch")
    ax2.set_title("(b) Predicted t̂ vs measured t* (rounds that needed retraining)", loc="left", color=ink, fontsize=11)
    ax2.grid(color=grid, lw=0.8)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="analysis_formula")
    ap.add_argument("--margin", type=float, default=0.1, help="safety margin in pp subtracted from the slack")
    args = ap.parse_args()

    run_dirs = sorted({d for pat in args.runs for d in glob.glob(pat)})
    runs = [load_run(d) for d in run_dirs]
    os.makedirs(args.out, exist_ok=True)

    p = fit(trial_frame(runs))
    s_lim = 1 - np.exp(-(runs[0]["tol"] + p["a0"]) / p["a1"])
    print("Fit on %d trials from %d runs:" % (len(trial_frame(runs)), len(runs)))
    print("  a0 = %.3f pp, a1 = %.3f pp per ln(1/d), kappa = %.4f pp per e-fold of LR budget, trial RMSE = %.3f pp"
          % (p["a0"], p["a1"], p["kappa"], p["rmse"]))
    print("  capacity limit (slack = 0) at sparsity %.1f%%" % (100 * s_lim))
    json.dump(p, open(os.path.join(args.out, "params.json"), "w"), indent=2)

    if len(runs) > 1:
        for m in sorted({0.0, args.margin}):
            ev = evaluate(runs, m)
            ev.to_csv(os.path.join(args.out, "loo_predictions_margin%.2f.csv" % m), index=False)
            f, b = summarize(ev, "t_hat"), summarize(ev, "t_base")
            print("\nLeave-one-run-out, margin %.2f pp (%d rounds with a measured t*):" % (m, f["n"]))
            print("  formula        MAE %.1f epochs, too late %d, mean wasted when early %.1f" % (f["mae"], f["too_late"], f["wasted_when_early"]))
            print("  sparsity line  MAE %.1f epochs, too late %d, mean wasted when early %.1f" % (b["mae"], b["too_late"], b["wasted_when_early"]))
        print(ev[["run", "round", "sparsity", "t_star", "t_hat", "flag", "t_base"]].to_string(index=False))

    make_figure(runs, p, os.path.join(args.out, "tstar_formula.png"))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
