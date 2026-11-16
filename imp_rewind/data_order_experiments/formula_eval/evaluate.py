"""Analysis of the data-order experiment and the check of the rewind formula.

Reads (never modifies):
  imp_rewind/runs/seed1, seed2                       10-seed discovery runs (tol 0.5)
  imp_rewind/repeated_experiments/runs/seed3..10     10-seed new runs (tol 0.5)
  imp_rewind/data_order_experiments/runs/<run>/orderNN   data-order runs (tol 2.5)

Repeated seeds 1 and 2 are left out (their round 1 came from the 2-epoch smoke test,
see eval_10seed/audit/AUDIT.md on branch eval-10seed). Trials of rounds without a
rounds.csv row (seed10 round 8, interrupted) are left out of every fit.

Writes results/*.csv, results/numbers.txt and figures/*.png next to this file.

    python evaluate.py [--data_order_run 20261102_165000]
"""

import argparse
import glob
import json
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from formula import add_round_features, b_inverse, budget_table, original, revised

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
IMP_REWIND = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
C10, CDO, CMOD = "#2a78d6", "#eb6834", "#6b6a64"     # 10-seed, data order, model lines
LABEL = {"10s": "10 seeds (α = 0.5)", "do": "10 data orders (α = 2.5)"}
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


# ------------------------------------------------------------------ loading

def registry(do_run):
    regs = [("seed1_orig", os.path.join(IMP_REWIND, "runs", "seed1"), "10s"),
            ("seed2_orig", os.path.join(IMP_REWIND, "runs", "seed2"), "10s")]
    regs += [(f"seed{k}", os.path.join(IMP_REWIND, "repeated_experiments", "runs", f"seed{k}"), "10s")
             for k in range(3, 11)]
    regs += [(os.path.basename(p), p, "do")
             for p in sorted(glob.glob(os.path.join(IMP_REWIND, "data_order_experiments", "runs", do_run, "order*")))
             if os.path.isdir(p)]
    return regs


def load(do_run):
    rounds, trials, runs = [], [], {}
    for label, path, ds in registry(do_run):
        cfg = json.load(open(os.path.join(path, "imp", "config.json")))
        dense = pd.read_csv(os.path.join(path, "dense", "dense_log.csv"))
        dense = dense.drop_duplicates("epoch", keep="last").sort_values("epoch")
        a_dense = float(json.load(open(os.path.join(path, "dense", "final_metrics.json")))["val_acc"])
        B = budget_table(dense)
        runs[label] = dict(B=B, T=int(dense["epoch"].max()), t_min=int(cfg["t_min"]), tol=float(cfg["tol"]),
                           a_dense=a_dense, ds=ds)
        r = pd.read_csv(os.path.join(path, "imp", "rounds.csv")).drop_duplicates("round", keep="last")
        r["run"], r["ds"], r["a_dense"] = label, ds, a_dense
        rounds.append(add_round_features(r))
        tj = pd.DataFrame([json.load(open(f)) for f in glob.glob(os.path.join(path, "imp", "round_*", "trial_*.json"))])
        tj = tj.merge(r[["round", "sparsity", "before_val_acc", "after_val_acc", "D_acc_val", "weight_perturbation",
                         "ln_inv_d", "parent_def", "ln_damage", "target_val_acc"]], on="round", how="inner")
        tj["run"], tj["ds"], tj["a_dense"] = label, ds, a_dense
        tj["y"] = tj["final_val_acc"] - a_dense
        tj["clock"] = np.log(B[runs[label]["t_min"]] / B[tj["t"].astype(int)])
        tj["recovered"] = tj["final_val_acc"] >= tj["target_val_acc"]
        trials.append(tj)
    R = pd.concat(rounds, ignore_index=True)
    R["needs_retrain"] = R["after_val_acc"] < R["target_val_acc"]
    return R, pd.concat(trials, ignore_index=True), runs


# ------------------------------------------------------------------ baselines

class SparsityLine:
    name = "Sparsity line + check"

    def fit(self, rounds):
        r = rounds[rounds["t_star"].notna() & rounds["needs_retrain"]]
        self.b1, self.b0 = np.polyfit(r["sparsity"], r["t_star"], 1)
        return self

    def predict(self, row, run):
        if not row["needs_retrain"]:
            return run["T"]
        return int(np.clip(round(self.b0 + self.b1 * row["sparsity"]), run["t_min"], run["T"]))


class RoundMedian:
    name = "Round median + check"

    def fit(self, rounds):
        r = rounds[rounds["t_star"].notna() & rounds["needs_retrain"]]
        self.med = r.groupby("round")["t_star"].median().to_dict()
        return self

    def predict(self, row, run):
        if not row["needs_retrain"]:
            return run["T"]
        ks = np.array(sorted(self.med))
        return int(np.floor(self.med[ks[np.argmin(np.abs(ks - int(row["round"])))]]))


def scores(P):
    """t* metrics of a prediction table with columns t_star, t_hat."""
    P = P[P["t_star"].notna()]
    e = P["t_hat"] - P["t_star"]
    rt = P["t_star"] < 80
    return pd.Series(dict(n=len(P), MAE=e.abs().mean(), MAE_retrain=e[rt].abs().mean(),
                          signed_retrain=e[rt].mean(), too_late=(e > 0).mean()))


def boot_runs(df, rng, n):
    runs = df["run"].unique()
    for _ in range(n):
        yield pd.concat([df[df["run"] == r] for r in rng.choice(runs, len(runs))], ignore_index=True)


# ------------------------------------------------------------------ analysis

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_order_run", default="20261102_165000")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(0)
    R, Tr, runs = load(args.data_order_run)
    do, s10 = Tr["ds"] == "do", Tr["ds"] == "10s"
    say(f"rounds: {(R.ds == '10s').sum()} (10 seeds), {(R.ds == 'do').sum()} (data order); "
        f"trials: {s10.sum()}, {do.sum()}")

    # 1. original formula, as fitted on the 10-seed runs
    f10 = original().fit(Tr[s10])
    say("\n[1] original formula refit on the 10-seed runs:",
        {k: round(v, 4) for k, v in f10.coef.items()}, f"n={f10.n_fit} rmse={f10.train_rmse:.3f}")

    # 2. noise from data order alone
    dense = pd.DataFrame([dict(run=k, ds=v["ds"], a_dense=v["a_dense"]) for k, v in runs.items()])
    sd = dense.groupby("ds")["a_dense"].agg(["mean", "std", "count"])
    F = sd.loc["do", "std"] ** 2 / sd.loc["10s", "std"] ** 2
    say("\n[2] dense val acc:", sd.round(3).to_dict("index"),
        f"variance ratio {F:.2f}, one-sided p={1 - stats.f.cdf(F, 9, 9):.3f}")
    test_sd = np.std([json.load(open(os.path.join(p_, "dense", "final_metrics.json")))["test_acc"]
                      for l_, p_, d_ in registry(args.data_order_run) if d_ == "do"], ddof=1)
    say(f"    dense test acc SD across data orders: {test_sd:.3f}")
    t2 = Tr[Tr["t"] == 2].copy()
    t2["dev"] = t2["final_val_acc"] - t2.groupby(["ds", "round"])["final_val_acc"].transform("mean")
    per_run = t2.groupby(["ds", "run"])["dev"].mean().reset_index().merge(dense, on=["run", "ds"])
    per_run["dense_dev"] = per_run["a_dense"] - per_run.groupby("ds")["a_dense"].transform("mean")
    rho, pv = stats.spearmanr(per_run[per_run.ds == "do"]["dense_dev"], per_run[per_run.ds == "do"]["dev"])
    say(f"    does a data order's dense acc carry into its retrains? Spearman {rho:+.2f} (p={pv:.2f}, 10 orders)")
    g = t2.groupby(["ds", "round"])["final_val_acc"].agg(["std", "count"]).reset_index()
    g = g[g["count"] >= 3]
    pooled = g.groupby("ds").apply(lambda x: np.sqrt(((x["std"] ** 2) * (x["count"] - 1)).sum() / (x["count"] - 1).sum()), include_groups=False)
    say("    SD of a t=2 retrain across runs at the same round (pooled):", pooled.round(3).to_dict())
    spread = []
    for ds_, gR in R[(R.t_star < 80) & R.t_star.notna()].groupby("ds"):
        B = next(v["B"] for v in runs.values() if v["ds"] == ds_)
        gR = gR.assign(lnB=np.log(B[gR["t_star"].astype(int)]))
        s = gR.groupby("round").agg(n=("t_star", "size"), sd_t=("t_star", "std"), sd_lnB=("lnB", "std"))
        s = s[s.n >= 3]
        w = s.n - 1
        spread.append(dict(ds=ds_, sd_t=np.sqrt((s.sd_t ** 2 * w).sum() / w.sum()),
                           sd_lnB=np.sqrt((s.sd_lnB ** 2 * w).sum() / w.sum()), rounds=len(s)))
    spread = pd.DataFrame(spread).set_index("ds")
    say("    t* spread at the same round (rounds needing retraining):", spread.round(2).to_dict("index"))
    tsr = R[R.ds == "do"].pivot(index="round", columns="run", values="t_star")
    tsr.to_csv(os.path.join(OUT, "t_star_data_order.csv"))
    say("    data-order t* per round: mean/sd/min/max\n" +
        tsr.T.describe().T[["mean", "std", "min", "max"]].round(1).to_string())
    viol = sum(any((not a) and b for i, a in enumerate(gg.sort_values("t")["recovered"])
                   for b in list(gg.sort_values("t")["recovered"])[i + 1:])
               for _, gg in Tr[do].groupby(["run", "round"]))
    say(f"    data-order rounds where a later rewind recovered after an earlier one failed: {viol}")

    # 3. the chain under alpha = 2.5
    chain = R.groupby(["ds", "round"]).agg(parent_def=("parent_def", "mean"),
                                           after_def=("after_val_acc", "mean"), D_acc=("D_acc_val", "mean"),
                                           rho=("weight_perturbation", "mean")).reset_index()
    chain["after_def"] -= chain["ds"].map(dense.groupby("ds")["a_dense"].mean())
    chain.to_csv(os.path.join(OUT, "chain_by_round.csv"), index=False)
    say("\n[3] chain by round:\n" + chain.pivot(index="round", columns="ds",
                                               values=["parent_def", "D_acc", "rho"]).round(3).to_string())

    # 4. original formula applied to the data-order runs (fit on 10 seeds only)
    Tr["res_orig"] = Tr["y"] - f10.yhat(Tr)
    Tr["phase"] = pd.cut(Tr["t"], [0, 39, 59, 80], labels=["t<40", "40-59", "60+"])
    say("\n[4] original formula (10-seed fit) on data-order trials: RMSE "
        f"{np.sqrt((Tr[do].res_orig ** 2).mean()):.3f}, bias {Tr[do].res_orig.mean():+.3f} pp")
    say(Tr.pivot_table(index="ds", columns="phase", values="res_orig", aggfunc="mean", observed=True).round(3).to_string())
    P = pd.DataFrame([dict(run=row.run, round=row["round"], t_star=row.t_star,
                           t_hat=f10.predict(row, runs[row.run])[0]) for _, row in R[R.ds == "do"].iterrows()])
    say("    t* prediction on data order:", scores(P).round(3).to_dict())

    # 5. budget slope per round
    sl = []
    for (ds_, run, rnd), gg in Tr.groupby(["ds", "run", "round"]):
        if gg["clock"].nunique() >= 3:
            sl.append(dict(ds=ds_, run=run, round=rnd, kappa=-np.polyfit(gg["clock"], gg["y"], 1)[0]))
    sl = pd.DataFrame(sl)
    sl.to_csv(os.path.join(OUT, "kappa_per_round.csv"), index=False)
    say("\n[5] per-round budget slope kappa:\n" +
        sl.groupby(["round", "ds"])["kappa"].mean().unstack().round(3).to_string())
    for ds_ in ["10s", "do"]:
        gg = Tr[Tr.ds == ds_].copy()
        fe = pd.get_dummies(gg["run"] + "_" + gg["round"].astype(str)).to_numpy(float)
        c = gg["clock"].to_numpy()
        for nm, cols in [("linear", [c]), ("quadratic", [c, c ** 2])]:
            X = np.column_stack([fe] + cols)
            w, *_ = np.linalg.lstsq(X, gg["y"].to_numpy(), rcond=None)
            res = gg["y"].to_numpy() - X @ w
            say(f"    {ds_} within-round {nm:9s} clock coefs {np.round(w[fe.shape[1]:], 4)}, "
                f"resid SD {np.sqrt((res ** 2).sum() / (len(res) - X.shape[1])):.3f}")

    # 6. revised formula: fit on both datasets, CIs, leftover dataset effect
    Tf = Tr[np.isfinite(revised().design(Tr)).all(axis=1)].reset_index(drop=True)
    fr = revised().fit(Tf)
    W = np.array([revised().fit(b).w for b in boot_runs(Tf, rng, args.n_boot)])
    lo, hi = np.percentile(W, [2.5, 97.5], axis=0)
    par = pd.DataFrame(dict(param=fr.param_names(), value=fr.w, ci_lo=lo, ci_hi=hi))
    for ds_ in ["10s", "do"]:
        par[f"fit_{ds_}_only"] = revised().fit(Tf[Tf.ds == ds_]).w
    f0p = original().fit(Tf)
    par0 = pd.DataFrame(dict(param=f0p.param_names(), value=f0p.w))
    par.to_csv(os.path.join(OUT, "revised_params.csv"), index=False)
    say("\n[6] revised formula, fit on both datasets (95% run-bootstrap CI):\n" + par.round(4).to_string(index=False))
    say("    original formula refit on both datasets:", dict(zip(par0.param, par0.value.round(4))))
    isdo = (Tf.ds == "do").to_numpy(float)
    for nm, m in [("original", original()), ("revised", fr)]:
        X = np.column_stack([m.design(Tf), isdo, -isdo * Tf["clock"].to_numpy()])
        y = Tf["y"].to_numpy()
        w = np.linalg.lstsq(X, y, rcond=None)[0][-2:]
        wb = []
        for _ in range(500):
            idx = np.concatenate([np.where(Tf["run"].to_numpy() == r)[0]
                                  for r in rng.choice(Tf["run"].unique(), Tf["run"].nunique())])
            wb.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][-1])
        say(f"    leftover data-order effect on kappa, {nm}: {w[1]:+.3f} pp/e-fold "
            f"[{np.percentile(wb, 2.5):+.3f}, {np.percentile(wb, 97.5):+.3f}] (offset {w[0]:+.3f})")

    # 7. held-out comparison: leave one run out over all 20 runs
    methods = [original(), original(0.1), revised(), revised(0.1)]
    preds = []
    for r in Tf["run"].unique():
        trn_t, trn_r = Tf[Tf.run != r], R[R.run != r]
        ds_ = runs[r]["ds"]
        fitted = [(m.name, m.fit(trn_t)) for m in methods]
        fitted += [(f"{b.name} (both datasets)", b.fit(trn_r)) for b in [SparsityLine(), RoundMedian()]]
        fitted += [(f"{b.name} (same alpha only)", b.fit(trn_r[trn_r.ds == ds_])) for b in [SparsityLine(), RoundMedian()]]
        for _, row in R[(R.run == r) & R.t_star.notna()].iterrows():
            for nm, m in fitted:
                th = m.predict(row, runs[r])
                preds.append(dict(method=nm, ds=ds_, run=r, round=row["round"], t_star=row.t_star,
                                  t_hat=th[0] if isinstance(th, tuple) else th))
    P = pd.DataFrame(preds)
    P.to_csv(os.path.join(OUT, "loro_predictions.csv"), index=False)
    S = P.groupby(["method", "ds"]).apply(scores, include_groups=False).unstack("ds")
    S.to_csv(os.path.join(OUT, "loro_scores.csv"))
    say("\n[7] leave-one-run-out over 20 runs:\n" + S.round(3).to_string())
    P10 = []
    T10 = Tf[Tf.ds == "10s"]
    for r in T10["run"].unique():
        for m in [original(), original(0.1), revised(), revised(0.1)]:
            m.fit(T10[T10.run != r])
            for _, row in R[(R.run == r) & R.t_star.notna()].iterrows():
                P10.append(dict(method=m.name, t_star=row.t_star, t_hat=m.predict(row, runs[r])[0]))
    say("    trained on the 10-seed runs only (leave one run out):\n" +
        pd.DataFrame(P10).groupby("method").apply(scores, include_groups=False).round(3).to_string())

    # 8. one cheap probe per round to calibrate kappa
    pr = []
    for r in Tf["run"].unique():
        m = revised().fit(Tf[Tf.run != r])
        for _, row in R[(R.run == r) & (R.t_star < 80) & R.t_star.notna()].iterrows():
            tt = Tf[(Tf.run == r) & (Tf["round"] == row["round"]) & (Tf["t"] >= 66)]
            if not len(tt):
                continue
            pb = tt.sort_values("t").iloc[0]
            cap, k = m.capacity_kappa(row)
            k_probe = max((cap - pb["y"]) / pb["clock"], 1e-3)
            pr.append(dict(ds=runs[r]["ds"], t_star=row.t_star, model=m.predict(row, runs[r])[0],
                           probe=m.predict(row, runs[r], kappa=k_probe)[0], probe_epochs=80 - pb["t"],
                           k_model=k, k_probe=k_probe))
    pr = pd.DataFrame(pr)
    pr.to_csv(os.path.join(OUT, "probe_calibration.csv"), index=False)
    out = []
    for ds_, gg in pr.groupby("ds"):
        for c in ["model", "probe"]:
            out.append(dict(ds=ds_, kappa_from=c, probe_epochs=gg.probe_epochs.mean(),
                            **scores(gg.rename(columns={c: "t_hat"})[["t_star", "t_hat"]]).to_dict(),
                            kappa_median=gg[f"k_{c}"].median()))
    say("\n[8] kappa from one late rewind (t >= 66) vs from the model, rounds that had such a trial:\n" +
        pd.DataFrame(out).round(3).to_string(index=False))

    figures(R, Tr, sl, chain, per_run, P, runs)
    open(os.path.join(OUT, "numbers.txt"), "w", encoding="utf-8").write("\n".join(LINES) + "\n")


# ------------------------------------------------------------------ figures

def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3dd", lw=0.8)
    ax.set_axisbelow(True)


def figures(R, Tr, sl, chain, per_run, P, runs):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": "#3d3c37",
                         "xtick.color": "#5f5e57", "ytick.color": "#5f5e57", "axes.edgecolor": "#b9b8b0"})
    jit = np.random.default_rng(1)

    # fig1: t* by round, measured vs predicted (leave one run out)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, ds_, col in [(axes[0], "10s", C10), (axes[1], "do", CDO)]:
        g = R[(R.ds == ds_) & R.t_star.notna()]
        ax.scatter(g["round"] + jit.uniform(-0.15, 0.15, len(g)), g["t_star"], s=14, color=col, alpha=0.6,
                   lw=0, label="measured t* (one run)")
        ax.plot(*zip(*g.groupby("round")["t_star"].mean().items()), color=col, lw=2, label="measured, mean")
        for nm, ls in [("Original formula", "--"), ("Revised formula", "-")]:
            q = P[(P.method == nm) & (P.ds == ds_)].groupby("round")["t_hat"].mean()
            ax.plot(q.index, q.values, color=CMOD if ls == "--" else "#1f1e1c", lw=1.5, ls=ls,
                    label=f"{nm.lower()}, mean prediction")
        ax.set_title(LABEL[ds_])
        ax.set_xlabel("IMP round")
        ax.set_xticks(range(1, 11))
        _style(ax)
    axes[0].set_ylabel("rewind epoch t*")
    axes[1].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("Measured t* and leave-one-run-out predictions (both formulas fit on all other runs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_tstar_by_round.png"), dpi=160)
    plt.close(fig)

    # fig2: budget slope kappa per round
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for ds_, col, dx in [("10s", C10, -0.12), ("do", CDO, 0.12)]:
        g = sl[sl.ds == ds_]
        ax.scatter(g["round"] + dx + jit.uniform(-0.05, 0.05, len(g)), g["kappa"], s=12, color=col, alpha=0.55, lw=0)
        m = g.groupby("round")["kappa"].mean()
        ax.plot(m.index + dx, m.values, color=col, lw=2, marker="o", ms=4, label=LABEL[ds_])
    ax.axhline(0.1025, color=CMOD, ls="--", lw=1.2)
    ax.text(10.4, 0.1025, "κ in the original\nformula (0.10)", va="center", fontsize=8, color="#5f5e57")
    ax.set_xlim(1.5, 12.3)
    ax.set_xticks(range(2, 11))
    ax.set_xlabel("IMP round")
    ax.set_ylabel("κ: pp lost per e-fold of LR budget")
    ax.set_title("Budget slope fitted separately in each round")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_kappa_per_round.png"), dpi=160)
    plt.close(fig)

    # fig3: the chain
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ds_, col in [("10s", C10), ("do", CDO)]:
        g = chain[chain.ds == ds_]
        axes[0].plot(g["round"], g["parent_def"], color=col, lw=2, marker="o", ms=4, label=LABEL[ds_])
        axes[1].plot(g["round"], g["D_acc"], color=col, lw=2, marker="o", ms=4, label=LABEL[ds_])
    axes[0].set_title("Chain model before pruning, vs dense")
    axes[0].set_ylabel("val acc − dense val acc (pp)")
    axes[1].set_title("Val acc lost to pruning (BN recalibrated)")
    axes[1].set_ylabel("D_acc_val (pp)")
    for ax in axes:
        ax.set_xlabel("IMP round")
        ax.set_xticks(range(1, 11))
        _style(ax)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_chain.png"), dpi=160)
    plt.close(fig)

    # fig4: dense accuracy does not carry into retraining
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    g = per_run[per_run.ds == "do"]
    ax.scatter(g["dense_dev"], g["dev"], s=36, color=CDO, edgecolor="white", lw=1.5, zorder=3)
    for _, row in g.iterrows():
        ax.annotate(row["run"].replace("order", ""), (row["dense_dev"], row["dev"]), xytext=(4, 3),
                    textcoords="offset points", fontsize=7, color="#5f5e57")
    lim = 0.7
    ax.plot([-lim, lim], [-lim, lim], color=CMOD, ls=":", lw=1)
    ax.text(0.42, 0.52, "full carry-over", fontsize=7, color="#5f5e57", rotation=33)
    ax.axhline(0, color="#b9b8b0", lw=0.8)
    ax.axvline(0, color="#b9b8b0", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("dense val acc − mean over orders (pp)")
    ax.set_ylabel("t = 2 retrains − round mean (pp)")
    ax.set_title("Data orders: dense vs retrain deviation", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_dense_carryover.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
