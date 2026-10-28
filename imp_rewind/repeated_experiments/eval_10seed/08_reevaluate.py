"""Step 8: re-run the predictor comparison against the steadier t* from step 7.

Same protocol as step 3: 56 splits, 5 test runs from the 8 new runs, training on the
other 3 new runs + the 2 discovery runs. Three scoreboards:

  protocol   t* from the binary search, own target           (= step 3, for reference)
  own        smoothed t*, own target                          (effect of smoothing only)
  pop        smoothed t*, common (population) target          (both changes)

Under "pop" the formula models A(t) - A_pop (A_pop = mean dense val acc), and every
"+ check" uses the common target. Rounds whose t* is censored are left out of the
t* metrics for every method alike.

New predictor: "Formula, 90% safe". It picks the latest t whose predicted chance of
recovering is >= 90%, assuming the error of a prediction for a new run is normal with
SD sigma. sigma is estimated on the TRAINING runs only, by leave-one-run-out within them
(fit on 4, predict the 5th run's trials), so it includes run-to-run offsets. Then
margin = z_0.9 * sigma. Nothing about the test runs is used.

Calibration check (pop target): for every held-out rewind trial, the safe formula's
probability of recovery is compared with what actually happened (Brier score, log loss,
reliability table), against two logistic-regression baselines fitted on the training
runs' trials: recover ~ sparsity + t, and recover ~ sparsity + ln(B(t_min)/B(t)).

Writes results/reeval_*.csv and figures/fig9_reeval.(png|pdf), fig10_calibration.(png|pdf).
"""

import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

from common_eval import FIG, OUT, build_tables, load_manifest
from models import Formula, RoundMedian, RuleColumn, SparsityLine, TMin

_spec = importlib.util.spec_from_file_location("fe", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_formula_eval.py"))
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)

LEVEL = 0.9


class SafeFormula(Formula):
    """Formula whose margin is set so the predicted recovery probability at t_hat is LEVEL."""

    def __init__(self, name, level=LEVEL):
        super().__init__(name)
        self.level = level

    def fit(self, rounds, trials):
        res = []
        for r in trials["run"].unique():
            inner = Formula("inner").fit(None, trials[trials["run"] != r])
            tt = trials[trials["run"] == r]
            res.append((inner.yhat(tt) - inner._y(tt)))
        self.sigma = float(np.sqrt(np.mean(np.concatenate(res) ** 2)))
        super().fit(rounds, trials)
        self.margin = stats.norm.ppf(self.level) * self.sigma
        return self

    def p_recover(self, df, tol=0.5):
        """P(final val acc >= target) for each trial row; y is relative to the reference
        dense accuracy and the target is reference - tol."""
        return stats.norm.cdf((self.yhat(df) + tol) / self.sigma)


def methods():
    return [Formula("Formula (margin 0)"), Formula("Formula (margin 0.1 pp)", margin=0.1),
            SafeFormula("Formula, 90% safe"),
            SparsityLine("Sparsity line + check", check=True), RoundMedian("Round median + check"),
            SparsityLine("Sparsity line"),
            RuleColumn("Loss match (train) + check", "pred_loss_match_train", check=True),
            RuleColumn("Accuracy match (val) + check", "pred_acc_match_val", check=True),
            RuleColumn("Fixed: before 1st LR drop + check", "pred_baseline_before_first_lr_drop", check=True),
            RuleColumn("Fixed: before last LR drop + check", "pred_baseline_before_last_lr_drop", check=True),
            TMin("t_min + check", check=True)]


def make_tables(rounds, trials, red, a_pop, which):
    """rounds/trials re-labelled for one t* definition."""
    r = rounds.merge(red[["run", "round", f"t_star_{which}", f"status_{which}", "target_pop"]],
                     on=["run", "round"], how="left")
    r["t_star"] = r[f"t_star_{which}"]
    r["round_status"] = r[f"status_{which}"]
    t = trials.copy()
    if which == "pop":
        r["target_val_acc"] = r["target_pop"]
        t["y"] = t["final_val_acc"] - a_pop
        t["target_val_acc"] = a_pop - 0.5
    r["needs_retrain"] = r["after_val_acc"] < r["target_val_acc"]
    return r, t


class Logit:
    def __init__(self, cols):
        self.cols = cols

    def _X(self, df):
        return np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in self.cols])

    def fit(self, df, y):
        X = self._X(df)
        mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-12
        self.mu, self.sd = mu, sd
        Xs = X.copy()
        Xs[:, 1:] = (X[:, 1:] - mu) / sd

        def nll(w):
            z = Xs @ w
            return np.sum(np.logaddexp(0, z) - y * z) + 1e-3 * np.sum(w[1:] ** 2)
        self.w = optimize.minimize(nll, np.zeros(X.shape[1]), method="BFGS").x
        return self

    def p(self, df):
        X = self._X(df)
        X[:, 1:] = (X[:, 1:] - self.mu) / self.sd
        return 1 / (1 + np.exp(-(X @ self.w)))


def calibration(rounds_p, trials_p, new, disc, a_pop):
    target = a_pop - 0.5
    tp = trials_p.copy()
    tp["recovered_pop"] = (tp["final_val_acc"] >= target - 1e-9).astype(float)
    rows, pairs = [], []
    for i, (train, test) in enumerate(fe.splits(new, disc, fe.STRICT)):
        tr, te = tp[tp["run"].isin(train)], tp[tp["run"].isin(test)]
        sf = SafeFormula("s").fit(None, tr)
        preds = {"Formula, 90% safe (probability)": sf.p_recover(te)}
        for name, cols in [("Logistic: sparsity + t", ["sparsity", "t"]),
                           ("Logistic: sparsity + budget clock", ["sparsity", "clock"])]:
            preds[name] = Logit(cols).fit(tr, tr["recovered_pop"].to_numpy()).p(te)
        preds["Constant rate"] = np.full(len(te), tr["recovered_pop"].mean())
        y = te["recovered_pop"].to_numpy()
        for name, p in preds.items():
            pc = np.clip(p, 1e-4, 1 - 1e-4)
            hi = p >= LEVEL
            rows.append(dict(split=i, model=name, brier=np.mean((p - y) ** 2),
                             log_loss=-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
                             n_p_ge_level=int(hi.sum()),
                             recovered_when_p_ge_level=float(y[hi].mean()) if hi.any() else np.nan,
                             sigma=sf.sigma if name.startswith("Formula") else np.nan))
            pairs.append(pd.DataFrame(dict(split=i, model=name, run=te["run"].to_numpy(), round=te["round"].to_numpy(),
                                           t=te["t"].to_numpy(), p=p, y=y)))
    return pd.DataFrame(rows), pd.concat(pairs, ignore_index=True)


def main():
    man = load_manifest()
    rounds, trials, runs = build_tables(man)
    inc = [e for e in man["runs"] if e["include"]]
    disc = [e["label"] for e in inc if e["discovery"]]
    new = [e["label"] for e in inc if not e["discovery"]]
    red = pd.read_csv(os.path.join(OUT, "tstar_redefined.csv"))
    a_pop = float(pd.read_json(os.path.join(OUT, "population_target.json"), typ="series")["a_pop"])

    cols = ["mae", "signed", "too_late", "too_late_rate", "too_late_gt_res", "within_res", "mae_retrain_rounds",
            "unrec_flagged", "false_stop", "saving_vs_tmin_check", "n_rounds", "n_unrecoverable"]
    summaries, tests, seeds = [], [], []
    for which in ("own", "pop"):
        r, t = make_tables(rounds, trials, red, a_pop, which)
        sc, pr, ft = fe.evaluate(fe.STRICT, methods(), r, t, runs, new, disc)
        sc.to_csv(os.path.join(OUT, f"reeval_split_scores_{which}.csv"), index=False)
        pr.to_csv(os.path.join(OUT, f"reeval_split_predictions_{which}.csv"), index=False)
        s = fe.summarize(sc, cols)
        s.insert(0, "tstar_definition", which)
        summaries.append(s)
        st = fe.per_seed(pr, r, runs)
        st.insert(0, "tstar_definition", which)
        seeds.append(st)
        for ref in ("Formula (margin 0.1 pp)", "Formula, 90% safe"):
            pt = fe.paired_tests(st.drop(columns="tstar_definition"), ref,
                                 ["Sparsity line + check", "Round median + check", "Loss match (train) + check"])
            pt.insert(0, "tstar_definition", which)
            tests.append(pt)
        print(f"\n=== {which}: held-out (56 splits) ===")
        print(s[["method", "n_rounds_mean", "mae_mean", "mae_p05", "mae_p95", "signed_mean", "too_late_rate_mean",
                 "within_res_mean", "unrec_flagged_mean", "false_stop_mean", "saving_vs_tmin_check_mean"]]
              .round(3).to_string(index=False))
    summ = pd.concat(summaries, ignore_index=True)
    summ.to_csv(os.path.join(OUT, "reeval_summary.csv"), index=False)
    pd.concat(seeds, ignore_index=True).to_csv(os.path.join(OUT, "reeval_per_seed.csv"), index=False)
    tests = pd.concat(tests, ignore_index=True)
    tests.to_csv(os.path.join(OUT, "reeval_paired_tests.csv"), index=False)
    print("\n=== paired per-seed tests (reference minus baseline) ===")
    print(tests[["tstar_definition", "reference", "baseline", "metric", "mean_diff_ref_minus_baseline", "ci95_lo",
                 "ci95_hi", "seeds_ref_lower", "seeds_ref_higher", "wilcoxon_p"]].round(3).to_string(index=False))

    # calibration of recovery probabilities on held-out trials (pop target)
    r, t = make_tables(rounds, trials, red, a_pop, "pop")
    cal, pairs = calibration(r, t, new, disc, a_pop)
    cal.to_csv(os.path.join(OUT, "reeval_calibration_splits.csv"), index=False)
    cs = cal.groupby("model").agg(brier=("brier", "mean"), brier_p05=("brier", lambda v: v.quantile(.05)),
                                  brier_p95=("brier", lambda v: v.quantile(.95)), log_loss=("log_loss", "mean"),
                                  n_p_ge_level=("n_p_ge_level", "mean"),
                                  recovered_when_p_ge_level=("recovered_when_p_ge_level", "mean"),
                                  sigma=("sigma", "mean")).reset_index()
    cs.to_csv(os.path.join(OUT, "reeval_calibration_summary.csv"), index=False)
    bins = np.array([0, .1, .3, .5, .7, .9, 1.0001])
    pairs["bin"] = pd.cut(pairs["p"], bins, right=False)
    rel = pairs.groupby(["model", "bin"], observed=True).agg(n=("y", "size"), mean_p=("p", "mean"),
                                                             frac_recovered=("y", "mean")).reset_index()
    rel["bin"] = rel["bin"].astype(str)
    rel.to_csv(os.path.join(OUT, "reeval_reliability.csv"), index=False)
    print("\n=== calibration on held-out trials (pop target; means over 56 splits) ===")
    print(cs.round(4).to_string(index=False))
    print(rel.round(3).to_string(index=False))
    # per-seed Brier, paired
    pb = pairs.assign(se=(pairs["p"] - pairs["y"]) ** 2).groupby(["model", "split", "run"])["se"].mean().reset_index()
    pb = pb.groupby(["model", "run"])["se"].mean().unstack(0)
    ref = "Formula, 90% safe (probability)"
    brows = []
    for m in pb.columns:
        if m == ref:
            continue
        d = (pb[ref] - pb[m]).dropna().to_numpy()
        boots = np.random.default_rng(0).choice(d, (5000, len(d)), replace=True).mean(1)
        brows.append(dict(reference=ref, baseline=m, n_seeds=len(d), mean_brier_diff=d.mean(),
                          ci95_lo=np.percentile(boots, 2.5), ci95_hi=np.percentile(boots, 97.5),
                          seeds_ref_lower=int((d < 0).sum()), wilcoxon_p=stats.wilcoxon(d).pvalue))
    bt = pd.DataFrame(brows)
    bt.to_csv(os.path.join(OUT, "reeval_calibration_paired.csv"), index=False)
    print(bt.round(4).to_string(index=False))

    # equal-safety comparison under the common target (same sweep as step 4), censored rounds left out
    spec = importlib.util.spec_from_file_location("mc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_model_checks.py"))
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    r, t = make_tables(rounds, trials, red, a_pop, "pop")
    r = r[r["round_status"].isin(["no_retrain", "searched", "unrecoverable"])]
    sw = mc.tradeoff_sweep(r, t, runs, new, disc)
    sw.to_csv(os.path.join(OUT, "reeval_tradeoff_sweep_pop.csv"), index=False)
    print("\n=== equal-safety sweep, common target ===")
    print(sw.round(3).to_string(index=False))

    figures(summ, rel)
    print("\nwrote results/reeval_*.csv, figures/fig9_reeval, fig10_calibration")


def figures(summ, rel):
    BLUE, AQUA, YELLOW, GRAY, INK, MUTED, GRID = "#2a78d6", "#1baf7a", "#eda100", "#9b9a94", "#1f1f1e", "#6b6a64", "#e6e5df"
    shown = ["Formula, 90% safe", "Formula (margin 0.1 pp)", "Formula (margin 0)", "Sparsity line + check",
             "Round median + check", "Loss match (train) + check", "Fixed: before last LR drop + check"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True, gridspec_kw=dict(wspace=0.12))
    y = np.arange(len(shown))[::-1]
    for ax, (col, title, mult) in zip(axes, [("mae", "t* MAE (epochs)", 1), ("too_late_rate", "too-late rate (%)", 100),
                                             ("saving_vs_tmin_check", "retraining saved (%)", 100)]):
        for k, (which, mk, off, lab) in enumerate([("own", "s", 0.15, "smoothed t*, own target"),
                                                    ("pop", "o", -0.15, "smoothed t*, common target")]):
            s = summ[summ["tstar_definition"] == which].set_index("method").loc[shown]
            colr = [BLUE if m.startswith("Formula") else (AQUA if m in ("Sparsity line + check", "Round median + check") else GRAY)
                    for m in shown]
            ax.hlines(y + off, s[f"{col}_p05"] * mult, s[f"{col}_p95"] * mult, color=colr, lw=1.6, alpha=0.45)
            ax.scatter(s[f"{col}_mean"] * mult, y + off, color=colr, marker=mk, s=24, zorder=3, edgecolor="white", lw=0.5,
                       label=lab if ax is axes[0] else None)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[2].axvline(0, color=MUTED, lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(shown, fontsize=7.5)
    axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(1.6, -0.12), ncol=2)
    fig.text(0.01, -0.12, "56 splits, 5 training / 5 unseen test runs; dot = mean, bar = 5-95% of splits.",
             fontsize=7.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig9_reeval.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    for m, col, mk in [("Formula, 90% safe (probability)", BLUE, "o"), ("Logistic: sparsity + t", GRAY, "s"),
                       ("Logistic: sparsity + budget clock", YELLOW, "^")]:
        d = rel[rel["model"] == m]
        ax.plot(d["mean_p"], d["frac_recovered"], color=col, marker=mk, ms=4, lw=1.3, label=m)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("predicted chance of recovering")
    ax.set_ylabel("fraction that actually recovered")
    ax.set_title("Held-out rewind trials (common target)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig10_calibration.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
