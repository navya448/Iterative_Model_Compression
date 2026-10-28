"""Step 3: fit the t* formula and test it on unseen seeds.

Protocol (no test-seed information is used for fitting):
  * Seeds: the 10 runs included by the audit = 2 discovery runs (seed1_orig, seed2_orig,
    used to design the formula) + 8 new runs (seed3..seed10).
  * Main evaluation ("strict"): every split with 5 TEST seeds drawn from the 8 new runs
    (C(8,5) = 56 splits). The 5 TRAINING seeds are the other 3 new runs + the 2 discovery
    runs, so the test seeds never influenced the formula's form, margin or parameters.
  * Secondary ("all"): all C(10,5) = 252 splits of the 10 runs.
  * Fitting: the formula is least squares on every rewind trial of the training seeds;
    data-driven baselines are fitted on the training seeds' rounds only.

Metrics (test rounds only):
  rounds with a measured t* (status no_retrain / searched):
    MAE, mean signed error (t_hat - t*), too late (t_hat > t*: retraining at t_hat would not
    recover, assuming later rewinds recover less as the binary search does), too late by
    more than the 2-epoch search resolution, within resolution (|err| <= 2)
  rounds where even t_min failed (unrecoverable): correctly flagged "stop"
  rounds with a t*: false "stop" flags
  cost: retraining epochs at t_hat; with fallback = plus one retrain at t_min whenever
    t_hat is too late; compared with "t_min + check" (rewind every round that needs it to
    t_min), the oracle (T - t*), and the binary search that was actually run
  accuracy fit (formula only): RMSE of predicted vs measured final val acc over trials

Writes results/*.csv, results/*.md and figures/*.png|pdf.
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from common_eval import FIG, OUT, build_tables, load_manifest
from models import Formula, main_methods

STRICT, ALL = "strict", "all"
RES = 2  # binary-search resolution


def run_split(methods, rounds, trials, runs, train, test):
    tr_r = rounds[rounds["run"].isin(train)]
    tr_t = trials[trials["run"].isin(train)]
    te_r = rounds[rounds["run"].isin(test)]
    te_t = trials[trials["run"].isin(test)]
    preds, fits = [], {}
    rows = te_r.to_dict("records")
    for m in methods:
        m.fit(tr_r, tr_t)
        for row in rows:
            t_hat, flag = m.predict(row, runs[row["run"]])
            preds.append(dict(method=m.name, run=row["run"], round=int(row["round"]),
                              t_hat=t_hat, flag=flag))
        if isinstance(m, Formula):
            e = m.yhat(te_t) - m._y(te_t)
            fits[m.name] = dict(coef=dict(m.coef), train_rmse=m.train_rmse,
                                test_rmse=float(np.sqrt(np.nanmean(e ** 2))))
    return pd.DataFrame(preds), fits


_SEARCH = {}


def search_cost(trials, run, rnd):
    if not _SEARCH:
        _SEARCH.update(trials.groupby(["run", "round"])["retrain_epochs"].sum().to_dict())
    return float(_SEARCH.get((run, rnd), 0.0))


def score(pred, rounds, trials, runs):
    """Per-method metrics for one set of predictions."""
    d = pred.merge(rounds[["run", "round", "t_star", "round_status", "needs_retrain"]], on=["run", "round"])
    out = []
    for name, g in d.groupby("method", sort=False):
        meas = g[g["round_status"].isin(["no_retrain", "searched"])].dropna(subset=["t_hat"])
        unrec = g[g["round_status"] == "unrecoverable"]
        err = meas["t_hat"] - meas["t_star"]
        T = np.array([runs[r]["T"] for r in meas["run"]])
        tmin = np.array([runs[r]["t_min"] for r in meas["run"]])
        too_late = (err > 0).to_numpy()
        cost = np.where(meas["t_hat"] >= T, 0, T - meas["t_hat"])
        cost_fb = cost + too_late * (T - tmin)
        oracle = np.where(meas["t_star"] >= T, 0, T - meas["t_star"])
        tmin_check = np.where(meas["needs_retrain"], T - tmin, 0)
        searched = np.array([search_cost(trials, r, k) for r, k in zip(meas["run"], meas["round"])])
        # unrecoverable rounds: every rewind fails; cost = attempt at t_hat (+ t_min if t_hat > t_min)
        Tu = np.array([runs[r]["T"] for r in unrec["run"]])
        tu = np.array([runs[r]["t_min"] for r in unrec["run"]])
        cost_u = (Tu - unrec["t_hat"].to_numpy()) + (unrec["t_hat"].to_numpy() > tu) * (Tu - tu)
        out.append(dict(
            method=name, n_rounds=len(meas), n_retrain_rounds=int(meas["needs_retrain"].sum()),
            mae=err.abs().mean(), signed=err.mean(), median_abs=err.abs().median(),
            too_late=int(too_late.sum()), too_late_rate=too_late.mean(),
            too_late_gt_res=int((err > RES).sum()), within_res=float((err.abs() <= RES).mean()),
            mae_retrain_rounds=err[meas["needs_retrain"].to_numpy()].abs().mean(),
            n_unrecoverable=len(unrec), unrec_flagged=int((unrec["flag"] == "stop").sum()),
            false_stop=int((meas["flag"] == "stop").sum()),
            retrain_epochs=float(cost.sum()), retrain_epochs_fallback=float(cost_fb.sum() + cost_u.sum()),
            oracle_epochs=float(oracle.sum()), tmin_check_epochs=float(tmin_check.sum() + (Tu - tu).sum()),
            search_epochs=float(searched.sum()),
        ))
    s = pd.DataFrame(out)
    s["saving_vs_tmin_check"] = 1 - s["retrain_epochs_fallback"] / s["tmin_check_epochs"]
    return s


def splits(labels_new, labels_disc, kind):
    if kind == STRICT:
        for test in itertools.combinations(labels_new, 5):
            yield sorted(set(labels_new) - set(test)) + list(labels_disc), list(test)
    else:
        allr = list(labels_disc) + list(labels_new)
        for test in itertools.combinations(allr, 5):
            yield sorted(set(allr) - set(test)), list(test)


def summarize(df, cols):
    g = df.groupby("method", sort=False)
    rows = []
    for name, d in g:
        r = dict(method=name, n_splits=len(d))
        for c in cols:
            v = d[c].dropna()
            r[f"{c}_mean"] = v.mean()
            r[f"{c}_p05"] = v.quantile(0.05)
            r[f"{c}_p95"] = v.quantile(0.95)
        rows.append(r)
    return pd.DataFrame(rows)


def evaluate(kind, methods, rounds, trials, runs, new, disc):
    all_scores, all_preds, all_fits = [], [], []
    for i, (train, test) in enumerate(splits(new, disc, kind)):
        pred, fits = run_split(methods, rounds, trials, runs, train, test)
        sc = score(pred, rounds, trials, runs)
        sc["split"] = i
        sc["test"] = ",".join(test)
        pred["split"] = i
        all_scores.append(sc)
        all_preds.append(pred)
        for name, f in fits.items():
            all_fits.append(dict(split=i, method=name, test=",".join(test), **f["coef"],
                                 train_rmse=f["train_rmse"], test_rmse=f["test_rmse"]))
    return (pd.concat(all_scores, ignore_index=True), pd.concat(all_preds, ignore_index=True),
            pd.DataFrame(all_fits))


def per_seed(preds, rounds, runs, trials=None):
    """Each new seed's metrics averaged over the splits in which it was a test seed."""
    d = preds.merge(rounds[["run", "round", "t_star", "round_status", "needs_retrain"]], on=["run", "round"])
    d["T"] = [runs[r]["T"] for r in d["run"]]
    d["tmin"] = [runs[r]["t_min"] for r in d["run"]]
    meas = d["round_status"].isin(["no_retrain", "searched"])
    unrec = d["round_status"] == "unrecoverable"
    d["abs_err"] = np.where(meas, (d["t_hat"] - d["t_star"]).abs(), np.nan)
    d["late"] = np.where(meas, (d["t_hat"] > d["t_star"]).astype(float), np.nan)
    base = np.where(d["t_hat"] >= d["T"], 0, d["T"] - d["t_hat"])
    d["cost"] = np.where(meas, base + (d["t_hat"] > d["t_star"]) * (d["T"] - d["tmin"]),
                         np.where(unrec, base + (d["t_hat"] > d["tmin"]) * (d["T"] - d["tmin"]), np.nan))
    d["cost_tmin_check"] = np.where(meas & d["needs_retrain"], d["T"] - d["tmin"],
                                    np.where(unrec, d["T"] - d["tmin"], 0))
    per = d.groupby(["method", "split", "run"]).agg(
        mae=("abs_err", "mean"), late=("late", "mean"), cost=("cost", "sum"),
        cost_ref=("cost_tmin_check", "sum")).reset_index()
    out = per.groupby(["method", "run"]).agg(mae=("mae", "mean"), too_late_rate=("late", "mean"),
                                             cost=("cost", "mean"), cost_tmin_check=("cost_ref", "mean")).reset_index()
    out["saving_vs_tmin_check"] = 1 - out["cost"] / out["cost_tmin_check"]
    return out


def paired_tests(seed_tab, ref, others, metrics=("mae", "too_late_rate", "saving_vs_tmin_check")):
    """Paired comparison across test seeds (each seed's value averaged over the splits it was
    tested in). Wilcoxon signed-rank (exact for n <= 25) and a seed bootstrap CI of the mean
    difference ref - baseline. Seeds are few (8), so treat p-values as rough."""
    rows = []
    a = seed_tab[seed_tab["method"] == ref].set_index("run")
    for m in others:
        b = seed_tab[seed_tab["method"] == m].set_index("run")
        common = a.index.intersection(b.index)
        for met in metrics:
            diff = (a.loc[common, met] - b.loc[common, met]).dropna().to_numpy()
            nz = diff[np.abs(diff) > 1e-12]
            p = stats.wilcoxon(nz).pvalue if len(nz) >= 1 else np.nan
            rng = np.random.default_rng(0)
            boots = rng.choice(diff, (5000, len(diff)), replace=True).mean(axis=1)
            rows.append(dict(reference=ref, baseline=m, metric=met, n_seeds=len(diff),
                             mean_diff_ref_minus_baseline=diff.mean(), ci95_lo=np.percentile(boots, 2.5),
                             ci95_hi=np.percentile(boots, 97.5),
                             seeds_ref_lower=int((diff < -1e-12).sum()), seeds_ref_higher=int((diff > 1e-12).sum()),
                             wilcoxon_p=p))
    return pd.DataFrame(rows)


def md_table(df, cols, fmt):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=[STRICT, ALL], choices=[STRICT, ALL],
                    help="which split families to evaluate (default both)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    man = load_manifest()
    rounds, trials, runs = build_tables(man)
    inc = [e for e in man["runs"] if e["include"]]
    disc = [e["label"] for e in inc if e["discovery"]]
    new = [e["label"] for e in inc if not e["discovery"]]
    print(f"discovery runs {disc}; new runs {new}")
    print(f"rounds by status:\n{rounds.groupby(['group', 'round_status']).size()}")
    print(f"trials: {len(trials)} ({trials.groupby('run').size().to_dict()})")

    # ---- data summary table
    ds = []
    for lab in disc + new:
        r = rounds[rounds["run"] == lab]
        ds.append(dict(run=lab, group=runs[lab]["group"], dense_val_acc=runs[lab]["a_dense"],
                       target=runs[lab]["target"], rounds=len(r),
                       max_sparsity_recovered=r.loc[r["t_star"].notna(), "sparsity"].max(),
                       t_star=" ".join("-" if pd.isna(v) else str(int(v)) for v in r["t_star"]),
                       n_trials=int((trials["run"] == lab).sum())))
    pd.DataFrame(ds).to_csv(os.path.join(OUT, "data_summary.csv"), index=False)

    # ---- B^-1 solution == brute-force search over t (verification), several variants
    from models import variant_methods
    mism, n = 0, 0
    for f in [Formula("check0", margin=0.0), Formula("check1", margin=0.1)] + variant_methods():
        f.fit(rounds, trials)
        for row in rounds.to_dict("records"):
            a = f.predict(row, runs[row["run"]])
            b = f.predict_bruteforce(row, runs[row["run"]])
            if f.kappa_eff(row["weight_perturbation"]) > 0:
                mism += a != b
                n += 1
    print(f"B^-1 vs brute-force mismatches: {mism} of {n}")
    assert mism == 0

    # ---- final parameters on all 10 runs, with seed-cluster bootstrap CIs
    labs = disc + new
    rng = np.random.default_rng(1)
    boot = []
    for _ in range(2000):
        pick = rng.choice(labs, len(labs), replace=True)
        tt = pd.concat([trials[trials["run"] == l] for l in pick], ignore_index=True)
        boot.append(Formula("b").fit(None, tt).coef)
    boot = pd.DataFrame(boot)
    fits = {"all 10 runs": Formula("all").fit(rounds, trials),
            "discovery runs only": Formula("disc").fit(rounds[rounds["run"].isin(disc)], trials[trials["run"].isin(disc)]),
            "8 new runs only": Formula("new").fit(rounds[rounds["run"].isin(new)], trials[trials["run"].isin(new)])}
    prow = []
    for k, m in fits.items():
        n = int(trials["run"].isin({"all 10 runs": labs, "discovery runs only": disc, "8 new runs only": new}[k]).sum())
        prow.append(dict(fit_on=k, n_trials=n, **m.coef, trial_rmse=m.train_rmse,
                         capacity_limit_sparsity=1 - np.exp(-(0.5 + m.coef["a0"]) / m.coef["a1"])))
    params = pd.DataFrame(prow)
    for c in ["a0", "a1", "kappa"]:
        params.loc[params["fit_on"] == "all 10 runs", f"{c}_ci95"] = f"[{boot[c].quantile(.025):.3f}, {boot[c].quantile(.975):.3f}]"
    params.to_csv(os.path.join(OUT, "formula_params.csv"), index=False)
    print(params.round(4).to_string(index=False))

    # ---- held-out evaluation
    metric_cols = ["mae", "signed", "too_late", "too_late_rate", "too_late_gt_res", "within_res",
                   "mae_retrain_rounds", "unrec_flagged", "false_stop", "retrain_epochs_fallback",
                   "saving_vs_tmin_check", "tmin_check_epochs", "oracle_epochs", "search_epochs", "n_rounds",
                   "n_unrecoverable"]
    results = {}
    for kind in args.splits:
        methods = main_methods()
        sc, pr, ft = evaluate(kind, methods, rounds, trials, runs, new, disc)
        sc.to_csv(os.path.join(OUT, f"split_scores_{kind}.csv"), index=False)
        pr.to_csv(os.path.join(OUT, f"split_predictions_{kind}.csv"), index=False)
        ft.to_csv(os.path.join(OUT, f"split_formula_fits_{kind}.csv"), index=False)
        summ = summarize(sc, metric_cols)
        summ.to_csv(os.path.join(OUT, f"heldout_summary_{kind}.csv"), index=False)
        results[kind] = (sc, pr, ft, summ)
        print(f"\n=== held-out ({kind}, {sc['split'].nunique()} splits) ===")
        print(summ[["method", "mae_mean", "mae_p05", "mae_p95", "signed_mean", "too_late_rate_mean",
                    "too_late_gt_res_mean", "unrec_flagged_mean", "false_stop_mean",
                    "saving_vs_tmin_check_mean"]].round(3).to_string(index=False))
        rm = ft.groupby("method")[["train_rmse", "test_rmse"]].agg(["mean", lambda x: x.quantile(.05), lambda x: x.quantile(.95)])
        print(rm.round(3))

    if STRICT not in results:
        return
    # ---- per-seed and paired tests (strict)
    sc, pr, ft, summ = results[STRICT]
    seed_tab = per_seed(pr, rounds, runs)
    seed_tab.to_csv(os.path.join(OUT, "per_seed_strict.csv"), index=False)
    ref = "Formula (margin 0.1 pp)"
    others = ["Formula (margin 0)", "Sparsity line", "Sparsity line + check", "Round median + check",
              "t_min + check", "Loss match (train)", "Loss match (train) + check",
              "Fixed: before last LR drop + check"]
    tests = pd.concat([paired_tests(seed_tab, ref, others),
                       paired_tests(seed_tab, "Formula (margin 0)", ["Sparsity line + check", "Round median + check",
                                                                     "Sparsity line"])], ignore_index=True)
    seed_tab.to_csv(os.path.join(OUT, "per_seed_strict.csv"), index=False)
    tests.to_csv(os.path.join(OUT, "paired_tests_strict.csv"), index=False)
    print("\n=== paired per-seed MAE differences (strict) ===")
    print(tests.round(3).to_string(index=False))

    # split-level win rates
    w = sc.pivot_table(index="split", columns="method", values="mae")
    wins = pd.DataFrame([dict(reference=ref, baseline=b,
                              frac_splits_ref_lower_mae=float((w[ref] < w[b] - 1e-12).mean()),
                              frac_ties=float(((w[ref] - w[b]).abs() <= 1e-12).mean()))
                         for b in others])
    wins.to_csv(os.path.join(OUT, "split_win_rates_strict.csv"), index=False)
    print(wins.round(3).to_string(index=False))

    # ---- pooled predictions for figures: each new seed's prediction from every split it was in
    json.dump(dict(discovery=disc, new=new), open(os.path.join(OUT, "split_definition.json"), "w"), indent=2)
    print(f"\nWrote tables to {OUT}")


if __name__ == "__main__":
    main()
