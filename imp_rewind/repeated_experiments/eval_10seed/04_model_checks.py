"""Step 4: does kappa depend on rho, does the capacity model need improvement, and how
much of this is statistically supported?

  (a) per-round slopes: within every searched round (>= 3 trials) regress final val acc,
      final train acc and ln(final train loss) on the budget clock ln(B(t_min)/B(t));
      correlate the slopes with rho and with sparsity, and the partial correlation with
      rho given sparsity. Seed-cluster bootstrap CIs.
  (b) per-round capacity: with kappa fixed at its pooled fit, the capacity of a round is
      the mean of y + kappa*clock over its trials; compare linear / quadratic in ln(1/d).
  (c) model variants (models.variant_methods) under the same strict 5-train / 5-test seed
      splits as step 3: trial RMSE on the test seeds, t* MAE / too-late / stop flags
      (margin 0 and 0.1 pp), and seed-cluster bootstrap CIs of the extra coefficients.
  (d) per-seed fits of (a0, a1, kappa) and the seed-to-seed spread of t* at each round.
  (e) safety/cost trade-off: sweep the formula's margin and an "earlier by delta epochs"
      shift of the sparsity line and round median (all + check), same strict splits, and
      report too-late rate, MAE and retraining saved for every setting. This compares the
      methods at equal safety instead of at their default settings. (The sweep is
      descriptive: no setting is chosen on the test seeds.)

Usage: python 04_model_checks.py [--parts slopes capacity variants perseed sweep]
Writes results/model_checks_*.csv.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

from common_eval import OUT, build_tables, load_manifest
from models import Formula, RoundMedian, SparsityLine, variant_methods

import importlib.util
_spec = importlib.util.spec_from_file_location("fe", os.path.join(os.path.dirname(__file__), "03_formula_eval.py"))
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)


def slope(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w[1]


def partial_corr(x, y, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return np.corrcoef(rx, ry)[0, 1]


def per_round_slopes(trials, rounds):
    rows = []
    for (run, k), g in trials[trials["round_status"] == "searched"].groupby(["run", "round"]):
        if len(g) < 3:
            continue
        r = rounds[(rounds["run"] == run) & (rounds["round"] == k)].iloc[0]
        c = g["clock"].to_numpy()
        rows.append(dict(run=run, round=k, sparsity=r["sparsity"], rho=r["weight_perturbation"],
                         D_acc_val=r["D_acc_val"], n_trials=len(g),
                         kappa_val=-slope(c, g["final_val_acc"].to_numpy()),
                         kappa_train_acc=-slope(c, g["final_train_acc"].to_numpy()),
                         kappa_ln_train_loss=slope(c, np.log(g["final_train_loss"].to_numpy()))))
    return pd.DataFrame(rows)


def slope_correlations(sl, n_boot=5000):
    rng = np.random.default_rng(2)
    runs = sl["run"].unique()
    out = []
    for target in ["kappa_val", "kappa_train_acc", "kappa_ln_train_loss"]:
        def stats_of(d):
            return (np.corrcoef(d["rho"], d[target])[0, 1], np.corrcoef(d["sparsity"], d[target])[0, 1],
                    partial_corr(d["rho"].to_numpy(), d[target].to_numpy(), d["sparsity"].to_numpy()))
        est = stats_of(sl)
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(runs, len(runs), replace=True)
            d = pd.concat([sl[sl["run"] == r] for r in pick])
            if d["rho"].std() > 0 and d[target].std() > 0:
                boots.append(stats_of(d))
        b = np.array(boots)
        p_rho = stats.pearsonr(sl["rho"], sl[target]).pvalue
        out.append(dict(slope=target, n_rounds=len(sl), n_runs=len(runs),
                        r_rho=est[0], r_rho_ci95=f"[{np.nanpercentile(b[:, 0], 2.5):.2f}, {np.nanpercentile(b[:, 0], 97.5):.2f}]",
                        p_rho_naive=p_rho,
                        r_sparsity=est[1], r_sparsity_ci95=f"[{np.nanpercentile(b[:, 1], 2.5):.2f}, {np.nanpercentile(b[:, 1], 97.5):.2f}]",
                        partial_r_rho_given_sparsity=est[2],
                        partial_ci95=f"[{np.nanpercentile(b[:, 2], 2.5):.2f}, {np.nanpercentile(b[:, 2], 97.5):.2f}]",
                        mean_slope=sl[target].mean(), sd_slope=sl[target].std()))
    return pd.DataFrame(out)


def capacity_per_round(trials, rounds, kappa):
    t = trials[trials["round_status"].isin(["searched", "unrecoverable", "incomplete"])].copy()
    t["cap"] = t["y"] + kappa * t["clock"]
    g = t.groupby(["run", "round"]).agg(cap=("cap", "mean"), n=("cap", "size"),
                                        ln_inv_d=("ln_inv_d", "first"), sparsity=("sparsity", "first")).reset_index()
    return g


def coef_bootstrap(trials, labels, model_factory, n_boot=2000):
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(n_boot):
        pick = rng.choice(labels, len(labels), replace=True)
        tt = pd.concat([trials[trials["run"] == l] for l in pick], ignore_index=True)
        rows.append(model_factory().fit(None, tt).coef)
    return pd.DataFrame(rows)


def tradeoff_sweep(rounds, trials, runs, new, disc):
    margins = np.round(np.arange(0, 0.41, 0.05), 2)
    shifts = np.arange(0, 21, 2)
    rows = []
    for i, (train, test) in enumerate(fe.splits(new, disc, fe.STRICT)):
        tr_r, tr_t = rounds[rounds["run"].isin(train)], trials[trials["run"].isin(train)]
        te = rounds[rounds["run"].isin(test)].to_dict("records")
        f = Formula("f").fit(tr_r, tr_t)
        sl = SparsityLine("sl", check=True).fit(tr_r, tr_t)
        rm = RoundMedian("rm").fit(tr_r, tr_t)
        settings = [("Formula", m, lambda row, m=m: (setattr(f, "margin", m), f.predict(row, runs[row["run"]]))[1])
                    for m in margins]
        for name, mdl in [("Sparsity line + check", sl), ("Round median + check", rm)]:
            for d in shifts:
                def pr(row, mdl=mdl, d=d):
                    t, flag = mdl.predict(row, runs[row["run"]])
                    return (t, flag) if flag == "no_retrain" else (max(runs[row["run"]]["t_min"], t - d), flag)
                settings.append((name, d, pr))
        for name, setting, fn in settings:
            cost = ref = late = n = ae = cost_m = ref_m = 0.0
            for row in te:
                run = runs[row["run"]]
                T, tmin = run["T"], run["t_min"]
                t, flag = fn(row)
                base = 0 if t >= T else T - t
                if row["round_status"] == "unrecoverable":
                    cost += base + (t > tmin) * (T - tmin)
                    ref += T - tmin
                    continue
                n += 1
                is_late = t > row["t_star"]
                late += is_late
                ae += abs(t - row["t_star"])
                cost += base + is_late * (T - tmin)
                ref += (T - tmin) if row["needs_retrain"] else 0
                cost_m += base + is_late * (T - tmin)
                ref_m += (T - tmin) if row["needs_retrain"] else 0
            rows.append(dict(split=i, method=name, setting=setting, too_late_rate=late / n, mae=ae / n,
                             saving_vs_tmin_check=1 - cost / ref,
                             saving_recoverable_rounds_only=1 - cost_m / ref_m))
    d = pd.DataFrame(rows)
    return d.groupby(["method", "setting"])[["too_late_rate", "mae", "saving_vs_tmin_check",
                                             "saving_recoverable_rounds_only"]].mean().reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["slopes", "capacity", "variants", "perseed", "sweep"])
    args = ap.parse_args()
    man = load_manifest()
    rounds, trials, runs = build_tables(man)
    inc = [e for e in man["runs"] if e["include"]]
    disc = [e["label"] for e in inc if e["discovery"]]
    new = [e["label"] for e in inc if not e["discovery"]]
    labels = disc + new

    if "sweep" in args.parts:
        sw = tradeoff_sweep(rounds, trials, runs, new, disc)
        sw.to_csv(os.path.join(OUT, "model_checks_tradeoff_sweep.csv"), index=False)
        print("=== safety / cost trade-off (strict splits, means over 56 splits) ===")
        print(sw.round(3).to_string(index=False))

    if "slopes" in args.parts:
        run_slopes(rounds, trials)
    if "capacity" in args.parts:
        run_capacity(rounds, trials, labels)
    if "variants" in args.parts:
        run_variants(rounds, trials, runs, new, disc)
    if "perseed" in args.parts:
        run_perseed(rounds, trials, labels)


def run_slopes(rounds, trials):
    # (a) per-round slopes vs rho
    sl = per_round_slopes(trials, rounds)
    sl.to_csv(os.path.join(OUT, "model_checks_round_slopes.csv"), index=False)
    corr = slope_correlations(sl)
    corr.to_csv(os.path.join(OUT, "model_checks_slope_vs_rho.csv"), index=False)
    print("=== per-round budget slopes vs rho (searched rounds with >= 3 trials) ===")
    print(corr.round(3).to_string(index=False))


def run_capacity(rounds, trials, labels):
    # (b) capacity per round
    base = Formula("base").fit(rounds, trials)
    cap = capacity_per_round(trials, rounds, base.coef["kappa"])
    cap.to_csv(os.path.join(OUT, "model_checks_capacity_per_round.csv"), index=False)
    lin = np.polyfit(cap["ln_inv_d"], cap["cap"], 1)
    quad = np.polyfit(cap["ln_inv_d"], cap["cap"], 2)
    rl = cap["cap"] - np.polyval(lin, cap["ln_inv_d"])
    rq = cap["cap"] - np.polyval(quad, cap["ln_inv_d"])
    by_round = cap.assign(resid_linear=rl).groupby(pd.Series(np.round(-np.log(1 - cap["sparsity"]) / -np.log(0.8)).astype(int), name="round_index"))
    br = by_round.agg(n=("cap", "size"), mean_cap=("cap", "mean"), sd_cap=("cap", "std"),
                      mean_resid_linear=("resid_linear", "mean")).reset_index()
    br.to_csv(os.path.join(OUT, "model_checks_capacity_by_round.csv"), index=False)
    print("\n=== capacity per round (kappa fixed at pooled fit) ===")
    print(f"linear fit: cap = {lin[1]:.3f} + {lin[0]:.3f}*ln(1/d), resid sd {rl.std():.3f}")
    print(f"quadratic : cap = {quad[2]:.3f} + {quad[1]:.3f}*x + {quad[0]:.3f}*x^2, resid sd {rq.std():.3f}")
    print(br.round(3).to_string(index=False))

    # coefficient bootstrap for the extra terms
    cb = []
    for name, fac, coefs in [("kappa(rho)", lambda: Formula("r", "lninvd", "rho"), ["kappa0", "kappa1_rho"]),
                             ("quadratic capacity", lambda: Formula("q", "quad", "const"), ["a2"])]:
        b = coef_bootstrap(trials, labels, fac)
        full = fac().fit(rounds, trials).coef
        for c in coefs:
            cb.append(dict(model=name, coef=c, estimate=full[c], ci95_lo=b[c].quantile(.025),
                           ci95_hi=b[c].quantile(.975), frac_boot_positive=float((b[c] > 0).mean())))
    cb = pd.DataFrame(cb)
    cb.to_csv(os.path.join(OUT, "model_checks_coef_bootstrap.csv"), index=False)
    print("\n=== seed-cluster bootstrap of the extra coefficients (all 10 runs) ===")
    print(cb.round(4).to_string(index=False))


def run_variants(rounds, trials, runs, new, disc):
    # (c) held-out comparison of variants
    rows = []
    for margin in (0.0, 0.1):
        methods = variant_methods()
        for m in methods:
            m.margin = margin
            m.name = f"{m.name} @ margin {margin}"
        sc, pr, ft = fe.evaluate(fe.STRICT, methods, rounds, trials, runs, new, disc)
        summ = fe.summarize(sc, ["mae", "signed", "too_late_rate", "too_late_gt_res", "unrec_flagged",
                                 "false_stop", "saving_vs_tmin_check"])
        rm = ft.groupby("method")[["train_rmse", "test_rmse"]].mean().reset_index()
        rows.append(summ.merge(rm, on="method"))
        # paired per-seed MAE vs the proposed form
        seed_tab = fe.per_seed(pr, rounds, runs)
        ref = f"F: ln(1/d), const kappa @ margin {margin}"
        others = [m.name for m in methods if m.name != ref]
        fe.paired_tests(seed_tab, ref, others).to_csv(
            os.path.join(OUT, f"model_checks_variants_paired_margin{margin}.csv"), index=False)
        # RMSE per split, paired
        piv = ft.pivot_table(index="split", columns="method", values="test_rmse")
        rr = pd.DataFrame([dict(reference=ref, variant=o, mean_test_rmse_diff=float((piv[o] - piv[ref]).mean()),
                                frac_splits_variant_lower=float((piv[o] < piv[ref]).mean())) for o in others])
        rr.to_csv(os.path.join(OUT, f"model_checks_variants_rmse_margin{margin}.csv"), index=False)
    var = pd.concat(rows, ignore_index=True)
    var.to_csv(os.path.join(OUT, "model_checks_variants_strict.csv"), index=False)
    print("\n=== model variants, strict 5/5 splits ===")
    print(var[["method", "train_rmse", "test_rmse", "mae_mean", "mae_p05", "mae_p95", "signed_mean",
               "too_late_rate_mean", "unrec_flagged_mean", "false_stop_mean"]].round(3).to_string(index=False))
    print(pd.read_csv(os.path.join(OUT, "model_checks_variants_rmse_margin0.0.csv")).round(4).to_string(index=False))


def run_perseed(rounds, trials, labels):
    # (d) per-seed parameters and seed-to-seed spread of t*
    ps = []
    for lab in labels:
        tt = trials[trials["run"] == lab]
        if tt["round"].nunique() >= 3 and len(tt) >= 10:
            f = Formula("s").fit(None, tt)
            ps.append(dict(run=lab, n_trials=len(tt), **f.coef, rmse=f.train_rmse))
        else:
            ps.append(dict(run=lab, n_trials=len(tt)))
    ps = pd.DataFrame(ps)
    ps.to_csv(os.path.join(OUT, "model_checks_per_seed_params.csv"), index=False)
    print("\n=== per-seed fits ===")
    print(ps.round(3).to_string(index=False))
    r = rounds[rounds["t_star"].notna() & rounds["needs_retrain"]]
    sp = r.groupby("round")["t_star"].agg(["size", "median", "std", "min", "max"]).reset_index()
    mad = r.groupby("round")["t_star"].apply(lambda v: (v - v.median()).abs().mean()).rename("mean_abs_dev_from_median")
    sp = sp.merge(mad.reset_index(), on="round")
    sp.to_csv(os.path.join(OUT, "model_checks_tstar_spread_by_round.csv"), index=False)
    print("\n=== seed-to-seed spread of t* at each round (rounds that needed retraining, all 10 runs) ===")
    print(sp.round(2).to_string(index=False))
    w = (r["t_star"] - r.groupby("round")["t_star"].transform("median")).abs().mean()
    print(f"in-sample mean |t* - round median| = {w:.2f} epochs (floor for any round-only predictor)")


if __name__ == "__main__":
    main()
