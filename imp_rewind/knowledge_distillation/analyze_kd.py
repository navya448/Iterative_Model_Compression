"""Analysis of the KD experiments. Works on whatever has finished so far; run_all.py
re-runs it after every IMP chain, so analysis/ is always up to date.

    python analyze_kd.py [--runs_dir runs] [--out analysis]

Answers, in order (each section is skipped until its data exists):
  1. CE vs KD in the same setting: does KD change t*, the capacity, kappa, and does
     the parent-deficit term a2 drop to ~0 when every round learns from the dense model?
  2. Fitted parameters per setting and method, with run-bootstrap 95% CIs.
  3. Which formula variant fits best inside a setting (scale x density x damage).
  4. Transfer: leave one SETTING out and predict its t* with
        (a) one global fit on the other settings
        (b) parameters predicted from the setting's structure (meta-model)
        (c) (b) with this round's kappa taken from its probe rewind
        (d) the setting's own fit (leave-one-run-out inside it; the upper bound)
  5. Parameters against the structure descriptors, and kappa fitted round by round.

Writes CSVs, figures/*.png and summary.md to --out.
"""

import argparse
import glob
import json
import os
import sys
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kd_formula import (META_DESCRIPTORS, PARAMS, PRIMARY, VARIANTS, Formula,  # noqa: E402
                        add_features, variant_name)

warnings.filterwarnings("ignore")
METHOD_COLOR = {"kd": "#2a78d6", "ce": "#eb6834", "kd_parent": "#1baf7a"}
PRED_COLOR = {"a_global": "#2a78d6", "b_meta": "#eb6834", "c_meta_probe": "#1baf7a",
              "d_own_fit": "#eda100"}
PRED_LABEL = {"a_global": "(a) global fit", "b_meta": "(b) from structure",
              "c_meta_probe": "(c) structure + probe", "d_own_fit": "(d) own fit (upper bound)"}
OLD = ("pp", "d_global", "acc")
INK, MUTED = "#2b2b2a", "#8a8981"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "font.size": 9, "lines.linewidth": 2,
                     "legend.frameon": False})


# ==========================================================================
# loading
# ==========================================================================

def load(runs_dir):
    runs, rounds, trials = {}, [], []
    for imp_dir in sorted(glob.glob(os.path.join(runs_dir, "*", "seed*", "*"))):
        method = os.path.basename(imp_dir)
        if method == "dense" or not os.path.exists(os.path.join(imp_dir, "rounds.csv")):
            continue
        seed_dir = os.path.dirname(imp_dir)
        setting = os.path.basename(os.path.dirname(seed_dir))
        seed = int(os.path.basename(seed_dir)[4:])
        cfg = json.load(open(os.path.join(imp_dir, "config.json")))
        desc = json.load(open(os.path.join(seed_dir, "dense", "descriptors.json")))
        label = f"{setting}/seed{seed}/{method}"
        runs[label] = dict(B=np.array(cfg["B"], float), T=int(cfg["T"]), t_min=int(cfg["t_min"]),
                           t_probe=cfg.get("t_probe"), setting=setting, method=method,
                           seed=seed, done=os.path.exists(os.path.join(imp_dir, "DONE.json")),
                           desc=desc)
        r = pd.read_csv(os.path.join(imp_dir, "rounds.csv")).drop_duplicates("round", keep="last")
        r["run"], r["setting"], r["method"], r["seed"] = label, setting, method, seed
        for k in ("z_overparam", "z_difficulty", "z_log_depth"):
            r[k] = desc[k]
        rounds.append(r)
        # One CSV per chain (kd_job.py rebuilds it after every round); the per-trial
        # JSONs are only read if it is missing.
        csv_path = os.path.join(imp_dir, "rewind_trials.csv")
        if os.path.exists(csv_path):
            tj = pd.read_csv(csv_path)
        else:
            tj = pd.DataFrame([json.load(open(f)) for f in
                               glob.glob(os.path.join(imp_dir, "round_*", "trial_t*.json"))])
        if len(tj):
            tj = tj.drop(columns=["fingerprint", "method", "seed", "target_val_acc", "recovered"],
                         errors="ignore")
            keep = ["round", "dense_val_acc", "before_val_acc", "after_val_acc", "target_val_acc",
                    "D_acc_val", "D_kl_val", "D_relerr_val", "d_global", "d_flops", "d_min",
                    "d_lgeo", "sparsity"]
            tj = tj.merge(r[keep], on="round", how="inner")    # only completed rounds
            tj["run"], tj["setting"], tj["method"], tj["seed"] = label, setting, method, seed
            trials.append(tj)
    if not rounds:
        return runs, pd.DataFrame(), pd.DataFrame()
    R = add_features(pd.concat(rounds, ignore_index=True), runs)
    R["needs_retrain"] = R["after_val_acc"] < R["target_val_acc"]
    Tr = add_features(pd.concat(trials, ignore_index=True), runs) if trials else pd.DataFrame()
    if len(Tr):
        Tr["recovered"] = Tr["final_val_acc"] >= Tr["target_val_acc"]
    return runs, R, Tr


# ==========================================================================
# helpers
# ==========================================================================

def fit_or_none(variant, trials):
    try:
        return Formula(variant).fit(trials)
    except Exception:
        return None


def boot_params(variant, trials, n_boot, rng):
    runs = trials["run"].unique()
    f = fit_or_none(variant, trials)
    if f is None:
        return None, None
    if len(runs) < 2 or n_boot <= 0:
        return f.params, None
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(runs, len(runs))
        g = fit_or_none(variant, pd.concat([trials[trials["run"] == r] for r in pick]))
        if g is not None:
            draws.append(g.w)
    if not draws:
        return f.params, None
    d = np.array(draws)
    return f.params, {k: (np.percentile(d[:, i], 2.5), np.percentile(d[:, i], 97.5))
                      for i, k in enumerate(PARAMS)}


def t_scores(P, col="t_hat"):
    P = P[P["t_star"].notna()]
    if not len(P):
        return dict(n=0)
    e = P[col] - P["t_star"]
    rt = P["needs_retrain"].astype(bool)
    return dict(n=len(P), MAE=e.abs().mean(),
                MAE_retrain=e[rt].abs().mean() if rt.any() else np.nan,
                signed_retrain=e[rt].mean() if rt.any() else np.nan,
                too_late=(e > 0).mean())


def predict_rounds(formula, rounds, runs, kappas=None):
    out = []
    for i, row in rounds.iterrows():
        k = None if kappas is None else kappas.get(i)
        out.append(formula.predict_t(row, runs[row["run"]], kappa=k))
    return np.array(out)


def probe_rows(Tr, runs):
    """Probe trial of each (run, round), or nothing."""
    out = {}
    for (run, rnd), g in Tr.groupby(["run", "round"]):
        tp = runs[run]["t_probe"]
        hit = g[g["t"] == tp]
        if len(hit):
            out[(run, rnd)] = hit.iloc[0]
    return out


def md_table(df, floatfmt=".3f"):
    if df is None or not len(df):
        return "_no data yet_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def short(setting):
    """cifar100_resnet56 -> c100/r56 (figure labels)."""
    return (setting.replace("cifar", "c").replace("_resnet", "/r").replace("_vgg", "/vgg")
            .replace("_", "/"))


def savefig(fig, out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    fig.savefig(os.path.join(out, "figures", name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# sections
# ==========================================================================

def section_overview(runs, R, Tr, out, S):
    rows = []
    for label, r in runs.items():
        rr = R[R["run"] == label]
        rows.append(dict(run=label, finished=r["done"], rounds=len(rr),
                         trials=int((Tr["run"] == label).sum()) if len(Tr) else 0,
                         dense_val_acc=rr["dense_val_acc"].iloc[0] if len(rr) else np.nan,
                         final_sparsity=rr["sparsity"].max() if len(rr) else np.nan,
                         search_epochs=rr["search_epochs"].sum() if len(rr) else 0))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "runs_overview.csv"), index=False)
    S.append("## Runs found\n\n" + md_table(df, ".2f"))

    # monotonicity of recovery in t (the binary search relies on it)
    bad = []
    for (run, rnd), g in Tr.groupby(["run", "round"]) if len(Tr) else []:
        g = g.sort_values("t")
        rec = g["recovered"].to_numpy()
        if any(rec[j] and not rec[:j].all() for j in range(len(rec))):
            bad.append(f"{run} round {rnd}")
    S.append(f"Monotonicity check: {len(bad)} round(s) where a later rewind recovered after an "
             f"earlier one failed" + (": " + ", ".join(bad) if bad else ".") + "\n")


def section_ce_vs_kd(R, Tr, out, S, n_boot, rng):
    S.append("## 1. CE vs KD in the same setting\n")
    both = [s for s in R["setting"].unique()
            if {"ce", "kd"} <= set(R.loc[R["setting"] == s, "method"])]
    if not both:
        S.append("_Needs a setting with both ce and kd chains._\n")
        return
    pair_rows, par_rows = [], []
    for s in both:
        a = R[(R["setting"] == s) & (R["method"] == "ce")]
        b = R[(R["setting"] == s) & (R["method"] == "kd")]
        m = a.merge(b, on=["seed", "round"], suffixes=("_ce", "_kd"))
        for _, r in m.iterrows():
            def tmin_acc(meth, seed=r["seed"], rnd=r["round"]):
                g = Tr[(Tr["setting"] == s) & (Tr["method"] == meth) & (Tr["seed"] == seed)
                       & (Tr["round"] == rnd)]
                g = g[g["t"] == g["t"].min()] if len(g) else g
                return g["final_val_acc"].iloc[0] if len(g) else np.nan
            pair_rows.append(dict(setting=s, seed=r["seed"], round=r["round"],
                                  sparsity=r["sparsity_kd"], t_star_ce=r["t_star_ce"],
                                  t_star_kd=r["t_star_kd"],
                                  d_t_star=r["t_star_kd"] - r["t_star_ce"],
                                  acc_tmin_ce=tmin_acc("ce"), acc_tmin_kd=tmin_acc("kd"),
                                  parent_def_ce=r["parent_pp_ce"], parent_def_kd=r["parent_pp_kd"],
                                  search_epochs_ce=r["search_epochs_ce"],
                                  search_epochs_kd=r["search_epochs_kd"]))
        for meth in ("ce", "kd"):
            tr = Tr[(Tr["setting"] == s) & (Tr["method"] == meth)]
            for v in (OLD, PRIMARY):
                p, ci = boot_params(v, tr, n_boot, rng)
                if p is None:
                    continue
                for k in PARAMS:
                    par_rows.append(dict(setting=s, method=meth, variant=variant_name(v), param=k,
                                         value=p[k], lo=ci[k][0] if ci else np.nan,
                                         hi=ci[k][1] if ci else np.nan))
    P = pd.DataFrame(pair_rows)
    P.to_csv(os.path.join(out, "ce_vs_kd_rounds.csv"), index=False)
    Q = pd.DataFrame(par_rows)
    Q.to_csv(os.path.join(out, "ce_vs_kd_params.csv"), index=False)

    P["d_acc_tmin"] = P["acc_tmin_kd"] - P["acc_tmin_ce"]
    summ = P.groupby("setting").agg(paired_rounds=("round", "size"),
                                    mean_d_t_star=("d_t_star", "mean"),
                                    mean_d_acc_at_tmin=("d_acc_tmin", "mean"),
                                    search_epochs_ce=("search_epochs_ce", "sum"),
                                    search_epochs_kd=("search_epochs_kd", "sum")).reset_index()
    S.append("Paired by seed and round (same dense model, same data order in every trial). "
             "Round 1 is an exact pair; later rounds start from different parents, because each "
             "chain continues from its own retrained model.\n\n"
             "`mean_d_t_star` > 0: KD allows a later rewind (cheaper retraining). "
             "`mean_d_acc_at_tmin` > 0: KD reaches higher accuracy with the full budget "
             "(more capacity).\n\n" + md_table(summ, ".2f"))
    if len(Q):
        show = Q.copy()
        show["95% CI"] = [f"[{l:.3f}, {h:.3f}]" if np.isfinite(l) else "" for l, h in
                          zip(show["lo"], show["hi"])]
        S.append("Parameters (badness convention, see kd_formula.py; for pp, a0 = -c0, a1 = "
                 "c_density, a2 = c_parent). The professor's question is mainly c_parent: "
                 "if its CI under kd contains 0, the parent term can be dropped.\n\n"
                 + md_table(show[["setting", "method", "variant", "param", "value", "95% CI"]]))

    # figure: parameters CE vs KD
    for s in both:
        q = Q[(Q["setting"] == s) & (Q["variant"] == variant_name(PRIMARY))]
        if not len(q):
            continue
        fig, axes = plt.subplots(1, len(PARAMS), figsize=(13, 3), layout="constrained")
        for ax, k in zip(axes, PARAMS):
            for j, meth in enumerate(("ce", "kd")):
                r = q[(q["param"] == k) & (q["method"] == meth)]
                if not len(r):
                    continue
                r = r.iloc[0]
                err = None if not np.isfinite(r["lo"]) else [[r["value"] - r["lo"]], [r["hi"] - r["value"]]]
                ax.errorbar([j], [r["value"]], yerr=err, fmt="o", ms=7, capsize=3,
                            color=METHOD_COLOR[meth], label=meth)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks([0, 1], ["ce", "kd"])
            ax.set_xlim(-0.6, 1.6)
            ax.set_title(k, color=INK)
        fig.suptitle(f"{s}: parameters of {variant_name(PRIMARY)} (95% run-bootstrap CI)",
                     color=INK)
        savefig(fig, out, f"ce_vs_kd_params_{s}.png")


def section_params(R, Tr, out, S, n_boot, rng):
    S.append("## 2. Parameters per setting and method\n")
    rows = []
    for (s, meth), tr in Tr.groupby(["setting", "method"]):
        for v in (OLD, PRIMARY):
            p, ci = boot_params(v, tr, n_boot, rng)
            if p is None:
                continue
            rows.append(dict(setting=s, method=meth, variant=variant_name(v),
                             n_runs=tr["run"].nunique(), n_trials=len(tr),
                             **{k: p[k] for k in PARAMS},
                             **{f"{k}_lo": ci[k][0] if ci else np.nan for k in PARAMS},
                             **{f"{k}_hi": ci[k][1] if ci else np.nan for k in PARAMS}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "params_by_group.csv"), index=False)
    if len(df):
        S.append(md_table(df[["setting", "method", "variant", "n_runs", "n_trials", *PARAMS]]))


def section_variants(R, Tr, runs, out, S):
    S.append("## 3. Formula variants inside each setting\n")
    rows = []
    for (s, meth), tr in Tr.groupby(["setting", "method"]):
        rr = R[(R["setting"] == s) & (R["method"] == meth)]
        run_ids = tr["run"].unique()
        for v in VARIANTS:
            f = fit_or_none(v, tr)
            if f is None:
                continue
            acc_rmse = float(np.sqrt(np.mean((f.acc(tr) - tr["final_val_acc"]) ** 2)))
            row = dict(setting=s, method=meth, variant=variant_name(v), n_runs=len(run_ids),
                       acc_rmse_in_sample=acc_rmse)
            if len(run_ids) >= 2:
                preds = []
                for r in run_ids:
                    g = fit_or_none(v, tr[tr["run"] != r])
                    if g is None:
                        continue
                    q = rr[rr["run"] == r].copy()
                    q["t_hat"] = predict_rounds(g, q, runs)
                    preds.append(q)
                if preds:
                    sc = t_scores(pd.concat(preds))
                    row.update({f"loro_{k}": v_ for k, v_ in sc.items() if k != "n"})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "variant_scores.csv"), index=False)
    if len(df):
        col = "loro_MAE" if "loro_MAE" in df else "acc_rmse_in_sample"
        S.append(f"Sorted by {col} (t* error in epochs, leave one run out inside the setting) "
                 f"where there are 2+ runs. `acc_rmse_in_sample` is in pp for every variant, so "
                 f"the scales can be compared. The old formula is `{variant_name(OLD)}`; the "
                 f"pre-registered one is `{variant_name(PRIMARY)}`.\n\n"
                 + md_table(df.sort_values(["setting", "method", col]), ".2f"))


def section_transfer(R, Tr, runs, out, S, method="kd"):
    S.append(f"## 4. Transfer to a new setting (method = {method})\n")
    Rm, Tm = R[R["method"] == method], Tr[Tr["method"] == method] if len(Tr) else Tr
    settings = sorted(Tm["setting"].unique()) if len(Tm) else []
    if len(settings) < 2:
        S.append("_Needs at least two settings with finished rounds._\n")
        return
    probes = probe_rows(Tm, runs)
    desc = Rm.groupby("setting")[["z_overparam", "z_difficulty", "z_log_depth"]].mean()
    pred_rows = []
    for v in VARIANTS:
        own = {s: fit_or_none(v, Tm[Tm["setting"] == s]) for s in settings}
        for s in settings:
            rr = Rm[Rm["setting"] == s].copy()
            others = [o for o in settings if o != s and own[o] is not None]
            if not len(rr) or not others:
                continue
            # (a) one global fit
            ga = fit_or_none(v, Tm[Tm["setting"] != s])
            # (b) meta-model on per-setting parameters
            meta = {}
            for k in PARAMS:
                vals = np.array([own[o].params[k] for o in others])
                z = META_DESCRIPTORS[k]
                if z is not None and len(others) >= 3:
                    zs = desc.loc[others, z].to_numpy()
                    slope, icpt = np.polyfit(zs, vals, 1)
                    meta[k] = icpt + slope * desc.loc[s, z]
                else:
                    meta[k] = vals.mean()
            gb = Formula(v).set_params(meta)
            # (c) meta + this round's probe kappa
            kap = {}
            for i, row in rr.iterrows():
                kap[i] = gb.probe_kappa(row, probes.get((row["run"], row["round"])))
            # (d) own fit, leave one run out inside the setting if possible
            d_hat = np.full(len(rr), np.nan)
            s_runs = rr["run"].unique()
            for r in s_runs:
                pool = Tm[(Tm["setting"] == s) & ((Tm["run"] != r) if len(s_runs) > 1 else True)]
                g = fit_or_none(v, pool)
                if g is not None:
                    idx = (rr["run"] == r).to_numpy()
                    d_hat[idx] = predict_rounds(g, rr[idx], runs)
            preds = dict(a_global=predict_rounds(ga, rr, runs) if ga else np.nan,
                         b_meta=predict_rounds(gb, rr, runs),
                         c_meta_probe=predict_rounds(gb, rr, runs, kap),
                         d_own_fit=d_hat)
            for j, (_, row) in enumerate(rr.iterrows()):
                for pname, arr in preds.items():
                    pred_rows.append(dict(variant=variant_name(v), held_out=s, run=row["run"],
                                          round=row["round"], t_star=row["t_star"],
                                          needs_retrain=row["needs_retrain"], predictor=pname,
                                          t_hat=arr[j] if np.ndim(arr) else np.nan,
                                          own_fit_is_in_sample=len(s_runs) < 2))
    P = pd.DataFrame(pred_rows)
    P.to_csv(os.path.join(out, f"transfer_predictions_{method}.csv"), index=False)
    rows = []
    for (v, pname), g in P.groupby(["variant", "predictor"]):
        g = g[g["t_hat"].notna()]
        rows.append(dict(variant=v, predictor=pname, **t_scores(g)))
    sc = pd.DataFrame(rows)
    piv = sc.pivot(index="variant", columns="predictor", values="MAE")
    if {"a_global", "b_meta", "d_own_fit"} <= set(piv.columns):
        piv["gap_closed_by_b"] = (piv["a_global"] - piv["b_meta"]) / (piv["a_global"] - piv["d_own_fit"])
    if {"a_global", "c_meta_probe", "d_own_fit"} <= set(piv.columns):
        piv["gap_closed_by_c"] = (piv["a_global"] - piv["c_meta_probe"]) / (piv["a_global"] - piv["d_own_fit"])
    piv = piv.reset_index()
    sc.to_csv(os.path.join(out, f"transfer_scores_{method}.csv"), index=False)
    piv.to_csv(os.path.join(out, f"transfer_mae_{method}.csv"), index=False)
    n_set = len(settings)
    S.append(f"Leave one setting out over {n_set} settings: {', '.join(settings)}. t* MAE in "
             "epochs. `gap_closed_by_b` = share of the gap between (a) and (d) that the "
             "structure-based parameters close (1 = as good as fitting the setting itself). "
             "The meta-model uses a descriptor only when 3+ training settings exist, otherwise "
             "the mean. With few settings this is a noisy estimate; choosing the best of "
             f"{len(VARIANTS)} variants flatters it, so read the pre-registered row "
             f"`{variant_name(PRIMARY)}` first.\n\n"
             + md_table(piv.sort_values("b_meta" if "b_meta" in piv else piv.columns[1]), ".2f"))

    # figure: predicted vs actual for the primary variant
    pv = P[(P["variant"] == variant_name(PRIMARY)) & P["t_star"].notna()]
    if len(pv):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True, sharey=True,
                                 layout="constrained")
        for ax, pname in zip(axes, PRED_LABEL):
            g = pv[pv["predictor"] == pname]
            ax.plot([0, 80], [0, 80], color=MUTED, lw=1, ls="--")
            ax.scatter(g["t_star"], g["t_hat"], s=22, color=PRED_COLOR[pname],
                       edgecolor="white", linewidth=0.8)
            sc_ = t_scores(g[g["t_hat"].notna()])
            ax.set_title(f"{PRED_LABEL[pname]}\nMAE {sc_.get('MAE', np.nan):.1f}", color=INK)
            ax.set_xlabel("measured t*")
        axes[0].set_ylabel("predicted t*")
        fig.suptitle(f"Leave-one-setting-out, {variant_name(PRIMARY)}, method {method}", color=INK)
        savefig(fig, out, f"transfer_{method}.png")


def section_structure(R, Tr, out, S):
    S.append("## 5. Parameters against structure, and kappa round by round\n")
    rows = []
    for (s, meth), tr in Tr.groupby(["setting", "method"]):
        f = fit_or_none(PRIMARY, tr)
        if f is None:
            continue
        d = R[(R["setting"] == s) & (R["method"] == meth)].iloc[0]
        rows.append(dict(setting=s, method=meth, **f.params, z_overparam=d["z_overparam"],
                         z_difficulty=d["z_difficulty"], z_log_depth=d["z_log_depth"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "params_vs_descriptors.csv"), index=False)
    if len(df):
        S.append(md_table(df))
        pairs = [("c_density", "z_overparam"), ("k0", "z_overparam"),
                 ("k_damage", "z_log_depth"), ("c_parent", "z_overparam")]
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.6),
                                 layout="constrained")
        for ax, (k, z) in zip(np.atleast_1d(axes), pairs):
            for meth, g in df.groupby("method"):
                ax.scatter(g[z], g[k], s=40, color=METHOD_COLOR.get(meth, MUTED), label=meth,
                           edgecolor="white", linewidth=0.8)
                for _, r in g.iterrows():
                    ax.annotate(short(r["setting"]), (r[z], r[k]), fontsize=7, color=MUTED,
                                xytext=(4, 2), textcoords="offset points")
            ax.margins(x=0.25, y=0.15)
            ax.set_xlabel(z)
            ax.set_ylabel(k)
        np.atleast_1d(axes)[0].legend()
        fig.suptitle(f"{variant_name(PRIMARY)} parameters per setting vs structure", color=INK)
        savefig(fig, out, "params_vs_descriptors.png")

    # kappa per round: slope of bad_ln on clock inside one round (3+ trials)
    kr = []
    for (run, rnd), g in Tr.groupby(["run", "round"]):
        if len(g) >= 3 and g["clock"].nunique() >= 3:
            slope = np.polyfit(g["clock"], g["bad_ln"], 1)[0]
            kr.append(dict(run=run, setting=g["setting"].iloc[0], method=g["method"].iloc[0],
                           round=rnd, sparsity=g["sparsity"].iloc[0], kappa_ln=slope,
                           n_trials=len(g)))
    K = pd.DataFrame(kr)
    K.to_csv(os.path.join(out, "kappa_per_round.csv"), index=False)
    if len(K):
        sets = sorted(K["setting"].unique())
        fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.2), sharey=True,
                                 squeeze=False, layout="constrained")
        for ax, s in zip(axes[0], sets):
            for meth, g in K[K["setting"] == s].groupby("method"):
                ax.scatter(g["round"], g["kappa_ln"], s=18, color=METHOD_COLOR.get(meth, MUTED),
                           label=meth, edgecolor="white", linewidth=0.6)
                mean = g.groupby("round")["kappa_ln"].mean()
                ax.plot(mean.index, mean.values, color=METHOD_COLOR.get(meth, MUTED), lw=1.5)
            ax.set_title(s, color=INK)
            ax.set_xlabel("round")
        axes[0][0].set_ylabel("kappa (e-folds of error per e-fold of budget)")
        axes[0][0].legend()
        savefig(fig, out, "kappa_per_round.png")


def figure_tstar(R, out):
    sets = sorted(R["setting"].unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.2), sharey=True,
                             squeeze=False, layout="constrained")
    for ax, s in zip(axes[0], sets):
        for meth, g in R[R["setting"] == s].groupby("method"):
            c = METHOD_COLOR.get(meth, MUTED)
            ax.scatter(g["round"], g["t_star"], s=14, color=c, alpha=0.6, edgecolor="none")
            mean = g.groupby("round")["t_star"].mean()
            ax.plot(mean.index, mean.values, color=c, label=meth)
        ax.set_title(s, color=INK)
        ax.set_xlabel("IMP round")
    axes[0][0].set_ylabel("t* (latest rewind that recovers)")
    axes[0][0].legend()
    savefig(fig, out, "tstar_by_round.png")


# ==========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir", default=os.path.join(HERE, "runs"))
    p.add_argument("--out", default=os.path.join(HERE, "analysis"))
    p.add_argument("--n_boot", type=int, default=300)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    runs, R, Tr = load(args.runs_dir)
    S = ["# KD experiments: results so far\n",
         "_Generated by analyze_kd.py; re-run after new chains finish. All accuracies are "
         "validation accuracy; t* uses the relative-error target err <= err_dense * (1 + rel_tol). "
         "A setting/method whose trials cannot identify all 5 parameters (e.g. its chains "
         "stopped after a few rounds) is left out of the fits instead of reporting arbitrary "
         "values._\n"]
    if not len(R):
        S.append("No finished IMP rounds yet.\n")
    else:
        R.to_csv(os.path.join(args.out, "all_rounds.csv"), index=False)
        if len(Tr):
            Tr.to_csv(os.path.join(args.out, "all_trials.csv"), index=False)
        sections = [lambda: section_overview(runs, R, Tr, args.out, S),
                    lambda: figure_tstar(R, args.out)]
        if len(Tr):
            sections += [lambda: section_ce_vs_kd(R, Tr, args.out, S, args.n_boot, rng),
                         lambda: section_params(R, Tr, args.out, S, args.n_boot, rng),
                         lambda: section_variants(R, Tr, runs, args.out, S),
                         lambda: section_transfer(R, Tr, runs, args.out, S, "kd"),
                         lambda: section_structure(R, Tr, args.out, S)]
        for fn in sections:
            try:
                fn()
            except Exception:
                S.append("**A section failed:**\n\n```\n" + traceback.format_exc() + "```\n")
                traceback.print_exc()
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S))
    print(f"Analysis written to {args.out}")


if __name__ == "__main__":
    main()
