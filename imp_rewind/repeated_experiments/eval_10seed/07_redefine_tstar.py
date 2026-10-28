"""Step 7: a steadier definition of t*.

Two changes to how t* is measured, both computed from the trials that already exist
(no new training):

  1. Population target. The protocol's target is this run's dense val acc - 0.5, and
     that dense accuracy is a single measurement (90.62 to 91.12 across the 10 runs).
     The population target is the mean dense val acc of the 10 included runs - 0.5,
     the same finish line for every run.
  2. Smoothed t*. Instead of reading t* off a binary search over single noisy retrains,
     fit a non-increasing curve (isotonic regression, i.e. "later rewinds never recover
     better", the same assumption the binary search makes) through every rewind trial of
     the round plus the un-retrained pruned model at t = T, and take t* as the last epoch
     where that curve is at or above the target (linear interpolation between the two
     trials that bracket the crossing). No functional form is assumed, so the formula
     being evaluated later is not built into the target. The crossing is only trusted when
     the two bracketing points are at most MAX_BRACKET = 10 epochs apart (about one late
     binary-search step); wider brackets are reported as "censored" with their interval.

Rounds whose pruned model already meets the target keep t* = T (no retraining), as in
the protocol. A round is "unrecoverable" if the fitted curve is below the target even at
t_min, and "censored" if the data cannot locate the crossing (the target is reached at
t_min but there is no trial between t_min and T).

Writes results/tstar_redefined.csv, results/tstar_dispersion.csv, results/tstar_dense_link.csv,
figures/fig8_tstar_redefined.(png|pdf).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common_eval import FIG, OUT, build_tables, load_manifest

EPS = 1e-9
MAX_BRACKET = 10


def pava_nonincreasing(y):
    """Least-squares non-increasing fit (pool-adjacent-violators)."""
    blocks = []                                    # run PAVA for non-decreasing on reversed data
    for v in list(map(float, y))[::-1]:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2 = blocks.pop()
            v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    out = []
    for v, n in blocks:
        out += [v] * n
    return np.array(out[::-1])


def smoothed_tstar(ts, accs, after, target, T, t_min, max_bracket=MAX_BRACKET):
    """Returns (t_star, status, bracket_lo, bracket_hi)."""
    if after >= target - EPS:
        return T, "no_retrain", T, T
    order = np.argsort(ts)
    t = np.append(np.asarray(ts, float)[order], T)
    a = np.append(np.asarray(accs, float)[order], after)
    if len(t) < 2 or t[0] != t_min:
        return np.nan, "censored", np.nan, np.nan
    f = pava_nonincreasing(a)
    if f[0] < target - EPS:
        return np.nan, "unrecoverable", np.nan, np.nan
    i = int(np.where(f >= target - EPS)[0].max())
    j = i + 1                                       # exists: f[-1] = after < target after pooling
    if t[j] - t[i] > max_bracket:
        return np.nan, "censored", t[i], t[j]       # crossing not located by the data
    tc = t[i] + (f[i] - target) * (t[j] - t[i]) / (f[i] - f[j]) if f[i] > f[j] else t[i]
    tstar = int(np.clip(np.floor(tc + EPS), t[i], t[j] - 1))
    return tstar, "searched", t[i], t[j]


def redefine(rounds, trials, runs, a_pop, tol=0.5, max_bracket=MAX_BRACKET):
    rows = []
    for _, r in rounds.iterrows():
        run = runs[r["run"]]
        tr = trials[(trials["run"] == r["run"]) & (trials["round"] == r["round"])]
        own = run["target"]
        pop = a_pop - tol
        out = dict(run=r["run"], group=r["group"], round=int(r["round"]), sparsity=r["sparsity"],
                   after_val_acc=r["after_val_acc"], n_trials=len(tr), status_raw=r["round_status"],
                   t_star_raw=r["t_star"], target_own=own, target_pop=pop, dense_val_acc=run["a_dense"])
        for name, tg in (("own", own), ("pop", pop)):
            t, st, lo, hi = smoothed_tstar(tr["t"].to_numpy(), tr["final_val_acc"].to_numpy(),
                                           r["after_val_acc"], tg, run["T"], run["t_min"], max_bracket)
            out.update({f"t_star_{name}": t, f"status_{name}": st, f"bracket_lo_{name}": lo,
                        f"bracket_hi_{name}": hi})
        rows.append(out)
    return pd.DataFrame(rows)


def dispersion(df, runs, cols):
    """Seed-to-seed spread of t* at each round index, over rounds that needed retraining
    and have a t* under EVERY definition in `cols` (same rounds for a fair comparison)."""
    B = next(iter(runs.values()))["B"]
    ok = np.ones(len(df), bool)
    for c, st in cols:
        ok &= (df[st] == "searched").to_numpy()
    d = df[ok].copy()
    rows = []
    for k, g in d.groupby("round"):
        if len(g) < 3:
            continue
        row = dict(round=k, n=len(g))
        for c, _ in cols:
            v = g[c].astype(float)
            u = np.log(B[v.astype(int)])
            row[f"sd_{c}"] = v.std()
            row[f"mad_{c}"] = (v - v.median()).abs().mean()
            row[f"sd_lnB_{c}"] = u.std()
        rows.append(row)
    tab = pd.DataFrame(rows)
    pooled = dict(round="pooled (sqrt mean var)", n=int(tab["n"].sum()))
    for c, _ in cols:
        w = tab["n"] - 1
        pooled[f"sd_{c}"] = np.sqrt((w * tab[f"sd_{c}"] ** 2).sum() / w.sum())
        pooled[f"mad_{c}"] = (tab["n"] * tab[f"mad_{c}"]).sum() / tab["n"].sum()
        pooled[f"sd_lnB_{c}"] = np.sqrt((w * tab[f"sd_lnB_{c}"] ** 2).sum() / w.sum())
    return pd.concat([tab, pd.DataFrame([pooled])], ignore_index=True), d


def dense_link(df, runs, col, status):
    """Does a run's own dense val acc predict whether its t* is early or late? Per run:
    mean over its searched rounds of (ln B(t*) - round median of ln B(t*))."""
    B = next(iter(runs.values()))["B"]
    d = df[df[status] == "searched"].copy()
    d["u"] = np.log(B[d[col].astype(int)])
    d["u_dev"] = d["u"] - d.groupby("round")["u"].transform("median")
    per = d.groupby("run").agg(u_dev=("u_dev", "mean"), n=("u_dev", "size"),
                               dense_val_acc=("dense_val_acc", "first")).reset_index()
    per = per[per["n"] >= 2]
    rho, p = stats.spearmanr(per["dense_val_acc"], per["u_dev"])
    return per, rho, p


def figure(df, disp, runs):
    BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a94", "#1f1f1e", "#6b6a64", "#e6e5df"
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.3))
    t = disp[disp["round"] != "pooled (sqrt mean var)"]
    x = np.arange(len(t))
    for k, (c, col, lab) in enumerate([("t_star_raw", GRAY, "binary search, own target (protocol)"),
                                       ("t_star_own", AQUA, "curve through all trials, own target"),
                                       ("t_star_pop", BLUE, "curve through all trials, common target")]):
        a.bar(x + (k - 1) * 0.27, t[f"sd_{c}"], width=0.25, color=col, label=lab)
    a.set_xticks(x)
    a.set_xticklabels([f"r{int(r)}\n(n={int(n)})" for r, n in zip(t["round"], t["n"])], fontsize=7.5)
    a.set_ylabel("SD of t* across runs (epochs)")
    a.set_title("(a) Seed-to-seed spread of t* at each round", loc="left", fontsize=9)
    a.grid(axis="y", color=GRID, lw=0.7)
    a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=7, loc="upper left")
    a.set_ylim(0, max(t[[f"sd_{c}" for c in ("t_star_raw", "t_star_own", "t_star_pop")]].max()) * 1.3)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
        b.spines[s].set_visible(False)
    rng = np.random.default_rng(0)
    for st, col, mk, lab in [("searched", BLUE, "o", "t* (common target)")]:
        d = df[df["status_pop"] == st]
        b.scatter(d["sparsity"] * 100 + rng.uniform(-.6, .6, len(d)), d["t_star_pop"], s=16, color=col, marker=mk,
                  edgecolor="white", lw=0.5, label=lab, zorder=3)
    u = df[df["status_pop"] == "unrecoverable"]
    b.scatter(u["sparsity"] * 100, np.full(len(u), -3), marker="x", color=ORANGE, s=22, label="unrecoverable", zorder=3)
    c = df[df["status_pop"] == "censored"]
    if len(c):
        b.scatter(c["sparsity"] * 100, np.full(len(c), -3), marker="|", color=MUTED, s=40, label="censored", zorder=3)
    for m in (40, 60):
        b.axhline(m, color=MUTED, lw=0.7, ls=":")
    b.grid(color=GRID, lw=0.7)
    b.set_axisbelow(True)
    b.set_xlabel("sparsity (%)")
    b.set_ylabel("t*")
    b.set_title("(b) Redefined t* (common target)", loc="left", fontsize=9)
    b.legend(frameon=False, fontsize=7, loc="upper right")
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig8_tstar_redefined.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    man = load_manifest()
    rounds, trials, runs = build_tables(man)
    a_dense = pd.Series({k: v["a_dense"] for k, v in runs.items()})
    a_pop = float(a_dense.mean())
    print(f"dense val acc over {len(a_dense)} runs: mean {a_pop:.3f}, sd {a_dense.std():.3f}, "
          f"range {a_dense.min():.2f}-{a_dense.max():.2f}; population target {a_pop - 0.5:.3f}")
    json.dump(dict(a_pop=a_pop, target_pop=a_pop - 0.5, dense_val_acc=a_dense.to_dict(),
                   note="mean dense val acc of the included runs"),
              open(os.path.join(OUT, "population_target.json"), "w"), indent=2)

    df = redefine(rounds, trials, runs, a_pop)
    df.to_csv(os.path.join(OUT, "tstar_redefined.csv"), index=False)
    print("\nstatus counts:")
    for c in ("status_raw", "status_own", "status_pop"):
        print(f"  {c}: {df[c].value_counts().to_dict()}")
    ch = df[df["status_raw"] != df["status_pop"]]
    print(f"\nrounds whose status changes under the common target: {len(ch)}")
    print(ch[["run", "round", "sparsity", "status_raw", "t_star_raw", "status_pop", "t_star_pop",
              "target_own", "target_pop"]].round(3).to_string(index=False))

    s = df[df["status_pop"] == "searched"]
    w = s["bracket_hi_pop"] - s["bracket_lo_pop"]
    print(f"\nbracket width around the crossing (common target): median {w.median():.0f}, "
          f"<=5: {(w <= 5).mean():.0%}, >20: {(w > 20).sum()} rounds")
    s2 = df[df["status_own"] == "searched"]
    print(f"smoothed vs raw t* (own target), same rounds: mean |diff| "
          f"{(s2['t_star_own'] - s2['t_star_raw']).abs().mean():.2f}, "
          f"rounds that differ: {(s2['t_star_own'] != s2['t_star_raw']).sum()} of {len(s2)}")

    alt = redefine(rounds, trials, runs, a_pop, max_bracket=20)
    print(f"sensitivity, brackets up to 20 epochs accepted: status_pop counts {alt['status_pop'].value_counts().to_dict()}")
    alt.to_csv(os.path.join(OUT, "tstar_redefined_bracket20.csv"), index=False)

    cols = [("t_star_raw", "status_raw"), ("t_star_own", "status_own"), ("t_star_pop", "status_pop")]
    disp, common = dispersion(df, runs, cols)
    disp.to_csv(os.path.join(OUT, "tstar_dispersion.csv"), index=False)
    print("\n=== seed-to-seed spread of t* at each round (same rounds under all definitions) ===")
    print(disp.round(2).to_string(index=False))

    rows = []
    for c, st in cols:
        per, rho, p = dense_link(df, runs, c, st)
        rows.append(dict(definition=c, n_runs=len(per), spearman_dense_acc_vs_tstar_dev=rho, p=p))
        per.to_csv(os.path.join(OUT, f"tstar_dense_link_{c}.csv"), index=False)
    link = pd.DataFrame(rows)
    link.to_csv(os.path.join(OUT, "tstar_dense_link.csv"), index=False)
    print("\n=== does the run's own dense accuracy push its t* early/late? ===")
    print("(per run: mean deviation of ln B(t*) from the round median; earlier t* = more budget left = higher value,\n so positive rho = runs with higher dense acc get earlier t*)")
    print(link.round(3).to_string(index=False))

    figure(df, disp, runs)
    print("\nwrote results/tstar_redefined.csv, tstar_dispersion.csv, tstar_dense_link.csv, figures/fig8")


if __name__ == "__main__":
    main()
