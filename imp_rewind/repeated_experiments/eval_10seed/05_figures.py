"""Step 5: figures for the report (PNG at 300 dpi and PDF), from the tables written by
steps 3 and 4 plus the raw rounds/trials. Run after 03_formula_eval.py and 04_model_checks.py.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_eval import FIG, OUT, build_tables, load_manifest
from models import Formula, SparsityLine

BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, MUTED, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                     "#9b9a94", "#1f1f1e", "#6b6a64", "#e6e5df")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def grid(ax):
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def t_hat_curve(model, run, s_grid):
    out = []
    for s in s_grid:
        row = dict(after_val_acc=-np.inf, target_val_acc=run["target"], round=int(round(np.log(1 - s) / np.log(0.8))),
                   ln_inv_d=-np.log(1 - s), weight_perturbation=0.2, sparsity=s)
        t, flag = model.predict(row, run)
        out.append(np.nan if flag == "stop" else t)
    return np.array(out)


def fig_tstar(rounds, trials, runs):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    grid(ax)
    rng = np.random.default_rng(0)
    for grp, col, mk, lab in [("repeated", BLUE, "o", "new runs (seeds 3-10)"),
                              ("original", ORANGE, "s", "discovery runs (orig. seeds 1-2)")]:
        r = rounds[(rounds["group"] == grp)]
        m = r[r["round_status"] == "searched"]
        jit = rng.uniform(-0.6, 0.6, len(m))
        ax.scatter(m["sparsity"] * 100 + jit, m["t_star"], s=22, color=col, marker=mk, edgecolor="white",
                   lw=0.6, zorder=3, label=f"measured t*, {lab}")
        u = r[r["round_status"] == "unrecoverable"]
        if len(u):
            ax.scatter(u["sparsity"] * 100 + rng.uniform(-0.6, 0.6, len(u)), np.full(len(u), -3), marker="x",
                       s=26, color=col, lw=1.2, zorder=3)
    any_run = next(iter(runs.values()))
    s_grid = np.linspace(0.45, 0.93, 600)
    f0 = Formula("f0").fit(rounds, trials)
    f1 = Formula("f1", margin=0.1).fit(rounds, trials)
    sl = SparsityLine("sl", check=True).fit(rounds, trials)
    ax.plot(s_grid * 100, t_hat_curve(f0, any_run, s_grid), color=INK, lw=1.6, label="formula, margin 0 (fit on all 10 runs)")
    ax.plot(s_grid * 100, t_hat_curve(f1, any_run, s_grid), color=INK, lw=1.4, ls="--", label="formula, margin 0.1 pp")
    ax.plot(s_grid * 100, np.clip(sl.b0 + sl.b1 * s_grid, 2, 80), color=AQUA, lw=1.4, label="sparsity line (rounds needing retraining)")
    lim = 1 - np.exp(-(0.5 + f0.coef["a0"]) / f0.coef["a1"])
    ax.axvspan(lim * 100, 93, color=GRID, alpha=0.8, lw=0)
    ax.text(lim * 100 + 0.3, 76, f"predicted\ncapacity limit\n(s > {lim*100:.1f}%)", fontsize=7.5, color=MUTED, va="top")
    for m in (40, 60):
        ax.axhline(m, color=MUTED, lw=0.7, ls=":")
    ax.text(50.2, -2.6, "x = round where even t=2 failed", fontsize=7.5, color=MUTED, va="center")
    ax.set_xlim(45, 93)
    ax.set_ylim(-6, 82)
    ax.set_xlabel("sparsity (%)")
    ax.set_ylabel("latest working rewind epoch t*")
    ax.set_title("Measured t* in every round that needed retraining (10 runs)", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "fig1_tstar_vs_sparsity")


def fig_budget(rounds, trials, runs):
    f = Formula("f").fit(rounds, trials)
    ps = pd.read_csv(os.path.join(OUT, "model_checks_per_seed_params.csv")).dropna(subset=["kappa"])
    par = pd.read_csv(os.path.join(OUT, "formula_params.csv"))
    ci = par.loc[par["fit_on"] == "all 10 runs", "kappa_ci95"].iloc[0]
    lo, hi = [float(v) for v in ci.strip("[]").split(",")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[2.1, 1]))
    grid(a)
    cap = f.coef["a0"] - f.coef["a1"] * trials["ln_inv_d"]
    yy = trials["y"] - cap
    for grp, col, lab in [("repeated", BLUE, "new runs"), ("original", ORANGE, "discovery runs")]:
        m = trials["run"].map(lambda r: runs[r]["group"]) == grp
        a.scatter(trials.loc[m, "clock"], yy[m], s=9, color=col, alpha=0.7, lw=0, label=lab)
    xs = np.linspace(0, trials["clock"].max(), 50)
    a.plot(xs, -f.coef["kappa"] * xs, color=INK, lw=1.6, label=f"slope -kappa = -{f.coef['kappa']:.3f} pp / e-fold")
    a.set_ylim(-1.6, 1.2)
    B = next(iter(runs.values()))["B"]
    for t in (40, 60, 78):
        x = np.log(B[2] / B[t])
        a.axvline(x, color=MUTED, lw=0.7, ls=":")
        a.text(x + 0.05, 1.15, f"t={t}", fontsize=7.5, color=MUTED, va="top")
    a.set_xlabel("ln( B(t_min) / B(t) )  = LR budget given up, in e-folds")
    a.set_ylabel("final val acc - A_dense - capacity(d)  (pp)")
    a.set_title("(a) All 369 rewind trials, capacity term removed", loc="left")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    grid(b)
    b.axhspan(lo, hi, color=GRID, lw=0)
    b.axhline(f.coef["kappa"], color=INK, lw=1.2)
    cols = [ORANGE if r.endswith("_orig") else BLUE for r in ps["run"]]
    b.scatter(range(len(ps)), ps["kappa"], color=cols, s=26, zorder=3, edgecolor="white", lw=0.6)
    b.set_xticks(range(len(ps)))
    b.set_xticklabels([r.replace("seed", "s").replace("_orig", "o") for r in ps["run"]], rotation=0, fontsize=7)
    b.set_ylabel("kappa (pp per e-fold)")
    b.set_title(f"(b) kappa fitted per run\npooled {f.coef['kappa']:.3f}, band = 95% CI [{lo:.3f}, {hi:.3f}]",
                loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig2_budget_law")


METHODS_SHOWN = ["Formula (margin 0.1 pp)", "Formula (margin 0)", "Sparsity line + check", "Round median + check",
                 "Sparsity line", "Loss match (train) + check", "Loss match (train)", "Accuracy match (val) + check",
                 "Fixed: before 1st LR drop + check", "Fixed: before last LR drop + check", "t_min + check"]


def fig_heldout():
    s = pd.read_csv(os.path.join(OUT, "heldout_summary_strict.csv")).set_index("method").loc[METHODS_SHOWN]
    s = s.iloc[::-1]
    y = np.arange(len(s))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.8), sharey=True, gridspec_kw=dict(wspace=0.12))
    panels = [("mae_retrain_rounds", "t* MAE (epochs)\nrounds needing retraining", 1),
              ("too_late_rate", "too-late rate\n(t_hat > t*)", 100),
              ("saving_vs_tmin_check", "retraining saved (%)\nvs t_min + check", 100)]
    for ax, (col, title, mult) in zip(axes, panels):
        grid(ax)
        colr = [BLUE if m.startswith("Formula") else (AQUA if m in ("Sparsity line + check", "Round median + check", "Sparsity line") else GRAY) for m in s.index]
        mean, lo, hi = s[f"{col}_mean"] * mult, s[f"{col}_p05"] * mult, s[f"{col}_p95"] * mult
        ax.hlines(y, lo, hi, color=colr, lw=2, alpha=0.5)
        ax.scatter(mean, y, color=colr, s=28, zorder=3, edgecolor="white", lw=0.6)
        ax.set_title(title, loc="left", fontsize=8.5)
        if col == "saving_vs_tmin_check":
            ax.axvline(0, color=MUTED, lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(s.index, fontsize=7.5)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.text(0.01, -0.02, "Held-out seeds: 56 splits, 5 training runs / 5 unseen test runs. Dot = mean, bar = 5-95% of splits.",
             fontsize=8, color=MUTED, ha="left")
    save(fig, "fig3_heldout_comparison")


def fig_pred_actual(rounds):
    p = pd.read_csv(os.path.join(OUT, "split_predictions_strict.csv"))
    p = p.merge(rounds[["run", "round", "t_star", "round_status"]], on=["run", "round"])
    p = p[p["round_status"] == "searched"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, m in zip(axes, ["Formula (margin 0.1 pp)", "Sparsity line + check", "Loss match (train) + check"]):
        grid(ax)
        d = p[p["method"] == m].groupby(["run", "round"]).agg(t_star=("t_star", "first"), mean=("t_hat", "mean"),
                                                              lo=("t_hat", "min"), hi=("t_hat", "max")).reset_index()
        late = d["mean"] > d["t_star"]
        ax.plot([0, 80], [0, 80], color=MUTED, lw=0.8, ls="--")
        ax.errorbar(d["t_star"], d["mean"], yerr=[d["mean"] - d["lo"], d["hi"] - d["mean"]], fmt="none",
                    ecolor=GRID, elinewidth=1.2, zorder=1)
        ax.scatter(d.loc[~late, "t_star"], d.loc[~late, "mean"], s=16, color=BLUE, zorder=3, label="early / exact")
        ax.scatter(d.loc[late, "t_star"], d.loc[late, "mean"], s=16, color=ORANGE, zorder=3, label="too late")
        mae = (p.loc[p["method"] == m, "t_hat"] - p.loc[p["method"] == m, "t_star"]).abs().mean()
        ax.set_title(f"{m}\nMAE {mae:.1f} (all split x round pairs)", loc="left", fontsize=8.5)
        ax.set_xlabel("measured t*")
    axes[0].set_ylabel("predicted t_hat (mean over splits;\nbar = min-max)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig4_predicted_vs_actual")


def fig_kappa_rho():
    sl = pd.read_csv(os.path.join(OUT, "model_checks_round_slopes.csv"))
    cr = pd.read_csv(os.path.join(OUT, "model_checks_slope_vs_rho.csv")).set_index("slope")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(wspace=0.35))
    for ax, col, lab in [(axes[0], "kappa_train_acc", "train-acc slope (pp per e-fold)"),
                         (axes[1], "kappa_val", "val-acc slope (pp per e-fold)")]:
        grid(ax)
        sc = ax.scatter(sl["weight_perturbation" if "weight_perturbation" in sl else "rho"], sl[col],
                        c=sl["sparsity"] * 100, cmap="viridis", s=20, edgecolor="white", lw=0.4)
        x = sl["rho"]
        w = np.polyfit(x, sl[col], 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, np.polyval(w, xs), color=INK, lw=1.2)
        c = cr.loc[col]
        ax.set_title(f"r = {c['r_rho']:.2f} {c['r_rho_ci95']}\npartial r | sparsity = "
                     f"{c['partial_r_rho_given_sparsity']:.2f} {c['partial_ci95']}", loc="left", fontsize=8)
        ax.set_xlabel("rho = |removed weights| / |W|")
        ax.set_ylabel(lab)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.03)
    cb.set_label("sparsity (%)")
    save(fig, "fig5_kappa_vs_rho")


def fig_capacity():
    cap = pd.read_csv(os.path.join(OUT, "model_checks_capacity_per_round.csv"))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    grid(ax)
    for run, d in cap.groupby("run"):
        ax.plot(d["sparsity"] * 100, d["cap"], color=ORANGE if run.endswith("_orig") else BLUE, lw=0.8,
                alpha=0.5, marker="o", ms=3)
    x = np.linspace(cap["ln_inv_d"].min(), cap["ln_inv_d"].max(), 50)
    lin = np.polyfit(cap["ln_inv_d"], cap["cap"], 1)
    quad = np.polyfit(cap["ln_inv_d"], cap["cap"], 2)
    s = (1 - np.exp(-x)) * 100
    ax.plot(s, np.polyval(lin, x), color=INK, lw=1.6, label="linear in ln(1/d)")
    ax.plot(s, np.polyval(quad, x), color=INK, lw=1.2, ls="--", label="quadratic in ln(1/d)")
    ax.axhline(-0.5, color=MUTED, lw=0.8, ls=":")
    ax.text(46, -0.47, "target (dense - 0.5 pp)", fontsize=7.5, color=MUTED)
    ax.set_xlabel("sparsity (%)")
    ax.set_ylabel("capacity: A(t_min) - A_dense (pp)")
    ax.set_title("Per-round capacity, one line per run", loc="left")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig6_capacity")


def fig_tradeoff():
    sw = pd.read_csv(os.path.join(OUT, "model_checks_tradeoff_sweep.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw=dict(wspace=0.38))
    styles = {"Formula": (BLUE, "o", "formula, margin 0 to 0.4 pp"),
              "Sparsity line + check": (AQUA, "s", "sparsity line + check, 0-20 epochs earlier"),
              "Round median + check": (YELLOW, "^", "round median + check, 0-20 epochs earlier")}
    cols = [("saving_vs_tmin_check", "retraining saved (%)", "(a) all rounds", 100),
            ("saving_recoverable_rounds_only", "retraining saved (%)", "(b) recoverable rounds only", 100),
            ("mae", "t* MAE (epochs)", "(c) error", 1)]
    for ax, (col, ylab, title, mult) in zip(axes, cols):
        grid(ax)
        for m, (c, mk, lab) in styles.items():
            d = sw[sw["method"] == m].sort_values("setting")
            ax.plot(d["too_late_rate"] * 100, d[col] * mult, color=c, marker=mk, ms=3.5, lw=1.3, label=lab)
            d0 = d.iloc[0]
            ax.scatter([d0["too_late_rate"] * 100], [d0[col] * mult], s=55, facecolor="none", edgecolor=c, lw=1.1)
        ax.set_xlabel("too-late rate (%)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=9)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    fig.suptitle("Safety/cost trade-off on held-out seeds (56 splits; open circle = default setting)",
                 x=0.01, ha="left", fontsize=9, color=INK, y=1.02)
    save(fig, "fig7_tradeoff")


def main():
    man = load_manifest()
    rounds, trials, runs = build_tables(man)
    fig_tstar(rounds, trials, runs)
    fig_budget(rounds, trials, runs)
    fig_heldout()
    fig_pred_actual(rounds)
    fig_kappa_rho()
    fig_capacity()
    fig_tradeoff()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
