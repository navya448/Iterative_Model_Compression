"""Predictors of the rewind point t* used by 03_formula_eval.py and 04_model_checks.py.

Every predictor has
    fit(train_rounds, train_trials)            -> self   (training seeds only)
    predict(row, run)                          -> (t_hat, flag)
where `row` is one IMP round (a rounds.csv row of a test seed) and `run` holds that
seed's constants (B table, T, target, tol, t_min). Flags:
    "no_retrain"  pruned model already meets the target, t_hat = T
    "stop"        predictor says no rewind point will recover (t_hat = t_min, the
                  cheapest last attempt before stopping the chain)
    "ok"          ordinary prediction
Nothing in predict() looks at t*, the trials or any other result of the round.
"""

import numpy as np
import pandas as pd

from common_eval import b_inverse


# ----------------------------------------------------------------- formula family

class Formula:
    """A(t) - A_dense = capacity(d) - kappa * ln(B(t_min)/B(t))

    cap   : "lninvd"  a0 - a1*ln(1/d)                      (proposed)
            "quad"    a0 - a1*ln(1/d) - a2*ln(1/d)^2
            "inv_d"   a0 - a1*(1/d)
            "round"   one free intercept per round index (most flexible)
    kappa : "const"   kappa                                (proposed)
            "rho"     kappa0 + kappa1*rho   (rho = |removed| / |W|, known right after pruning)
    margin: safety margin in pp subtracted from the slack before solving for t
    ref   : "dense"     model A(t) - A_dense (proposed; capacity relative to this seed's
                        dense val acc, which itself carries ~0.4 pp measurement noise)
            "absolute"  model A(t) directly; the recovery target is unchanged
    """

    def __init__(self, name, cap="lninvd", kappa="const", margin=0.0, ref="dense"):
        self.name, self.cap, self.kappa, self.margin, self.ref = name, cap, kappa, margin, ref
        self.rounds_seen = None

    def _y(self, df):
        return (df["y"] if self.ref == "dense" else df["final_val_acc"]).to_numpy(float)

    # design matrix --------------------------------------------------------------
    def _cap_cols(self, df):
        x = df["ln_inv_d"].to_numpy(float)
        if self.cap == "lninvd":
            return [np.ones(len(df)), -x], ["a0", "a1"]
        if self.cap == "quad":
            return [np.ones(len(df)), -x, -x ** 2], ["a0", "a1", "a2"]
        if self.cap == "inv_d":
            return [np.ones(len(df)), -np.exp(x)], ["a0", "a1_inv_d"]
        if self.cap == "round":
            k = df["round"].to_numpy(int)
            if self.rounds_seen is not None:
                seen = np.array(self.rounds_seen)
                k = np.array([seen[np.argmin(np.abs(seen - v))] for v in k])
            return [(k == r).astype(float) for r in self.rounds_seen], [f"cap_r{r}" for r in self.rounds_seen]
        raise ValueError(self.cap)

    def _kappa_cols(self, df):
        c = df["clock"].to_numpy(float)
        if self.kappa == "const":
            return [-c], ["kappa"]
        if self.kappa == "rho":
            rho = df["weight_perturbation"].to_numpy(float)
            return [-c, -rho * c], ["kappa0", "kappa1_rho"]
        raise ValueError(self.kappa)

    def design(self, df):
        a, na = self._cap_cols(df)
        b, nb = self._kappa_cols(df)
        return np.column_stack(a + b), na + nb

    # fit / predict -----------------------------------------------------------------
    def fit(self, rounds, trials):
        if self.cap == "round":
            self.rounds_seen = sorted(trials["round"].unique().tolist())
        X, names = self.design(trials)
        y = self._y(trials)
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)   # kappa(rho) needs rho; the
        X, y = X[keep], y[keep]                                # interrupted round has none
        self.n_fit = int(keep.sum())
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.coef = dict(zip(names, coef))
        self._w = coef
        resid = X @ coef - y
        self.train_rmse = float(np.sqrt(np.mean(resid ** 2)))
        return self

    def yhat(self, df):
        X, _ = self.design(df)
        return X @ self._w

    def kappa_eff(self, rho):
        if self.kappa == "const":
            return self.coef["kappa"]
        return self.coef["kappa0"] + self.coef["kappa1_rho"] * rho

    def capacity(self, row):
        """Predicted A(t_min) - A_dense for this round (the clock term is 0 at t_min)."""
        c, x = self.coef, float(row["ln_inv_d"])
        if self.cap == "lninvd":
            return c["a0"] - c["a1"] * x
        if self.cap == "quad":
            return c["a0"] - c["a1"] * x - c["a2"] * x * x
        if self.cap == "inv_d":
            return c["a0"] - c["a1_inv_d"] * np.exp(x)
        if self.cap == "round":
            seen = np.array(self.rounds_seen)
            k = int(seen[np.argmin(np.abs(seen - int(row["round"])))])
            return c[f"cap_r{k}"]
        raise ValueError(self.cap)

    def predict(self, row, run):
        """Latest t with predicted A(t) >= target + margin, i.e.
        capacity - kappa*ln(B(t_min)/B(t)) >= -tol + margin
        <=> B(t) >= B(t_min) * exp(-(capacity + tol - margin) / kappa)   (B^-1 below)."""
        T, t_min, tol = run["T"], run["t_min"], run["tol"]
        if row["after_val_acc"] >= row["target_val_acc"]:
            return T, "no_retrain"
        if self.ref == "dense":
            S = self.capacity(row) + tol - self.margin      # slack in pp
        else:
            S = self.capacity(row) - row["target_val_acc"] - self.margin
        k = self.kappa_eff(row["weight_perturbation"])
        if S <= 0 or k <= 0:
            return t_min, "stop"
        B = run["B"]
        b_req = B[t_min] * np.exp(-S / k)
        return b_inverse(B[:T], b_req, t_min), "ok"      # t = T (no retraining) excluded

    def predict_bruteforce(self, row, run):
        """Independent check of predict(): evaluate the fitted A(t) on every t and take the
        latest one that reaches the target (no closed form, no B^-1)."""
        T, t_min, tol = run["T"], run["t_min"], run["tol"]
        if row["after_val_acc"] >= row["target_val_acc"]:
            return T, "no_retrain"
        B = run["B"]
        ts = np.arange(t_min, T)
        grid = pd.DataFrame(dict(t=ts, round=int(row["round"]), ln_inv_d=row["ln_inv_d"],
                                 weight_perturbation=row["weight_perturbation"],
                                 clock=np.log(B[t_min] / B[ts])))
        need = (-tol if self.ref == "dense" else row["target_val_acc"]) + self.margin
        ok = ts[self.yhat(grid) >= need - 1e-12]
        if len(ok) == 0:
            return t_min, "stop"
        return int(ok.max()), "ok"


# ----------------------------------------------------------------- baselines

class RuleColumn:
    """One of the original candidate rules; its prediction was recorded in rounds.csv
    before the search (run_imp.py step 4). Optionally preceded by the no-retraining check."""

    def __init__(self, name, col, check=False):
        self.name, self.col, self.check = name, col, check

    def fit(self, rounds, trials):
        return self

    def predict(self, row, run):
        if self.check and row["after_val_acc"] >= row["target_val_acc"]:
            return run["T"], "no_retrain"
        v = row[self.col]
        if pd.isna(v):
            return np.nan, "missing"
        return int(v), "ok"


class TMin:
    """Always rewind to t_min (the safest choice), optionally after the check."""

    def __init__(self, name, check=False):
        self.name, self.check = name, check

    def fit(self, rounds, trials):
        return self

    def predict(self, row, run):
        if self.check and row["after_val_acc"] >= row["target_val_acc"]:
            return run["T"], "no_retrain"
        return run["t_min"], "ok"


class SparsityLine:
    """Least-squares line t* = b0 + b1*s on the training seeds (mid-term report baseline).
    Without the check it is fitted on all rounds with a measured t* (as in the report);
    with the check it is fitted on the rounds that needed retraining."""

    def __init__(self, name, check=False):
        self.name, self.check = name, check

    def fit(self, rounds, trials):
        r = rounds[rounds["t_star"].notna()]
        if self.check:
            r = r[r["needs_retrain"]]
        self.b1, self.b0 = np.polyfit(r["sparsity"], r["t_star"], 1)
        return self

    def predict(self, row, run):
        if self.check and row["after_val_acc"] >= row["target_val_acc"]:
            return run["T"], "no_retrain"
        return int(np.clip(round(self.b0 + self.b1 * row["sparsity"]), run["t_min"], run["T"])), "ok"


class RoundMedian:
    """Median t* of the training seeds at the same round index (rounds that needed
    retraining), after the check. Nearest round index if the index was never seen."""

    def __init__(self, name):
        self.name = name

    def fit(self, rounds, trials):
        r = rounds[rounds["t_star"].notna() & rounds["needs_retrain"]]
        self.med = r.groupby("round")["t_star"].median().to_dict()
        return self

    def predict(self, row, run):
        if row["after_val_acc"] >= row["target_val_acc"]:
            return run["T"], "no_retrain"
        ks = np.array(sorted(self.med))
        k = ks[np.argmin(np.abs(ks - int(row["round"])))]
        return int(np.floor(self.med[k])), "ok"


ORIGINAL_RULES = {
    "pred_loss_match_val": "Loss match (val)",
    "pred_loss_match_train": "Loss match (train)",
    "pred_loss_match_val_vs_dense": "Loss match (val, vs dense)",
    "pred_acc_match_val": "Accuracy match (val)",
    "pred_movement_match": "Weight-movement match",
    "pred_specialization": "Specialization point",
    "pred_baseline_5pct": "Fixed: 5% of T",
    "pred_baseline_before_first_lr_drop": "Fixed: before 1st LR drop",
    "pred_baseline_before_last_lr_drop": "Fixed: before last LR drop",
}


def main_methods():
    """Methods compared in the held-out evaluation. The formula variants with margin
    0 and 0.1 pp were both specified on the discovery seeds before the new seeds existed."""
    m = []
    for col, label in ORIGINAL_RULES.items():
        m.append(RuleColumn(label, col, check=False))
        m.append(RuleColumn(label + " + check", col, check=True))
    m += [TMin("Always t_min"), TMin("t_min + check", check=True),
          SparsityLine("Sparsity line"), SparsityLine("Sparsity line + check", check=True),
          RoundMedian("Round median + check"),
          Formula("Formula (margin 0)", margin=0.0),
          Formula("Formula (margin 0.1 pp)", margin=0.1)]
    return m


def variant_methods():
    """Model variants for step 6 (kappa(rho), capacity form)."""
    return [Formula("F: ln(1/d), const kappa", "lninvd", "const"),
            Formula("F: ln(1/d), kappa(rho)", "lninvd", "rho"),
            Formula("F: quadratic ln(1/d), const kappa", "quad", "const"),
            Formula("F: 1/d, const kappa", "inv_d", "const"),
            Formula("F: per-round capacity, const kappa", "round", "const"),
            Formula("F: quadratic ln(1/d), kappa(rho)", "quad", "rho"),
            Formula("F: ln(1/d), const kappa, absolute acc", "lninvd", "const", ref="absolute")]
