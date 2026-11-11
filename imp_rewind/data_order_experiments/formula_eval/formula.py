"""The rewind-point formula, original and revised.

Original (eval_10seed/models.py on branch eval-10seed, fit on the 10-seed runs):

    A(t) - A_dense = a0 - a1*ln(1/d) - kappa*ln(B(t_min)/B(t))

Revised after the data-order experiment (see report.md). Capacity also depends on
how far the parent model of the chain is below dense, and kappa grows with the
damage pruning just did:

    A(t) - A_dense = a0 - a1*ln(1/d) + a2*parent_def
                     - (k0 + k1*ln(1 + D_acc)) * ln(B(t_min)/B(t))

    parent_def = before_val_acc - A_dense   (<= 0; the chain model before this round's pruning)
    D_acc      = D_acc_val, clipped at 0    (val acc lost to pruning, after BN recalibration)

Both are measured right after pruning, before any retraining. t_hat is the latest t
whose predicted A(t) reaches the target:

    t_hat = B^-1( B(t_min) * exp(-S / kappa) ),   S = capacity + tol - margin

S <= 0 means no rewind point is predicted to recover ("stop", t_hat = t_min).
"""

import numpy as np

# term name -> column. Capacity terms are added to the capacity, kappa terms are
# added to kappa (so they multiply the clock).
CAP_TERMS = {"parent_def": "parent_def"}
KAPPA_TERMS = {"ln_damage": "ln_damage", "rho": "weight_perturbation"}


def budget_table(dense):
    """B[t] = sum of the per-epoch LR over epochs t..T-1, in units of the peak LR.
    Row e of dense_log.csv holds the LR of epoch e-1 -> e (as in eval_10seed)."""
    lr = dense["lr"].fillna(0).to_numpy(dtype=float)
    ep = dense["epoch"].to_numpy(dtype=int)
    T = int(ep.max())
    return np.array([lr[ep > t].sum() / lr.max() for t in range(T + 1)])


def b_inverse(B, b_req, t_min):
    """Latest integer t in [t_min, len(B)-1] with B[t] >= b_req."""
    if b_req <= 0:
        return len(B) - 1
    ok = np.where(B[t_min:] >= b_req - 1e-12)[0]
    return int(t_min + ok.max()) if len(ok) else t_min


def add_round_features(df):
    """Adds ln_inv_d, parent_def and ln_damage. Needs sparsity, before_val_acc,
    D_acc_val and a_dense (this run's dense val acc)."""
    df["ln_inv_d"] = -np.log(1 - df["sparsity"])
    df["parent_def"] = df["before_val_acc"] - df["a_dense"]
    df["ln_damage"] = np.log1p(df["D_acc_val"].clip(lower=0))
    return df


class Formula:
    """A(t) - A_dense = cap - kappa * clock, linear in its parameters.

    cap   = a0 - a1*ln(1/d) + sum of the `cap` terms
    kappa = k0 + sum of the `kappa` terms
    Formula() is the original; Formula(cap=("parent_def",), kappa=("ln_damage",))
    is the revised one."""

    def __init__(self, name="formula", cap=(), kappa=(), margin=0.0):
        self.name, self.cap, self.kappa, self.margin = name, tuple(cap), tuple(kappa), margin

    def param_names(self):
        return (["a0", "a1"] + [f"a_{c}" for c in self.cap]
                + ["k0"] + [f"k_{k}" for k in self.kappa])

    def _cap_cols(self, df):
        return [np.ones(len(df)), -df["ln_inv_d"].to_numpy(float)] + \
               [df[CAP_TERMS[c]].to_numpy(float) for c in self.cap]

    def _kappa_cols(self, df):
        return [np.ones(len(df))] + [df[KAPPA_TERMS[k]].to_numpy(float) for k in self.kappa]

    def design(self, df):
        clock = df["clock"].to_numpy(float)
        return np.column_stack(self._cap_cols(df) + [-clock * c for c in self._kappa_cols(df)])

    def fit(self, trials):
        X, y = self.design(trials), trials["y"].to_numpy(float)
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
        self.w, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        self.n_fit = int(keep.sum())
        self.train_rmse = float(np.sqrt(np.mean((X[keep] @ self.w - y[keep]) ** 2)))
        return self

    @property
    def coef(self):
        return dict(zip(self.param_names(), self.w))

    def yhat(self, df):
        return self.design(df) @ self.w

    def capacity_kappa(self, row):
        """Predicted A(t_min) - A_dense and the effective kappa for one round."""
        df = row.to_frame().T
        nc = 2 + len(self.cap)
        cap = float(np.column_stack(self._cap_cols(df))[0] @ self.w[:nc])
        kappa = float(np.column_stack(self._kappa_cols(df))[0] @ self.w[nc:])
        return cap, kappa

    def predict(self, row, run, kappa=None):
        """(t_hat, flag) for one round. `kappa` overrides the fitted kappa (used by
        the one-probe calibration)."""
        T, t_min, tol = run["T"], run["t_min"], run["tol"]
        if row["after_val_acc"] >= row["target_val_acc"]:
            return T, "no_retrain"
        cap, k = self.capacity_kappa(row)
        k = k if kappa is None else kappa
        S = cap + tol - self.margin
        if S <= 0 or k <= 0:
            return t_min, "stop"
        B = run["B"]
        return b_inverse(B[:T], B[t_min] * np.exp(-S / k), t_min), "ok"


def original(margin=0.0):
    return Formula("Original formula" + (f" (margin {margin})" if margin else ""), margin=margin)


def revised(margin=0.0):
    return Formula("Revised formula" + (f" (margin {margin})" if margin else ""),
                   cap=("parent_def",), kappa=("ln_damage",), margin=margin)
