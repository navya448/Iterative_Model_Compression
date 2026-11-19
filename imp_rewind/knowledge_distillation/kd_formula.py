"""The rewind formula, in the old units (pp) and in normalized units.

Everything is written as a "badness" that grows with the rewind clock:

    bad(t) = cap + kappa * clock(t),      clock(t) = ln(B(t_min) / B(t))
    cap    = c0 + c_density * ln(1/x) + c_parent * parent
    kappa  = k0 + k_damage * damage

scale "pp" (the revised formula of data_order_experiments/formula_eval):
    bad    = A_dense - A(t)                    (pp of val accuracy)
    parent = A_dense - before_val_acc
    x      = d (global kept fraction), damage = ln(1 + D_acc)
    -> a0 = -c0, a1 = c_density, a2 = c_parent, k0, k1 = k_damage in report.md's signs

scale "ln" (normalized, meant to transfer across datasets):
    bad    = ln(err(t) / err_dense)            (e-folds of relative error)
    parent = ln(err_before / err_dense)
    i.e. err(t) = err_dense * e^c0 * x^(-c_density) * (err_before/err_dense)^c_parent
                  * (B(t_min)/B(t))^kappa     -- power laws in density and budget

Density x (all measured from the mask, no training):
    d_global  kept fraction of conv weights
    d_flops   kept fraction of conv multiply-accumulates (layers weighted by compute)
    d_min     density of the most-pruned layer
    d_lgeo    geometric mean of per-layer densities

Damage (measured right after pruning + BN recalibration, before retraining):
    acc     ln(1 + D_acc)                   D_acc = val acc lost, pp
    kl      ln(1 + D_kl / 0.01)             D_kl  = rise of KL(dense || model), nats
    relerr  max(ln(err_after/err_before), 0)

t_hat: the latest t whose predicted bad(t) is within the tolerance threshold,
    t_hat = B^-1( B(t_min) * exp(-(thr - cap) / kappa) )
T if the pruned model already meets the target; t_min ("stop") if thr <= cap.
"""

import numpy as np

KL_UNIT = 0.01

DENSITIES = ("d_global", "d_flops", "d_min", "d_lgeo")
DAMAGES = ("acc", "kl", "relerr")

# Variants compared. The first is the old revised formula, unchanged.
VARIANTS = [("pp", "d_global", "acc")] + [("ln", x, dm) for x in DENSITIES for dm in DAMAGES]
# Chosen before seeing any KD data, so the transfer test is not tuned to the result.
PRIMARY = ("ln", "d_flops", "kl")

PARAMS = ("c0", "c_density", "c_parent", "k0", "k_damage")

# Meta-model: each parameter is predicted from at most one setting descriptor, chosen
# in advance from what the parameter means (None = same value for every setting).
#   c_density ~ over-parameterization  z_overparam = ln(n_prunable / n_train)
# Every setting is CIFAR-10, so only the model changes; k0 (which should depend on the
# task's difficulty, z_difficulty = ln(err_dense / chance_err)) is shared. If other
# datasets are added later, set "k0": "z_difficulty".
META_DESCRIPTORS = {"c0": None, "c_density": "z_overparam", "c_parent": None,
                    "k0": None, "k_damage": None}


def variant_name(v):
    return "/".join(v)


def add_features(df, B_of_run):
    """Adds every feature column to a frame of trials (or rounds, without t).
    Needs columns run, dense_val_acc, before_val_acc, after_val_acc, target_val_acc,
    D_acc_val, D_kl_val, D_relerr_val, d_*; and t, final_val_acc for trials."""
    df = df.copy()
    A = df["dense_val_acc"].astype(float)
    err_T = (100 - A).clip(lower=1e-2)
    df["err_T"] = err_T
    for x in DENSITIES:
        df[f"lninv_{x}"] = -np.log(df[x].astype(float).clip(lower=1e-12))
    df["parent_pp"] = A - df["before_val_acc"]
    df["parent_ln"] = np.log((100 - df["before_val_acc"]).clip(lower=1e-2) / err_T)
    df["dmg_acc"] = np.log1p(df["D_acc_val"].clip(lower=0))
    df["dmg_kl"] = np.log1p(df["D_kl_val"].clip(lower=0) / KL_UNIT)
    df["dmg_relerr"] = df["D_relerr_val"].clip(lower=0)
    df["thr_pp"] = A - df["target_val_acc"]
    df["thr_ln"] = np.log((100 - df["target_val_acc"]) / err_T)
    if "t" in df and "final_val_acc" in df:
        df["bad_pp"] = A - df["final_val_acc"]
        df["bad_ln"] = np.log((100 - df["final_val_acc"]).clip(lower=1e-2) / err_T)
        df["clock"] = [np.log(B_of_run[r]["B"][B_of_run[r]["t_min"]] / B_of_run[r]["B"][int(t)])
                       for r, t in zip(df["run"], df["t"])]
    return df


def b_inverse(B, b_req, t_min, T):
    """Latest integer t in [t_min, T-1] with B[t] >= b_req."""
    ok = np.where(B[t_min:T] >= b_req - 1e-12)[0]
    return int(t_min + ok.max()) if len(ok) else t_min


class Formula:
    def __init__(self, variant):
        self.scale, self.density, self.damage = variant
        self.name = variant_name(variant)
        self.w = None

    # columns
    def _cap(self, df):
        return [np.ones(len(df)), df[f"lninv_{self.density}"].to_numpy(float),
                df[f"parent_{self.scale}"].to_numpy(float)]

    def _kap(self, df):
        return [np.ones(len(df)), df[f"dmg_{self.damage}"].to_numpy(float)]

    def design(self, df):
        clock = df["clock"].to_numpy(float)
        return np.column_stack(self._cap(df) + [clock * c for c in self._kap(df)])

    def fit(self, trials):
        X, y = self.design(trials), trials[f"bad_{self.scale}"].to_numpy(float)
        keep = np.isfinite(X).all(1) & np.isfinite(y)
        if keep.sum() < len(PARAMS):
            raise ValueError("not enough trials to fit")
        # Refuse fits whose data cannot separate the 5 parameters (e.g. a chain that
        # stopped after 2-3 rounds): lstsq would return arbitrary values for them.
        Xk = X[keep]
        rank = np.linalg.matrix_rank(Xk / np.maximum(np.abs(Xk).max(0), 1e-12), tol=1e-6)
        if rank < len(PARAMS):
            raise ValueError(f"parameters not identifiable (rank {rank} < {len(PARAMS)})")
        self.w, *_ = np.linalg.lstsq(Xk, y[keep], rcond=None)
        self.n_fit = int(keep.sum())
        return self

    def set_params(self, params):
        self.w = np.array([params[k] for k in PARAMS], dtype=float)
        return self

    @property
    def params(self):
        return dict(zip(PARAMS, self.w))

    # predictions
    def bad(self, df):
        return self.design(df) @ self.w

    def acc(self, df):
        b = self.bad(df)
        if self.scale == "pp":
            return df["dense_val_acc"].to_numpy(float) - b
        return 100 - df["err_T"].to_numpy(float) * np.exp(b)

    def cap_kappa(self, row):
        df = row.to_frame().T
        cap = float(np.column_stack(self._cap(df))[0] @ self.w[:3])
        kap = float(np.column_stack(self._kap(df))[0] @ self.w[3:])
        return cap, kap

    def predict_t(self, row, run, kappa=None):
        T, t_min, B = run["T"], run["t_min"], run["B"]
        if row["after_val_acc"] >= row["target_val_acc"]:
            return T
        cap, k = self.cap_kappa(row)
        if kappa is not None and kappa > 0:
            k = kappa
        S = row[f"thr_{self.scale}"] - cap
        if S <= 0 or k <= 0:
            return t_min
        return b_inverse(B, B[t_min] * np.exp(-S / k), t_min, T)

    def probe_kappa(self, row, probe):
        """kappa for one round from its probe trial: (bad_probe - cap) / clock_probe."""
        cap, _ = self.cap_kappa(row)
        if probe is None or not probe["clock"] > 0:
            return None
        k = (probe[f"bad_{self.scale}"] - cap) / probe["clock"]
        return k if k > 0 else None
