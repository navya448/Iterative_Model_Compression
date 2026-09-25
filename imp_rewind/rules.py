"""Candidate rewind rules. Pure numpy/pandas (no PyTorch), so the analysis script
can use them anywhere, and run_imp.py can record each rule's prediction BEFORE
the search for t* runs (so the predictions can't be tuned to the answer).

Conventions: the dense log has one row per epoch 0..T, where row e is the model
after e epochs. A "rewind point" t means: take the dense weights at epoch t,
apply the mask, and train epochs t..T-1 (i.e. T - t retraining epochs).
"""

import numpy as np
import pandas as pd


def load_dense_log(path):
    df = pd.read_csv(path).drop_duplicates("epoch", keep="last")  # safe after a resume
    return df.sort_values("epoch").reset_index(drop=True)


def _latest(condition, epochs):
    """Largest epoch where `condition` holds; 0 if it never holds."""
    idx = np.where(condition)[0]
    return int(epochs[idx.max()]) if len(idx) else 0


def loss_match(dense, damage, col="val_loss"):
    """Latest t with L(t) - L(T) >= damage.
    'Rewind far enough back that the dense model still had to shed as much loss
    as pruning just added.'"""
    L = dense[col].to_numpy()
    return _latest(L - L[-1] >= damage, dense["epoch"].to_numpy())


def acc_match(dense, damage_acc, col="val_acc"):
    """Latest t with A(T) - A(t) >= accuracy damage (in % points)."""
    A = dense[col].to_numpy()
    return _latest(A[-1] - A >= damage_acc, dense["epoch"].to_numpy())


def movement_match(dense, perturbation_abs):
    """Latest t with S(t, T) = sum_{i=t+1..T} ||W_i - W_{i-1}|| >= ||dW_prune||.
    'Rewind far enough back that normal training still moves the weights by at
    least as much as pruning did.'"""
    wc = dense["weight_change"].fillna(0).to_numpy()
    # S[t] = sum of weight_change over epochs t+1..T
    S = np.concatenate([np.cumsum(wc[::-1])[::-1][1:], [0.0]])
    return _latest(S >= perturbation_abs, dense["epoch"].to_numpy())


def specialization_point(dense, k=3):
    """t_s = the last epoch before validation loss rises for k epochs in a row
    while training loss keeps falling. Returns None if that never happens
    (common for CIFAR ResNets with augmentation — that is itself a finding)."""
    vl = dense["val_loss"].to_numpy()
    tl = dense["train_loss"].to_numpy()
    ep = dense["epoch"].to_numpy()
    for i in range(1, len(vl) - k + 1):
        ok = all(vl[j] > vl[j - 1] and tl[j] < tl[j - 1] for j in range(i, i + k))
        if ok:
            return int(ep[i - 1])
    return None


def lr_remaining_fraction(dense, t):
    """Fraction of the total learning-rate 'budget' (sum of per-epoch LR) that
    is still ahead when rewinding to t. An alternative clock to epochs."""
    lr = dense["lr"].fillna(0).to_numpy()   # row e holds the LR of epoch e-1 -> e
    total = lr.sum()
    ahead = lr[dense["epoch"].to_numpy() > t].sum()
    return ahead / total if total > 0 else np.nan


def all_predictions(dense, damage, milestones=(40, 60)):
    """Every candidate rule's predicted rewind point for one IMP round.

    `damage` needs: D_loss_val, D_loss_train, D_acc_val, D_loss_val_vs_dense,
    removed_weight_norm.
    """
    T = int(dense["epoch"].max())
    ts = specialization_point(dense)
    return dict(
        pred_loss_match_val=loss_match(dense, damage["D_loss_val"], "val_loss"),
        pred_loss_match_train=loss_match(dense, damage["D_loss_train"], "train_loss"),
        pred_loss_match_val_vs_dense=loss_match(dense, damage["D_loss_val_vs_dense"], "val_loss"),
        pred_acc_match_val=acc_match(dense, damage["D_acc_val"], "val_acc"),
        pred_movement_match=movement_match(dense, damage["removed_weight_norm"]),
        pred_specialization=ts if ts is not None else np.nan,
        pred_baseline_5pct=int(round(0.05 * T)),
        pred_baseline_before_last_lr_drop=int(max(milestones)) - 1,
        pred_baseline_before_first_lr_drop=int(min(milestones)) - 1,
    )
