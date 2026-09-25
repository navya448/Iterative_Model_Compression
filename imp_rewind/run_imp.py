"""Experiment B: IMP with a binary search for t* in every round.

Needs the dense run from train_dense.py (same --run_dir).

Each round k:
  1. prune 20% of the surviving conv weights of the current model (global magnitude)
  2. recalibrate BatchNorm (no weight updates)
  3. measure the damage caused by pruning
  4. record every candidate rule's predicted t  (BEFORE searching, so it can't be tuned)
  5. binary-search for t* = the latest rewind epoch whose retrained model reaches
     dense_val_acc - tol. Each try: dense weights at epoch t * mask, train t -> T.
  6. continue the IMP chain from the model retrained from t*

Outputs (in <run_dir>/imp/):
  rounds.csv           one row per round (damage, predictions, t*, ...)
  rewind_trials.csv    one row per rewind attempt
  masks/round_XX.pt    the mask of each round
  round_XX/            per-round cache: trial models, per-epoch curves, summary
The script resumes automatically: finished rounds and trials are not re-run.

Usage:
  python run_imp.py --seed 1 --run_dir runs/seed1 --data_dir data
"""

import argparse
import glob
import os

import pandas as pd
import torch

from common import (DEFAULTS, CIFARData, ResNet20, apply_mask, ckpt_path, evaluate,
                    evaluate_all, flat_prunable, full_mask, get_device,
                    global_magnitude_prune, load_json, make_optimizer, prunable_params,
                    recalibrate_bn, save_json, set_seed, sparsity, train_one_epoch,
                    write_csv, append_csv)
from rules import all_predictions, load_dense_log, lr_remaining_fraction


def load_dense_model(run_dir, epoch, device):
    model = ResNet20().to(device)
    model.load_state_dict(torch.load(ckpt_path(run_dir, epoch), map_location=device))
    return model


def run_trial(args, cfg, data, device, mask, rnd, t, rdir):
    """Rewind to dense epoch t, apply mask, retrain to T. Cached on disk."""
    tag = f"t{t:03d}"
    model_file = os.path.join(rdir, f"trial_{tag}.pt")
    meta_file = os.path.join(rdir, f"trial_{tag}.json")
    if os.path.exists(meta_file) and os.path.exists(model_file):
        model = ResNet20().to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        return model, load_json(meta_file)

    T = cfg["epochs"]
    model = load_dense_model(args.run_dir, t, device)
    apply_mask(model, mask)
    opt = make_optimizer(model, cfg)   # fresh optimizer; LR follows the schedule from epoch t
    gen = torch.Generator().manual_seed(args.seed * 100_000 + rnd * 1000 + t)
    curve_path = os.path.join(rdir, f"curve_{tag}.csv")
    if os.path.exists(curve_path):
        os.remove(curve_path)
    for epoch in range(t, T):
        run_loss, run_acc, gnorm = train_one_epoch(model, data, opt, epoch, gen, mask, cfg)
        vl, va = evaluate(model, data.x_val, data.y_val)
        tl, ta = evaluate(model, data.x_train_eval, data.y_train_eval)
        append_csv(curve_path, dict(epoch=epoch + 1, val_loss=vl, val_acc=va,
                                    train_loss=tl, train_acc=ta, grad_norm=gnorm))
    final = evaluate_all(model, data)
    meta = dict(round=rnd, t=t, retrain_epochs=T - t, **{f"final_{k}": v for k, v in final.items()})
    torch.save(model.state_dict(), model_file)
    save_json(meta_file, meta)
    print(f"    round {rnd} | t={t:3d} ({T - t:2d} epochs) -> val acc {final['val_acc']:.2f}%")
    return model, meta


def rebuild_trials_csv(imp_dir, target):
    rows = []
    for f in sorted(glob.glob(os.path.join(imp_dir, "round_*", "trial_*.json"))):
        m = load_json(f)
        m["target_val_acc"] = target
        m["recovered"] = int(m["final_val_acc"] >= target)
        rows.append(m)
    if rows:
        write_csv(os.path.join(imp_dir, "rewind_trials.csv"),
                  sorted(rows, key=lambda r: (r["round"], r["t"])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--prune_frac", type=float, default=0.2)
    p.add_argument("--tol", type=float, default=0.5,
                   help="recovered if final val acc >= dense val acc - tol (%% points)")
    p.add_argument("--resolution", type=int, default=2,
                   help="stop the binary search when the bracket is this many epochs wide")
    p.add_argument("--t_min", type=int, default=2,
                   help="earliest rewind point tried; if even this fails, the chain stops")
    p.add_argument("--bn_batches", type=int, default=100)
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    dense_dir = os.path.join(args.run_dir, "dense")
    dense_final = load_json(os.path.join(dense_dir, "final_metrics.json"))
    cfg["epochs"] = T = dense_final["epochs"]
    dense = load_dense_log(os.path.join(dense_dir, "dense_log.csv"))
    target = dense_final["val_acc"] - args.tol

    device = get_device()
    set_seed(args.seed)
    data = CIFARData(args.data_dir, device, cfg)
    imp_dir = os.path.join(args.run_dir, "imp")
    os.makedirs(os.path.join(imp_dir, "masks"), exist_ok=True)
    save_json(os.path.join(imp_dir, "config.json"), dict(vars(args), **cfg, target_val_acc=target))
    print(f"Dense val acc {dense_final['val_acc']:.2f}% -> recovery target {target:.2f}%")

    # round 0 = the dense model
    model = load_dense_model(args.run_dir, T, device)
    mask = full_mask(model)
    summaries = []

    for rnd in range(1, args.rounds + 1):
        rdir = os.path.join(imp_dir, f"round_{rnd:02d}")
        os.makedirs(rdir, exist_ok=True)
        summary_file = os.path.join(rdir, "summary.json")
        final_file = os.path.join(rdir, "final_model.pt")
        mask_file = os.path.join(imp_dir, "masks", f"round_{rnd:02d}.pt")

        if os.path.exists(summary_file):           # resume: round already finished
            s = load_json(summary_file)
            summaries.append(s)
            if s["t_star"] is None:
                print(f"Round {rnd}: previously found unrecoverable; stopping.")
                break
            mask = torch.load(mask_file, map_location=device)
            model = ResNet20().to(device)
            model.load_state_dict(torch.load(final_file, map_location=device))
            print(f"Round {rnd}: already done (t* = {s['t_star']}), skipping.")
            continue

        # ---- 1. prune -------------------------------------------------------
        before = evaluate_all(model, data)
        new_mask = global_magnitude_prune(model, mask, args.prune_frac)
        torch.save(new_mask, mask_file)
        w_before = flat_prunable(model)
        removed = torch.cat([(w * (mask[n] - new_mask[n])).flatten()
                             for n, w in prunable_params(model).items()]).detach()

        pruned = ResNet20().to(device)
        pruned.load_state_dict(model.state_dict())
        apply_mask(pruned, new_mask)
        after_raw = evaluate_all(pruned, data)          # stale BN stats
        # ---- 2. recalibrate BN ---------------------------------------------
        recalibrate_bn(pruned, data, torch.Generator().manual_seed(args.seed * 7 + rnd),
                       args.bn_batches)
        after = evaluate_all(pruned, data)               # the damage we use

        # ---- 3. damage -------------------------------------------------------
        dense_T = dense.iloc[-1]
        damage = dict(
            D_loss_val=after["val_loss"] - before["val_loss"],
            D_loss_train=after["train_loss"] - before["train_loss"],
            D_acc_val=before["val_acc"] - after["val_acc"],
            D_acc_train=before["train_acc"] - after["train_acc"],
            D_loss_val_vs_dense=after["val_loss"] - dense_T["val_loss"],
            D_acc_val_vs_dense=dense_T["val_acc"] - after["val_acc"],
            D_loss_val_no_bn_recal=after_raw["val_loss"] - before["val_loss"],
            D_acc_val_no_bn_recal=before["val_acc"] - after_raw["val_acc"],
            removed_weight_norm=removed.norm().item(),
            weight_perturbation=removed.norm().item() / w_before.norm().item(),
        )
        # ---- 4. predictions, fixed before the search -------------------------
        preds = all_predictions(dense, damage, cfg["milestones"])
        print(f"Round {rnd}: sparsity {sparsity(new_mask)*100:.1f}% | "
              f"D_loss_val {damage['D_loss_val']:.4f} | D_acc_val {damage['D_acc_val']:.2f} | "
              f"loss-match predicts t={preds['pred_loss_match_val']}")

        # ---- 5. binary search for t* -------------------------------------------
        t_star, best_model, n_trials = None, None, 0
        if after["val_acc"] >= target:
            # no retraining needed at all
            t_star, best_model = T, pruned
        else:
            m_lo, meta = run_trial(args, cfg, data, device, new_mask, rnd, args.t_min, rdir)
            n_trials += 1
            if meta["final_val_acc"] >= target:
                lo, hi, best_model = args.t_min, T, m_lo    # lo works, hi (= no retrain) fails
                while hi - lo > args.resolution:
                    mid = (lo + hi) // 2
                    m_mid, meta = run_trial(args, cfg, data, device, new_mask, rnd, mid, rdir)
                    n_trials += 1
                    if meta["final_val_acc"] >= target:
                        lo, best_model = mid, m_mid
                    else:
                        hi = mid
                t_star = lo
        rebuild_trials_csv(imp_dir, target)

        s = dict(round=rnd, sparsity=sparsity(new_mask),
                 **{f"before_{k}": v for k, v in before.items()},
                 **{f"after_{k}": v for k, v in after.items()},
                 **damage, **preds,
                 t_star=t_star,
                 retrain_epochs=(T - t_star) if t_star is not None else None,
                 t_star_frac=(t_star / T) if t_star is not None else None,
                 lr_remaining_frac=lr_remaining_fraction(dense, t_star) if t_star is not None else None,
                 n_trials=n_trials, target_val_acc=target)
        if t_star is not None:
            final = evaluate_all(best_model, data)
            s.update({f"final_{k}": v for k, v in final.items()})
            torch.save(best_model.state_dict(), final_file)
        save_json(summary_file, s)
        summaries.append(s)
        write_csv(os.path.join(imp_dir, "rounds.csv"), summaries)

        if t_star is None:
            print(f"Round {rnd}: even t={args.t_min} does not recover. Stopping the chain.")
            break
        print(f"Round {rnd}: t* = {t_star} ({T - t_star} retraining epochs, {n_trials} trials)")
        # ---- 6. continue the chain -------------------------------------------
        model, mask = best_model, new_mask

    if summaries:
        write_csv(os.path.join(imp_dir, "rounds.csv"), summaries)
    print("IMP done. Results in", imp_dir)


if __name__ == "__main__":
    main()
