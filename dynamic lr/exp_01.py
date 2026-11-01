"""Experiment 01: IMP with dynamic-LR retraining instead of rewinding.

Same setup as imp_rewind (ResNet-20 / CIFAR-10, 45k train / 5k val / 10k test,
global magnitude pruning of 20% of the surviving conv weights per round), but
after each pruning step the model is NOT rewound. It is retrained from where it
is, and the learning rate is adjusted on the fly by watching how fast the loss
and accuracy are still improving.

Steps:
  1. train a dense ResNet-20 (T = 80 epochs, SGD 0.1, /10 at epochs 40 and 60)
  2. for every IMP round:
       a. prune 20% of the surviving conv weights (global magnitude)
       b. recalibrate BatchNorm, measure the damage
       c. retrain with the dynamic LR controller until the accuracy is recovered
       d. continue the chain from the retrained model

Dynamic LR controller (one run per round):
  - fresh SGD, `--warmup` epochs of linear warmup up to `--lr0`
  - every epoch: EMA-smoothed val loss and val acc, and their improvement rates
    over the last `--window` epochs of the current LR stage
        loss_rate = (ema_loss[-window] - ema_loss[now]) / window   (> 0 = improving)
        acc_rate  = (ema_acc[now] - ema_acc[-window]) / window     (> 0 = improving)
  - plateau: loss_rate < --loss_eps AND acc_rate < --acc_eps for `--patience`
    epochs in a row (or the stage is `--max_stage_epochs` long)  ->  LR *= --lr_factor
  - stop when
        recovered:          val acc >= dense val acc - tol for `--confirm` epochs in a row
        plateau_at_min_lr:  plateau and the next LR would be below --lr_min
        max_epochs:         --max_epochs retraining epochs used
    If not recovered, the best-val-acc model of the round is kept.

Outputs (in <run_dir>/):
  dense/dense_log.csv, dense/final_model.pt, dense/final_metrics.json
  imp/rounds.csv                 one row per round (damage, epochs used, LR schedule, final metrics)
  imp/round_XX/curve.csv         per-epoch retraining curve with LR, EMAs, rates and events
  imp/masks/round_XX.pt
Both stages resume automatically if interrupted.

Usage:
  python exp_01.py --seed 1
  python exp_01.py --seed 99 --dense_epochs 2 --rounds 2 --max_epochs 4 --run_dir runs/smoke
"""

import argparse
import copy
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "imp_rewind"))

from common import (DEFAULTS, CIFARData, ResNet20, apply_mask, append_csv, evaluate,  # noqa: E402
                    evaluate_all, flat_prunable, full_mask, get_device,
                    global_magnitude_prune, load_json, lr_at_epoch, make_optimizer,
                    prunable_params, recalibrate_bn, save_json, set_seed, sparsity,
                    train_one_epoch, write_csv)


# --------------------------------------------------------------------------
# 1. Dense training
# --------------------------------------------------------------------------

def train_dense(args, cfg, data, device, dense_dir):
    """Standard dense run. Resumes from dense/last.pt; skipped if already finished."""
    final_file = os.path.join(dense_dir, "final_model.pt")
    metrics_file = os.path.join(dense_dir, "final_metrics.json")
    model = ResNet20().to(device)
    if os.path.exists(final_file) and os.path.exists(metrics_file):
        model.load_state_dict(torch.load(final_file, map_location=device))
        print("Dense model already trained, loading it.")
        return model, load_json(metrics_file)

    T = cfg["epochs"]
    opt = make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(args.seed)
    last_file = os.path.join(dense_dir, "last.pt")
    log_file = os.path.join(dense_dir, "dense_log.csv")
    start = 0
    if os.path.exists(last_file):
        ckpt = torch.load(last_file, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        gen.set_state(ckpt["gen"])
        start = ckpt["epoch"]
        print(f"Resuming dense training from epoch {start}.")
    elif os.path.exists(log_file):
        os.remove(log_file)

    for epoch in range(start, T):
        run_loss, run_acc, gnorm = train_one_epoch(model, data, opt, epoch, gen, None, cfg)
        vl, va = evaluate(model, data.x_val, data.y_val)
        append_csv(log_file, dict(epoch=epoch + 1, lr=lr_at_epoch(epoch, cfg),
                                  run_train_loss=run_loss, run_train_acc=run_acc,
                                  val_loss=vl, val_acc=va, grad_norm=gnorm))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        gen=gen.get_state(), epoch=epoch + 1), last_file)
        print(f"  dense epoch {epoch + 1:3d}/{T} | lr {lr_at_epoch(epoch, cfg):.4f} | "
              f"train loss {run_loss:.4f} | val acc {va:.2f}%")

    final = dict(evaluate_all(model, data), epochs=T)
    torch.save(model.state_dict(), final_file)
    save_json(metrics_file, final)
    os.remove(last_file)
    return model, final


# --------------------------------------------------------------------------
# 2. Dynamic-LR retraining
# --------------------------------------------------------------------------

def retrain_dynamic(args, cfg, data, model, mask, rnd, target, rdir):
    """Retrain the pruned `model` in place with the dynamic LR controller.
    Returns (model, info dict)."""
    opt = make_optimizer(model, cfg)   # fresh optimizer, momentum restarts
    gen = torch.Generator().manual_seed(args.seed * 100_000 + rnd * 1000)
    curve_path = os.path.join(rdir, "curve.csv")
    if os.path.exists(curve_path):
        os.remove(curve_path)

    lr = args.lr0
    ema_loss = ema_acc = None
    hist = []                 # (ema_loss, ema_acc) of the current LR stage
    stage_lrs, stage_epochs = [lr], [0]
    stalled_for = hits = 0
    best_acc, best_state = -1.0, None
    stop_reason = "max_epochs"

    for ep in range(args.max_epochs):
        warming = ep < args.warmup
        cur_lr = args.lr0 * (ep + 1) / (args.warmup + 1) if warming else lr
        ecfg = dict(cfg, lr=cur_lr, milestones=[])     # constant LR for this epoch
        run_loss, run_acc, gnorm = train_one_epoch(model, data, opt, ep, gen, mask, ecfg)
        vl, va = evaluate(model, data.x_val, data.y_val)
        tl, ta = evaluate(model, data.x_train_eval, data.y_train_eval)

        a = args.ema
        ema_loss = vl if ema_loss is None else a * vl + (1 - a) * ema_loss
        ema_acc = va if ema_acc is None else a * va + (1 - a) * ema_acc
        if va > best_acc:
            best_acc, best_state = va, copy.deepcopy(model.state_dict())
        hits = hits + 1 if va >= target else 0

        loss_rate = acc_rate = math.nan
        if not warming:
            hist.append((ema_loss, ema_acc))
            stage_epochs[-1] += 1
            if len(hist) > args.window:
                l0, a0 = hist[-1 - args.window]
                loss_rate = (l0 - ema_loss) / args.window
                acc_rate = (ema_acc - a0) / args.window
                stalled = loss_rate < args.loss_eps and acc_rate < args.acc_eps
                stalled_for = stalled_for + 1 if stalled else 0

        event, stop = "", False
        if hits >= args.confirm:
            event, stop, stop_reason = "recovered", True, "recovered"
        elif not warming and (stalled_for >= args.patience
                              or stage_epochs[-1] >= args.max_stage_epochs):
            new_lr = lr * args.lr_factor
            if new_lr < args.lr_min * (1 - 1e-9):
                event, stop, stop_reason = "plateau_at_min_lr", True, "plateau_at_min_lr"
            else:
                event = f"lr {lr:g} -> {new_lr:g}"
                lr = new_lr
                hist, stalled_for = [], 0
                stage_lrs.append(lr)
                stage_epochs.append(0)

        append_csv(curve_path, dict(epoch=ep + 1, lr=cur_lr, run_train_loss=run_loss,
                                    run_train_acc=run_acc, val_loss=vl, val_acc=va,
                                    train_loss=tl, train_acc=ta, grad_norm=gnorm,
                                    ema_val_loss=ema_loss, ema_val_acc=ema_acc,
                                    loss_rate=loss_rate, acc_rate=acc_rate, event=event))
        print(f"    round {rnd} | ep {ep + 1:3d} | lr {cur_lr:.4f} | val loss {vl:.4f} "
              f"| val acc {va:.2f}% (target {target:.2f}) {event}")
        if stop:
            break

    epochs_used = ep + 1
    if stop_reason != "recovered":
        model.load_state_dict(best_state)
    schedule = ",".join(f"{l:g}x{n}" for l, n in zip(stage_lrs, stage_epochs) if n > 0)
    info = dict(retrain_epochs=epochs_used, warmup_epochs=min(args.warmup, epochs_used),
                lr_schedule=schedule, n_lr_drops=len(stage_lrs) - 1,
                stop_reason=stop_reason, best_val_acc_seen=best_acc)
    return model, info


# --------------------------------------------------------------------------
# 3. IMP chain
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run_dir", default=None, help="default: <script dir>/runs/seed<seed>")
    p.add_argument("--data_dir", default=os.path.join(HERE, "..", "imp_rewind", "data"))
    p.add_argument("--dense_epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--prune_frac", type=float, default=0.2)
    p.add_argument("--tol", type=float, default=0.5,
                   help="recovered if val acc >= dense val acc - tol (%% points)")
    p.add_argument("--bn_batches", type=int, default=100)
    p.add_argument("--stop_when_unrecovered", action="store_true",
                   help="stop the chain at the first round that does not recover")
    # controller
    p.add_argument("--lr0", type=float, default=0.1, help="LR the retraining starts at")
    p.add_argument("--lr_factor", type=float, default=0.1, help="LR multiplier on plateau")
    p.add_argument("--lr_min", type=float, default=1e-3, help="lowest LR stage")
    p.add_argument("--warmup", type=int, default=2, help="linear warmup epochs up to lr0")
    p.add_argument("--ema", type=float, default=0.5, help="EMA weight of the newest epoch")
    p.add_argument("--window", type=int, default=3, help="epochs over which rates are measured")
    p.add_argument("--loss_eps", type=float, default=0.005,
                   help="val loss improvement per epoch below which loss counts as stalled")
    p.add_argument("--acc_eps", type=float, default=0.1,
                   help="val acc improvement (%% points / epoch) below which acc counts as stalled")
    p.add_argument("--patience", type=int, default=2,
                   help="stalled epochs in a row before the LR is lowered")
    p.add_argument("--max_stage_epochs", type=int, default=30)
    p.add_argument("--confirm", type=int, default=2,
                   help="epochs in a row at or above target to count as recovered")
    p.add_argument("--max_epochs", type=int, default=80, help="retraining budget per round")
    args = p.parse_args()

    run_dir = args.run_dir or os.path.join(HERE, "runs", f"seed{args.seed}")
    dense_dir = os.path.join(run_dir, "dense")
    imp_dir = os.path.join(run_dir, "imp")
    os.makedirs(dense_dir, exist_ok=True)
    os.makedirs(os.path.join(imp_dir, "masks"), exist_ok=True)

    cfg = dict(DEFAULTS, epochs=args.dense_epochs)
    device = get_device()
    set_seed(args.seed)
    data = CIFARData(args.data_dir, device, cfg)

    # ---- dense ---------------------------------------------------------------
    model, dense_final = train_dense(args, cfg, data, device, dense_dir)
    target = dense_final["val_acc"] - args.tol
    save_json(os.path.join(run_dir, "config.json"),
              dict(vars(args), run_dir=run_dir, **cfg, target_val_acc=target))
    print(f"Dense val acc {dense_final['val_acc']:.2f}% -> recovery target {target:.2f}%")

    # ---- IMP rounds ------------------------------------------------------------
    mask = full_mask(model)
    summaries = []
    for rnd in range(1, args.rounds + 1):
        rdir = os.path.join(imp_dir, f"round_{rnd:02d}")
        os.makedirs(rdir, exist_ok=True)
        summary_file = os.path.join(rdir, "summary.json")
        final_file = os.path.join(rdir, "final_model.pt")
        mask_file = os.path.join(imp_dir, "masks", f"round_{rnd:02d}.pt")

        if os.path.exists(summary_file) and os.path.exists(final_file):   # resume
            s = load_json(summary_file)
            summaries.append(s)
            mask = torch.load(mask_file, map_location=device)
            model = ResNet20().to(device)
            model.load_state_dict(torch.load(final_file, map_location=device))
            print(f"Round {rnd}: already done ({s['retrain_epochs']} epochs), skipping.")
            if not s["recovered"] and args.stop_when_unrecovered:
                break
            continue

        # a. prune
        before = evaluate_all(model, data)
        new_mask = global_magnitude_prune(model, mask, args.prune_frac)
        torch.save(new_mask, mask_file)
        w_before = flat_prunable(model)
        removed = torch.cat([(w * (mask[n] - new_mask[n])).flatten()
                             for n, w in prunable_params(model).items()]).detach()
        apply_mask(model, new_mask)

        # b. recalibrate BN and measure damage
        recalibrate_bn(model, data, torch.Generator().manual_seed(args.seed * 7 + rnd),
                       args.bn_batches)
        after = evaluate_all(model, data)
        damage = dict(D_loss_val=after["val_loss"] - before["val_loss"],
                      D_acc_val=before["val_acc"] - after["val_acc"],
                      D_acc_val_vs_dense=dense_final["val_acc"] - after["val_acc"],
                      weight_perturbation=removed.norm().item() / w_before.norm().item())
        print(f"Round {rnd}: sparsity {sparsity(new_mask) * 100:.1f}% | "
              f"val acc {before['val_acc']:.2f} -> {after['val_acc']:.2f}% after pruning")

        # c. dynamic-LR retraining
        if after["val_acc"] >= target:
            info = dict(retrain_epochs=0, warmup_epochs=0, lr_schedule="", n_lr_drops=0,
                        stop_reason="no_retraining_needed", best_val_acc_seen=after["val_acc"])
        else:
            model, info = retrain_dynamic(args, cfg, data, model, new_mask, rnd, target, rdir)
        final = evaluate_all(model, data)
        recovered = final["val_acc"] >= target

        s = dict(round=rnd, sparsity=sparsity(new_mask),
                 **{f"before_{k}": v for k, v in before.items()},
                 **{f"after_{k}": v for k, v in after.items()},
                 **damage, **info, recovered=recovered, target_val_acc=target,
                 **{f"final_{k}": v for k, v in final.items()})
        torch.save(model.state_dict(), final_file)
        save_json(summary_file, s)
        summaries.append(s)
        write_csv(os.path.join(imp_dir, "rounds.csv"), summaries)
        print(f"Round {rnd}: {info['stop_reason']} after {info['retrain_epochs']} epochs "
              f"[{info['lr_schedule']}] -> val {final['val_acc']:.2f}% / test {final['test_acc']:.2f}%")

        # d. continue the chain from the retrained model
        mask = new_mask
        if not recovered and args.stop_when_unrecovered:
            print("Not recovered; stopping the chain.")
            break

    # ---- summary ---------------------------------------------------------------
    print("\nround  sparsity  after-prune  epochs  schedule                    val     test    recovered")
    for s in summaries:
        print(f"{s['round']:5d}  {s['sparsity'] * 100:7.1f}%  {s['after_val_acc']:10.2f}%  "
              f"{s['retrain_epochs']:6d}  {s['lr_schedule'] or '-':26s}  "
              f"{s['final_val_acc']:6.2f}  {s['final_test_acc']:6.2f}  {s['recovered']}")
    print(f"Total retraining epochs: {sum(s['retrain_epochs'] for s in summaries)}")
    print("Results in", imp_dir)


if __name__ == "__main__":
    main()
