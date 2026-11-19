"""One job of the KD experiments. run_all.py calls this; you normally don't need to.

  python kd_job.py prepare --dataset cifar100
  python kd_job.py dense   --dataset cifar10 --model resnet20 --seed 1
  python kd_job.py imp     --dataset cifar10 --model resnet20 --seed 1 --method kd

dense : train the dense model (= the teacher) and save a checkpoint every epoch.
imp   : IMP chain. Every round: prune 20% of the surviving conv weights, recalibrate
        BN, measure the damage, then binary-search t*, the latest rewind epoch whose
        retrain still recovers. Retraining uses
            ce        : cross-entropy only (the control, same as the old experiments)
            kd        : KD from the dense model at epoch T  (the professor's setup)
            kd_parent : KD from the chain model before this round's pruning
        The search runs t_min first (gives the capacity), then one short late "probe"
        rewind at probe_frac * T (gives kappa cheaply), then bisects the remaining bracket.

Both jobs resume where they stopped: finished epochs, trials and rounds are reused.
"""

import argparse
import copy
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kd_common import (DATASETS, Data, apply_mask, append_csv, build_model,   # noqa: E402
                       budget_table, conv_flops, dense_ckpt, density_measures, evaluate_all,
                       flat_prunable, full_mask, get_device, global_magnitude_prune, load_json,
                       load_raw, load_torch, lr_at_epoch, make_cfg, make_optimizer,
                       prunable_params, recalibrate_bn, save_json, save_torch, seed_dir,
                       set_seed, sparsity, teacher_entropy, teacher_reference, train_one_epoch,
                       truncate_csv, write_csv)

DEFAULT_DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
DEFAULT_RUNS = os.path.join(HERE, "runs")
METHODS = ("ce", "kd", "kd_parent")
KL_UNIT = 0.01   # nats; damage feature ln(1 + KL_rise / KL_UNIT), see kd_formula.py


def common_args(p):
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--data_dir", default=DEFAULT_DATA)
    p.add_argument("--runs_dir", default=DEFAULT_RUNS)
    p.add_argument("--train_limit", type=int, default=None,
                   help="use only this many training images (smoke tests only)")


# ==========================================================================
# prepare
# ==========================================================================

def cmd_prepare(args):
    xtr, ytr, xte, yte = load_raw(args.dataset, args.data_dir)
    print(f"{args.dataset}: train {xtr.shape}, test {xte.shape}, "
          f"classes {len(np.unique(ytr))}", flush=True)


# ==========================================================================
# dense
# ==========================================================================

def dense_fingerprint(args, cfg):
    return dict(dataset=args.dataset, model=args.model, seed=args.seed, epochs=cfg["epochs"],
                lr=cfg["lr"], train_limit=args.train_limit)


def cmd_dense(args):
    cfg = make_cfg(args.epochs, args.model)
    sd = seed_dir(args.runs_dir, args.dataset, args.model, args.seed)
    out = os.path.join(sd, "dense")
    final_path = os.path.join(out, "final_metrics.json")
    fp = dense_fingerprint(args, cfg)
    if os.path.exists(final_path):
        old = load_json(final_path).get("fingerprint")
        if old != fp:
            raise SystemExit(f"{final_path} was made with {old}, this job asks for {fp}. "
                             "Use another --runs_dir.")
        print(f"Dense run already finished: {final_path}", flush=True)
        return
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    log_path = os.path.join(out, "dense_log.csv")
    resume_path = os.path.join(out, "resume.pt")

    device = get_device()
    set_seed(args.seed)
    data = Data(args.dataset, args.data_dir, device, cfg, args.train_limit)
    model = build_model(args.model, data.num_classes).to(device)
    opt = make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(args.seed)

    start = 0
    if os.path.exists(resume_path):
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        if st.get("fingerprint") != fp:
            raise SystemExit(f"{resume_path} was made with {st.get('fingerprint')}, "
                             f"this job asks for {fp}. Use another --runs_dir.")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"])
        start = st["epoch"]
        truncate_csv(log_path, start)
        print(f"Resuming dense training from epoch {start}", flush=True)
    else:
        if os.path.exists(log_path):
            os.remove(log_path)
        save_torch(dense_ckpt(sd, 0), model.state_dict())
        m = evaluate_all(model, data)
        append_csv(log_path, dict(epoch=0, lr=float("nan"), run_train_loss=float("nan"),
                                  run_train_acc=float("nan"), **m, grad_norm=float("nan"),
                                  weight_change=float("nan"),
                                  weight_norm=flat_prunable(model).norm().item()))

    prev_w = flat_prunable(model).clone()
    for epoch in range(start, cfg["epochs"]):
        run_loss, run_acc, gnorm = train_one_epoch(model, data, opt, epoch, gen, cfg)
        e = epoch + 1
        w = flat_prunable(model)
        m = evaluate_all(model, data)
        append_csv(log_path, dict(epoch=e, lr=lr_at_epoch(epoch, cfg), run_train_loss=run_loss,
                                  run_train_acc=run_acc, **m, grad_norm=gnorm,
                                  weight_change=(w - prev_w).norm().item(),
                                  weight_norm=w.norm().item()))
        prev_w = w.clone()
        save_torch(dense_ckpt(sd, e), model.state_dict())
        save_torch(resume_path, dict(model=model.state_dict(), opt=opt.state_dict(),
                                     gen=gen.get_state(), epoch=e, fingerprint=fp))
        print(f"[dense {args.dataset}/{args.model}/s{args.seed}] epoch {e:3d}/{cfg['epochs']} | "
              f"lr {lr_at_epoch(epoch, cfg):.4f} | train {m['train_loss']:.3f}/{m['train_acc']:.2f}% "
              f"| val {m['val_loss']:.3f}/{m['val_acc']:.2f}%", flush=True)

    # The final model is the teacher. Record it and the setting's structure.
    final = evaluate_all(model, data)
    ref = teacher_reference(model, data)
    flops = conv_flops(model, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_prunable = sum(w.numel() for w in prunable_params(model).values())
    err = 100.0 - final["val_acc"]
    chance_err = 100.0 * (1 - 1 / data.num_classes)
    descriptors = dict(
        dataset=args.dataset, model=args.model, seed=args.seed,
        n_classes=data.num_classes, n_train=data.n_train,
        n_params=n_params, n_prunable=n_prunable, n_conv_layers=len(flops),
        conv_macs=float(sum(flops.values())),
        max_width=max(m.out_channels for m in model.modules() if isinstance(m, torch.nn.Conv2d)),
        dense_val_err=err, dense_test_err=100.0 - final["test_acc"], chance_err=chance_err,
        teacher_val_entropy=teacher_entropy(ref["val"]),
        teacher_val_nll=final["val_loss"],
        # pre-chosen meta-model descriptors (see kd_formula.META_DESCRIPTORS)
        z_overparam=float(np.log(n_prunable / data.n_train)),
        z_difficulty=float(np.log(err / chance_err)),
        z_log_depth=float(np.log(len(flops))),
    )
    save_json(os.path.join(out, "descriptors.json"), descriptors)
    save_json(final_path, dict(fingerprint=fp, epochs=cfg["epochs"], **final))
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print("Dense done:", {k: round(v, 3) for k, v in final.items()}, flush=True)


# ==========================================================================
# imp
# ==========================================================================

class Context:
    """Everything a trial needs."""


def imp_fingerprint(args, T):
    return dict(dataset=args.dataset, model=args.model, seed=args.seed, method=args.method,
                T=T, prune_frac=args.prune_frac, rel_tol=args.rel_tol, t_min=args.t_min,
                resolution=args.resolution, probe_frac=args.probe_frac,
                kd_T=args.kd_T if args.method != "ce" else None,
                kd_lambda=args.kd_lambda if args.method != "ce" else None,
                train_limit=args.train_limit)


def load_dense_epoch(ctx, epoch):
    m = build_model(ctx.args.model, ctx.data.num_classes).to(ctx.device)
    m.load_state_dict(load_torch(dense_ckpt(ctx.sd, epoch), ctx.device))
    return m


def run_trial(ctx, mask, rnd, t, rdir, teacher):
    """Rewind to dense epoch t, apply the mask, retrain to T (CE or KD). Returns the trial's
    metadata; weights go to trial_tXXX.pt. Cached on disk, and resumable inside the trial
    (a checkpoint is kept after every epoch)."""
    args, cfg, T = ctx.args, ctx.cfg, ctx.cfg["epochs"]
    tag = f"t{t:03d}"
    model_file = os.path.join(rdir, f"trial_{tag}.pt")
    meta_file = os.path.join(rdir, f"trial_{tag}.json")
    curve_file = os.path.join(rdir, f"curve_{tag}.csv")
    resume_file = os.path.join(rdir, f"trial_{tag}_resume.pt")

    if os.path.exists(meta_file):
        meta = load_json(meta_file)
        if meta.get("fingerprint") != ctx.fp:
            raise RuntimeError(f"{meta_file} is from a different config. Use another --runs_dir.")
        print(f"    reusing trial {tag}: val acc {meta['final_val_acc']:.2f}%", flush=True)
        return meta

    model = load_dense_epoch(ctx, t)
    apply_mask(model, mask)
    opt = make_optimizer(model, cfg)
    # Same data order for every method, so CE and KD trials are paired.
    gen = torch.Generator().manual_seed(args.seed * 100_000 + rnd * 1000 + t)
    start = t
    if os.path.exists(resume_file):
        st = torch.load(resume_file, map_location=ctx.device, weights_only=False)
        if st.get("fingerprint") == ctx.fp:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            gen.set_state(st["gen"])
            start = st["epoch"]
            truncate_csv(curve_file, start)
            print(f"    resuming trial {tag} at epoch {start}", flush=True)
    if start == t and os.path.exists(curve_file):
        os.remove(curve_file)

    for epoch in range(start, T):
        run_loss, run_acc, gnorm = train_one_epoch(
            model, ctx.data, opt, epoch, gen, cfg, mask, teacher, args.kd_T, args.kd_lambda)
        m = evaluate_all(model, ctx.data, ctx.ref, splits=("val", "train"))
        append_csv(curve_file, dict(epoch=epoch + 1, run_train_loss=run_loss,
                                    run_train_acc=run_acc, grad_norm=gnorm, **m))
        if epoch + 1 < T:
            save_torch(resume_file, dict(model=model.state_dict(), opt=opt.state_dict(),
                                         gen=gen.get_state(), epoch=epoch + 1,
                                         fingerprint=ctx.fp))

    final = evaluate_all(model, ctx.data, ctx.ref)
    meta = dict(round=rnd, t=t, retrain_epochs=T - t, method=args.method, seed=args.seed,
                fingerprint=ctx.fp, **{f"final_{k}": v for k, v in final.items()})
    save_torch(model_file, model.state_dict())
    save_json(meta_file, meta)
    if os.path.exists(resume_file):
        os.remove(resume_file)
    print(f"    round {rnd} | t={t:3d} ({T - t:2d} epochs, {args.method}) -> "
          f"val acc {final['val_acc']:.2f}% (target {ctx.target:.2f}%) "
          f"KL {final['val_kl']:.4f}", flush=True)
    return meta


def rebuild_trials_csv(imp_dir, target):
    rows = []
    for f in sorted(glob.glob(os.path.join(imp_dir, "round_*", "trial_t*.json"))):
        m = load_json(f)
        m.pop("fingerprint", None)
        m["target_val_acc"] = target
        m["recovered"] = int(m["final_val_acc"] >= target)
        rows.append(m)
    if rows:
        write_csv(os.path.join(imp_dir, "rewind_trials.csv"),
                  sorted(rows, key=lambda r: (r["round"], r["t"])))


def cmd_imp(args):
    sd = seed_dir(args.runs_dir, args.dataset, args.model, args.seed)
    dense_final_path = os.path.join(sd, "dense", "final_metrics.json")
    if not os.path.exists(dense_final_path):
        raise SystemExit(f"No finished dense run at {dense_final_path}. Run the dense job first.")
    dense_final = load_json(dense_final_path)
    if dense_final["fingerprint"].get("train_limit") != args.train_limit:
        raise SystemExit("--train_limit differs from the dense run's.")
    T = int(dense_final["epochs"])
    cfg = make_cfg(T, args.model)
    if not 1 <= args.t_min < T:
        raise SystemExit(f"--t_min must be in [1, {T - 1}]")

    imp_dir = os.path.join(sd, args.method)
    os.makedirs(os.path.join(imp_dir, "masks"), exist_ok=True)
    fp = imp_fingerprint(args, T)
    cfg_path = os.path.join(imp_dir, "config.json")
    if os.path.exists(cfg_path) and load_json(cfg_path).get("fingerprint") != fp:
        raise SystemExit(f"{cfg_path} was made with a different config "
                         f"({load_json(cfg_path).get('fingerprint')} vs {fp}). "
                         "Use another --runs_dir.")
    if os.path.exists(os.path.join(imp_dir, "DONE.json")):
        print(f"IMP chain already finished: {imp_dir}", flush=True)
        return

    # Recovery target in relative error: err <= err_dense * (1 + rel_tol).
    # rel_tol 0.055 is about 0.5 pp for ResNet-20 on CIFAR-10 (the old alpha).
    dense_err = 100.0 - dense_final["val_acc"]
    target = 100.0 - dense_err * (1 + args.rel_tol)

    device = get_device()
    set_seed(args.seed)
    data = Data(args.dataset, args.data_dir, device, cfg, args.train_limit)
    ctx = Context()
    ctx.args, ctx.cfg, ctx.device, ctx.data, ctx.sd, ctx.fp, ctx.target = \
        args, cfg, device, data, sd, fp, target

    teacher_dense = load_dense_epoch(ctx, T)
    teacher_dense.eval()
    ctx.ref = teacher_reference(teacher_dense, data)       # KL is always measured to dense
    flops = conv_flops(teacher_dense, device)
    B = budget_table(cfg)
    t_probe = int(round(args.probe_frac * T)) if args.probe_frac > 0 else None
    save_json(cfg_path, dict(fingerprint=fp, **vars(args), T=T, lr=cfg["lr"],
                             milestones=cfg["milestones"], dense_val_acc=dense_final["val_acc"],
                             target_val_acc=target, t_probe=t_probe, B=list(B)))
    print(f"IMP {args.dataset}/{args.model}/seed{args.seed}/{args.method}: dense val acc "
          f"{dense_final['val_acc']:.2f}% -> target {target:.2f}% (rel_tol {args.rel_tol})",
          flush=True)

    model = copy.deepcopy(teacher_dense)
    mask = full_mask(model)
    summaries, status = [], "completed"

    for rnd in range(1, args.rounds + 1):
        rdir = os.path.join(imp_dir, f"round_{rnd:02d}")
        os.makedirs(rdir, exist_ok=True)
        summary_file = os.path.join(rdir, "summary.json")
        final_file = os.path.join(rdir, "final_model.pt")
        mask_file = os.path.join(imp_dir, "masks", f"round_{rnd:02d}.pt")

        if os.path.exists(summary_file):
            s = load_json(summary_file)
            summaries.append(s)
            if s["t_star"] is None:
                print(f"Round {rnd}: previously unrecoverable; chain stopped.", flush=True)
                status = "stopped_unrecoverable"
                break
            mask = load_torch(mask_file, device)
            model = build_model(args.model, data.num_classes).to(device)
            model.load_state_dict(load_torch(final_file, device))
            print(f"Round {rnd}: already done (t* = {s['t_star']}), skipping.", flush=True)
            continue

        # 1. Prune (the mask for a round is fixed once written, so a resume reuses it).
        before = evaluate_all(model, data, ctx.ref)
        if os.path.exists(mask_file):
            new_mask = load_torch(mask_file, device)
        else:
            new_mask = global_magnitude_prune(model, mask, args.prune_frac)
            save_torch(mask_file, new_mask)
        w_before = flat_prunable(model)
        removed = torch.cat([(w * (mask[n] - new_mask[n])).flatten()
                             for n, w in prunable_params(model).items()]).detach()
        pruned = copy.deepcopy(model)
        apply_mask(pruned, new_mask)
        after_raw = evaluate_all(pruned, data, ctx.ref, splits=("val",))

        # 2. BN recalibration.
        recalibrate_bn(pruned, data, torch.Generator().manual_seed(args.seed * 7 + rnd),
                       args.bn_batches)
        after = evaluate_all(pruned, data, ctx.ref)

        # 3. Damage and structure, measured before any retraining.
        dens, layer_d = density_measures(new_mask, flops)
        write_csv(os.path.join(rdir, "layer_density.csv"),
                  [dict(layer=n, density=v, n_weights=int(new_mask[n].numel()), macs=flops[n])
                   for n, v in layer_d.items()])
        err_b, err_a = 100 - before["val_acc"], 100 - after["val_acc"]
        kl_rise = max(after["val_kl"] - before["val_kl"], 0.0)
        damage = dict(
            D_acc_val=before["val_acc"] - after["val_acc"],
            D_loss_val=after["val_loss"] - before["val_loss"],
            D_acc_val_no_bn_recal=before["val_acc"] - after_raw["val_acc"],
            D_kl_val=kl_rise,
            D_relerr_val=float(np.log(max(err_a, 1e-2) / max(err_b, 1e-2))),
            weight_perturbation=removed.norm().item() / w_before.norm().item(),
        )
        print(f"Round {rnd}: sparsity {sparsity(new_mask) * 100:.1f}% | d_flops "
              f"{dens['d_flops']:.3f} | d_min {dens['d_min']:.3f} | D_acc {damage['D_acc_val']:.2f} "
              f"| D_kl {kl_rise:.4f}", flush=True)

        # Teacher used for retraining in this round.
        teacher = {"ce": None, "kd": teacher_dense, "kd_parent": copy.deepcopy(model).eval()
                   }[args.method]

        # 4. Search for t*: t_min, then the probe, then bisection.
        t_star, best, n_trials, spent = None, None, 0, 0

        def trial(t):
            nonlocal n_trials, spent
            meta = run_trial(ctx, new_mask, rnd, t, rdir, teacher)
            n_trials += 1
            spent += T - t
            return meta["final_val_acc"] >= target

        if after["val_acc"] >= target:
            t_star, best = T, pruned
        else:
            ok = trial(args.t_min)
            if ok:
                lo, hi, best_t = args.t_min, T, args.t_min
                if t_probe is not None and lo < t_probe < hi:
                    ok_p = trial(t_probe)
                    if ok_p:
                        lo, best_t = t_probe, t_probe
                    else:
                        hi = t_probe
                while hi - lo > args.resolution:
                    mid = (lo + hi) // 2
                    ok_m = trial(mid)
                    if ok_m:
                        lo, best_t = mid, mid
                    else:
                        hi = mid
                t_star = lo
                best = build_model(args.model, data.num_classes).to(device)
                best.load_state_dict(load_torch(
                    os.path.join(rdir, f"trial_t{best_t:03d}.pt"), device))

        rebuild_trials_csv(imp_dir, target)
        s = dict(round=rnd, method=args.method, seed=args.seed, sparsity=sparsity(new_mask),
                 **dens, **{f"before_{k}": v for k, v in before.items()},
                 **{f"after_{k}": v for k, v in after.items()},
                 after_raw_val_acc=after_raw["val_acc"], **damage,
                 t_star=t_star, retrain_epochs=None if t_star is None else T - t_star,
                 t_probe=t_probe, n_trials=n_trials, search_epochs=spent,
                 target_val_acc=target, dense_val_acc=dense_final["val_acc"], T=T,
                 rel_tol=args.rel_tol)
        if t_star is not None:
            final = evaluate_all(best, data, ctx.ref)
            s.update({f"final_{k}": v for k, v in final.items()})
            save_torch(final_file, best.state_dict())
        save_json(summary_file, s)
        summaries.append(s)
        write_csv(os.path.join(imp_dir, "rounds.csv"), summaries)

        if not args.keep_trial_models:          # trial weights are no longer needed
            for f in glob.glob(os.path.join(rdir, "trial_t*.pt")):
                os.remove(f)

        if t_star is None:
            print(f"Round {rnd}: even t={args.t_min} does not recover. Chain stopped.", flush=True)
            status = "stopped_unrecoverable"
            break
        print(f"Round {rnd}: t* = {t_star} ({T - t_star} retraining epochs, {n_trials} trials, "
              f"{spent} search epochs)", flush=True)
        model, mask = best, new_mask

    if summaries:
        write_csv(os.path.join(imp_dir, "rounds.csv"), summaries)
    save_json(os.path.join(imp_dir, "DONE.json"),
              dict(status=status, rounds=len(summaries), fingerprint=fp))
    print(f"IMP done ({status}). Results in {imp_dir}", flush=True)


# ==========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="download / check a dataset")
    common_args(pp)

    pd_ = sub.add_parser("dense", help="train the dense model (the teacher)")
    common_args(pd_)
    pd_.add_argument("--model", required=True)
    pd_.add_argument("--seed", type=int, required=True)
    pd_.add_argument("--epochs", type=int, default=80)

    pi = sub.add_parser("imp", help="IMP chain with t* search")
    common_args(pi)
    pi.add_argument("--model", required=True)
    pi.add_argument("--seed", type=int, required=True)
    pi.add_argument("--method", choices=METHODS, default="kd")
    pi.add_argument("--rounds", type=int, default=10)
    pi.add_argument("--prune_frac", type=float, default=0.2)
    pi.add_argument("--rel_tol", type=float, default=0.055,
                    help="recovered if val error <= dense val error * (1 + rel_tol)")
    pi.add_argument("--resolution", type=int, default=2)
    pi.add_argument("--t_min", type=int, default=2)
    pi.add_argument("--probe_frac", type=float, default=0.82,
                    help="probe rewind at round(probe_frac * T) after t_min; 0 disables")
    pi.add_argument("--kd_T", type=float, default=4.0)
    pi.add_argument("--kd_lambda", type=float, default=0.9)
    pi.add_argument("--bn_batches", type=int, default=100)
    pi.add_argument("--keep_trial_models", action="store_true",
                    help="keep every trial's weights (default: deleted once a round is done)")

    args = p.parse_args()
    {"prepare": cmd_prepare, "dense": cmd_dense, "imp": cmd_imp}[args.cmd](args)


if __name__ == "__main__":
    main()
