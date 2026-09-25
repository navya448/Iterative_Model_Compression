"""Experiment A: train the dense ResNet-20 once per seed and record everything.

Outputs (in <run_dir>/dense/):
  checkpoints/epoch_000.pt ... epoch_080.pt   model weights + BN stats after each epoch
  dense_log.csv                              one row per epoch (see columns below)
  final_metrics.json                         dense accuracy at T (used as the IMP target)

Usage:
  python train_dense.py --seed 1 --run_dir runs/seed1 --data_dir data

The script resumes automatically if it was interrupted (e.g. a Colab disconnect).
"""

import argparse
import os

import torch

from common import (DEFAULTS, CIFARData, ResNet20, append_csv, ckpt_path, evaluate_all,
                    flat_prunable, get_device, lr_at_epoch, make_optimizer, save_json,
                    set_seed, train_one_epoch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    args = p.parse_args()

    cfg = dict(DEFAULTS, epochs=args.epochs)
    device = get_device()
    out = os.path.join(args.run_dir, "dense")
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    log_path = os.path.join(out, "dense_log.csv")
    resume_path = os.path.join(out, "resume.pt")

    set_seed(args.seed)
    data = CIFARData(args.data_dir, device, cfg)
    model = ResNet20().to(device)
    opt = make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(args.seed)

    start = 0
    if os.path.exists(resume_path):
        st = torch.load(resume_path, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"])
        start = st["epoch"]
        print(f"Resuming dense training from epoch {start}")
    else:
        if os.path.exists(log_path):
            os.remove(log_path)
        # epoch 0 = the initialization
        torch.save(model.state_dict(), ckpt_path(args.run_dir, 0))
        m = evaluate_all(model, data)
        append_csv(log_path, dict(epoch=0, lr=float("nan"), run_train_loss=float("nan"),
                                  run_train_acc=float("nan"), **m, grad_norm=float("nan"),
                                  weight_change=float("nan"), weight_norm=flat_prunable(model).norm().item()))

    prev_w = flat_prunable(model).clone()
    for epoch in range(start, cfg["epochs"]):
        run_loss, run_acc, gnorm = train_one_epoch(model, data, opt, epoch, gen, cfg=cfg)
        e = epoch + 1  # the state AFTER this epoch is "epoch e"
        w = flat_prunable(model)
        m = evaluate_all(model, data)
        append_csv(log_path, dict(
            epoch=e,
            lr=lr_at_epoch(epoch, cfg),          # LR used to get from e-1 to e
            run_train_loss=run_loss,             # running (augmented) training loss
            run_train_acc=run_acc,
            **m,                                 # clean val / train-subset / test metrics
            grad_norm=gnorm,                     # mean per-batch gradient norm this epoch
            weight_change=(w - prev_w).norm().item(),   # ||W_e - W_{e-1}|| (conv weights)
            weight_norm=w.norm().item(),
        ))
        prev_w = w.clone()
        torch.save(model.state_dict(), ckpt_path(args.run_dir, e))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        gen=gen.get_state(), epoch=e), resume_path)
        print(f"epoch {e:3d} | lr {lr_at_epoch(epoch, cfg):.4f} | "
              f"train {m['train_loss']:.3f}/{m['train_acc']:.2f}% | "
              f"val {m['val_loss']:.3f}/{m['val_acc']:.2f}%")

    final = evaluate_all(model, data)
    save_json(os.path.join(out, "final_metrics.json"),
              dict(seed=args.seed, epochs=cfg["epochs"], **final))
    print("Dense done:", final)


if __name__ == "__main__":
    main()
