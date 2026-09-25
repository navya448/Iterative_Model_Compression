"""Shared pieces: config, data, ResNet-20, training, evaluation, pruning.

Everything here follows the experiment plan:
  ResNet-20 / CIFAR-10, 45k train / 5k val / 10k test, T = 80 epochs,
  SGD(0.1, momentum 0.9, wd 1e-4, batch 128), LR / 10 at epochs 40 and 60,
  global magnitude pruning of all conv weights (final FC layer kept dense).
"""

import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULTS = dict(
    epochs=80,              # T
    batch_size=128,
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    milestones=[40, 60],    # LR is divided by 10 at these epochs
    gamma=0.1,
    val_size=5000,          # held out from the 50k CIFAR-10 training set
    train_eval_size=10000,  # fixed clean subset used to measure training loss/acc
    split_seed=0,           # the train/val split is the SAME for every seed
)

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def get_device():
    """Use the CUDA GPU when PyTorch can see one, else the CPU. Prints which."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)}, "
              f"torch {torch.__version__}, CUDA {torch.version.cuda})")
    else:
        device = torch.device("cpu")
        print(f"Using device: cpu (torch {torch.__version__}, CUDA build: {torch.version.cuda})")
        print("  WARNING: CUDA not available. If this machine has an NVIDIA GPU, install a "
              "CUDA build of PyTorch, e.g.\n"
              "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at_epoch(epoch, cfg=DEFAULTS):
    """Learning rate used DURING epoch `epoch` (epochs are 0-indexed: 0..T-1).

    Because the LR depends only on the epoch number, rewinding to epoch t and
    training to T automatically replays the original schedule from t onward.
    """
    lr = cfg["lr"]
    for m in cfg["milestones"]:
        if epoch >= m:
            lr *= cfg["gamma"]
    return lr


# --------------------------------------------------------------------------
# Data: whole CIFAR-10 kept as tensors on the device, augmentation on the fly.
# This is much faster than a DataLoader for a model this small.
# --------------------------------------------------------------------------

class CIFARData:
    def __init__(self, root, device, cfg=DEFAULTS):
        import torchvision

        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)

        def to_tensor(ds):
            x = torch.tensor(ds.data).permute(0, 3, 1, 2).float().div_(255)
            x = (x - MEAN) / STD
            y = torch.tensor(ds.targets, dtype=torch.long)
            return x.to(device), y.to(device)

        x_all, y_all = to_tensor(tr)
        g = torch.Generator().manual_seed(cfg["split_seed"])
        perm = torch.randperm(len(y_all), generator=g).to(device)
        val_idx = perm[: cfg["val_size"]]
        train_idx = perm[cfg["val_size"]:]

        self.x_train, self.y_train = x_all[train_idx], y_all[train_idx]
        self.x_val, self.y_val = x_all[val_idx], y_all[val_idx]
        self.x_test, self.y_test = to_tensor(te)
        # fixed, un-augmented subset of the TRAINING data for clean train metrics
        n = cfg["train_eval_size"]
        self.x_train_eval = self.x_train[:n]
        self.y_train_eval = self.y_train[:n]
        self.device = device
        self.batch_size = cfg["batch_size"]

    def train_batches(self, generator):
        """One epoch of shuffled, augmented batches (random crop + flip)."""
        n = len(self.y_train)
        perm = torch.randperm(n, generator=generator).to(self.device)
        for i in range(0, n, self.batch_size):
            idx = perm[i: i + self.batch_size]
            yield augment(self.x_train[idx], generator), self.y_train[idx]


def augment(x, generator):
    """Random 32x32 crop from a 4-pixel zero-padded image + horizontal flip."""
    b = x.shape[0]
    dev = x.device
    padded = F.pad(x, (4, 4, 4, 4))
    ix = torch.randint(0, 9, (b,), generator=generator).to(dev)
    iy = torch.randint(0, 9, (b,), generator=generator).to(dev)
    base = torch.arange(32, device=dev)
    rows = (iy[:, None] + base)[:, None, :, None]          # b,1,32,1
    cols = (ix[:, None] + base)[:, None, None, :]          # b,1,1,32
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(3, device=dev)[None, :, None, None]
    out = padded[bi, ci, rows, cols]
    flip = (torch.rand(b, generator=generator) < 0.5).to(dev)
    out[flip] = out[flip].flip(3)
    return out


# --------------------------------------------------------------------------
# ResNet-20 (He et al. 2016, CIFAR version: 3 stages x 3 blocks, 16/32/64)
# --------------------------------------------------------------------------

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        layers, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(3):
                layers.append(BasicBlock(cin, cout, stride if i == 0 else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def prunable_names(model):
    """Names of the weights that IMP is allowed to prune: every conv weight.
    The final FC layer (and all BN params / biases) stay dense."""
    return [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def prunable_params(model):
    mods = dict(model.named_modules())
    return {n: mods[n].weight for n in prunable_names(model)}


# --------------------------------------------------------------------------
# Masks and pruning
# --------------------------------------------------------------------------

def full_mask(model):
    return {n: torch.ones_like(w) for n, w in prunable_params(model).items()}


def apply_mask(model, mask):
    with torch.no_grad():
        for n, w in prunable_params(model).items():
            w.mul_(mask[n])


def sparsity(mask):
    total = sum(m.numel() for m in mask.values())
    kept = sum(m.sum().item() for m in mask.values())
    return 1.0 - kept / total


def global_magnitude_prune(model, mask, frac):
    """Remove `frac` of the currently surviving prunable weights, globally by |w|."""
    params = prunable_params(model)
    alive = torch.cat([params[n].detach().abs()[mask[n].bool()] for n in params])
    k = int(round(frac * alive.numel()))
    if k == 0:
        return {n: m.clone() for n, m in mask.items()}
    threshold = torch.kthvalue(alive.cpu(), k).values.item()
    new_mask = {}
    for n, w in params.items():
        keep = (w.detach().abs() > threshold) & mask[n].bool()
        new_mask[n] = keep.float()
    return new_mask


def flat_prunable(model):
    return torch.cat([w.detach().flatten() for w in prunable_params(model).values()])


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------

def make_optimizer(model, cfg=DEFAULTS):
    return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                           weight_decay=cfg["weight_decay"])


def train_one_epoch(model, data, opt, epoch, generator, mask=None, cfg=DEFAULTS):
    """Train for one epoch. If `mask` is given, pruned weights stay exactly zero.
    Returns (running train loss, running train acc, mean gradient norm)."""
    lr = lr_at_epoch(epoch, cfg)
    for g in opt.param_groups:
        g["lr"] = lr
    model.train()
    params = prunable_params(model) if mask is not None else None
    tot_loss = tot_correct = tot_n = 0
    grad_norms = []
    for x, y in data.train_batches(generator):
        out = model(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if mask is not None:
            for n, w in params.items():
                w.grad.mul_(mask[n])
        gn = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                            for p in model.parameters() if p.grad is not None))
        grad_norms.append(gn.item())
        opt.step()
        if mask is not None:
            apply_mask(model, mask)
        tot_loss += loss.item() * len(y)
        tot_correct += (out.argmax(1) == y).sum().item()
        tot_n += len(y)
    return tot_loss / tot_n, 100.0 * tot_correct / tot_n, float(np.mean(grad_norms))


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    tot_loss = tot_correct = 0.0
    for i in range(0, len(y), batch_size):
        out = model(x[i: i + batch_size])
        tot_loss += F.cross_entropy(out, y[i: i + batch_size], reduction="sum").item()
        tot_correct += (out.argmax(1) == y[i: i + batch_size]).sum().item()
    return tot_loss / len(y), 100.0 * tot_correct / len(y)


def evaluate_all(model, data):
    """Loss/acc on validation, clean train subset and test."""
    vl, va = evaluate(model, data.x_val, data.y_val)
    tl, ta = evaluate(model, data.x_train_eval, data.y_train_eval)
    sl, sa = evaluate(model, data.x_test, data.y_test)
    return dict(val_loss=vl, val_acc=va, train_loss=tl, train_acc=ta,
                test_loss=sl, test_acc=sa)


@torch.no_grad()
def recalibrate_bn(model, data, generator, num_batches=100):
    """Re-estimate BatchNorm running stats for the current (pruned) weights.
    No weights are updated. Uses a cumulative average over `num_batches`."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i, (x, _) in enumerate(data.train_batches(generator)):
        if i >= num_batches:
            break
        model(x)
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()


# --------------------------------------------------------------------------
# Small file helpers
# --------------------------------------------------------------------------

def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def write_csv(path, rows):
    fields = []
    for r in rows:                      # union of keys, in first-seen order
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ckpt_path(run_dir, epoch):
    return os.path.join(run_dir, "dense", "checkpoints", f"epoch_{epoch:03d}.pt")
