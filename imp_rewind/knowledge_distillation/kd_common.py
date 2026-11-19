"""Shared code for the knowledge-distillation (KD) experiments.

Same protocol as imp_rewind/common.py (45k/5k split with split_seed 0, T = 80,
SGD 0.1 / momentum 0.9 / wd 1e-4 / batch 128, LR / 10 at 50% and 75% of T,
global magnitude pruning of all conv weights, FC kept dense), extended to

  datasets : cifar10 (used); cifar100, svhn are supported but not in the plan
  models   : resnet20/32/44/56, vgg11          (CIFAR versions)
  training : plain cross-entropy, or KD from a teacher's soft targets
  measures : KL to the dense teacher, per-layer densities, FLOP-weighted density

Nothing here imports the old code, so the old experiments are untouched.
For CIFAR-10 the split and the ResNet-20 module names match common.py exactly.
"""

import csv
import hashlib
import json
import math
import os
import pickle
import random
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULTS = dict(
    epochs=80,
    batch_size=128,
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    milestone_fracs=(0.5, 0.75),   # LR / 10 at 50% and 75% of T (40 and 60 for T = 80)
    gamma=0.1,
    val_size=5000,
    train_eval_size=10000,
    split_seed=0,
)

# Peak LR per model. VGG-11 is less stable at 0.1. The rewind clock is a ratio of
# LR budgets in units of the peak LR, so a different peak does not change its scale.
MODEL_LR = {"resnet20": 0.1, "resnet32": 0.1, "resnet44": 0.1, "resnet56": 0.1, "vgg11": 0.05}


def make_cfg(epochs=None, model=None, **overrides):
    cfg = dict(DEFAULTS)
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    if model is not None:
        cfg["lr"] = MODEL_LR[model]
    cfg.update(overrides)
    T = cfg["epochs"]
    cfg["milestones"] = sorted({max(1, int(round(f * T))) for f in cfg["milestone_fracs"]})
    return cfg


def lr_at_epoch(epoch, cfg):
    """LR used DURING epoch `epoch` (0-indexed). Depends only on the epoch, so a
    rewind to t followed by training to T replays the original schedule."""
    lr = cfg["lr"]
    for m in cfg["milestones"]:
        if epoch >= m:
            lr *= cfg["gamma"]
    return lr


def budget_table(cfg):
    """B[t] = LR budget left after epoch t, in units of the peak LR (t = 0..T)."""
    T = cfg["epochs"]
    lrs = np.array([lr_at_epoch(e, cfg) for e in range(T)]) / cfg["lr"]
    return np.array([lrs[t:].sum() for t in range(T + 1)])


# --------------------------------------------------------------------------
# Sharing the GPU (same as common.py; set IMP_GPU_FRACTION, e.g. 0.5)
# --------------------------------------------------------------------------

class GpuThrottle:
    def __init__(self, fraction=1.0, min_sleep=0.02):
        if not 0 < fraction <= 1:
            raise ValueError(f"GPU fraction must be in (0, 1], got {fraction}")
        self.fraction, self.min_sleep = fraction, min_sleep
        self.owed, self.last, self.enabled = 0.0, None, False

    def start(self):
        self.last = time.perf_counter()

    def tick(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self.last is not None:
            self.owed += (now - self.last) * (1 - self.fraction) / self.fraction
            if self.owed >= self.min_sleep:
                t0 = time.perf_counter()
                time.sleep(self.owed)
                self.owed -= time.perf_counter() - t0
        self.last = time.perf_counter()


THROTTLE = GpuThrottle(float(os.environ.get("IMP_GPU_FRACTION", "1")))


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)}, torch {torch.__version__})",
              flush=True)
        if THROTTLE.fraction < 1:
            THROTTLE.enabled = True
            torch.cuda.set_per_process_memory_fraction(THROTTLE.fraction, device)
            print(f"  GPU share limited to {THROTTLE.fraction:.0%}", flush=True)
    else:
        device = torch.device("cpu")
        print(f"Using device: cpu (torch {torch.__version__}). WARNING: no CUDA GPU visible.",
              flush=True)
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Datasets: raw files are read directly (no torchvision needed)
# --------------------------------------------------------------------------

DATASETS = {
    "cifar10": dict(classes=10, flip=True,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "cifar100": dict(classes=100, flip=True,
                     mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    # Digits: no horizontal flip.
    "svhn": dict(classes=10, flip=False,
                 mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)),
}

DOWNLOADS = {
    "cifar10": [("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                 "cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a")],
    "cifar100": [("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
                  "cifar-100-python.tar.gz", "eb9058c3a382ffc7106e4002c42a8d85")],
    "svhn": [("http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
              os.path.join("svhn", "train_32x32.mat"), "e26dedcc434d2e4c54c9b2d4a06d8373"),
             ("http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
              os.path.join("svhn", "test_32x32.mat"), "eb5a983be6a315427106f1b164d9cef3")],
}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, path, md5):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and _md5(path) == md5:
        return
    tmp = path + ".part"
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    got = _md5(tmp)
    if got != md5:
        os.remove(tmp)
        raise RuntimeError(f"MD5 mismatch for {url}: got {got}, expected {md5}")
    os.replace(tmp, path)


def _extract(tar_path, root):
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers()
                   if not (m.name.startswith("/") or ".." in m.name.split("/"))
                   and (m.isfile() or m.isdir())]
        tf.extractall(root, members=members)


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _load_cifar(root, name):
    if name == "cifar10":
        d = os.path.join(root, "cifar-10-batches-py")
        train_files, test_file, key = [f"data_batch_{i}" for i in range(1, 6)], "test_batch", b"labels"
    else:
        d = os.path.join(root, "cifar-100-python")
        train_files, test_file, key = ["train"], "test", b"fine_labels"
    if not all(os.path.exists(os.path.join(d, f)) for f in train_files + [test_file]):
        url, fname, md5 = DOWNLOADS[name][0]
        tar_path = os.path.join(root, fname)
        _download(url, tar_path, md5)
        _extract(tar_path, root)

    def read(files):
        xs, ys = [], []
        for f in files:
            b = _unpickle(os.path.join(d, f))
            xs.append(np.asarray(b[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
            ys.append(np.asarray(b[key], dtype=np.int64))
        return np.concatenate(xs), np.concatenate(ys)

    return (*read(train_files), *read([test_file]))


def _load_svhn(root):
    from scipy.io import loadmat
    out = []
    for url, fname, md5 in DOWNLOADS["svhn"]:
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            _download(url, path, md5)
        m = loadmat(path)
        x = np.ascontiguousarray(m["X"].transpose(3, 2, 0, 1)).astype(np.uint8)  # N,3,32,32
        y = (m["y"].astype(np.int64).flatten() % 10)                             # label 10 = digit 0
        out += [x, y]
    return tuple(out)


def load_raw(name, root):
    """(x_train uint8 N,3,32,32, y_train, x_test, y_test). Downloads if missing."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name}; choose from {list(DATASETS)}")
    return _load_svhn(root) if name == "svhn" else _load_cifar(root, name)


class Data:
    """Whole dataset as normalized tensors on the device; augmentation on the fly."""

    def __init__(self, name, root, device, cfg, train_limit=None):
        info = DATASETS[name]
        xtr, ytr, xte, yte = load_raw(name, root)
        mean = torch.tensor(info["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(info["std"]).view(1, 3, 1, 1)

        def to_t(x, y):
            x = (torch.from_numpy(x).float().div_(255) - mean) / std
            return x.to(device), torch.from_numpy(y).long().to(device)

        x_all, y_all = to_t(xtr, ytr)
        g = torch.Generator().manual_seed(cfg["split_seed"])
        perm = torch.randperm(len(y_all), generator=g).to(device)
        val_idx, train_idx = perm[: cfg["val_size"]], perm[cfg["val_size"]:]
        if train_limit:
            train_idx = train_idx[: int(train_limit)]
        self.x_train, self.y_train = x_all[train_idx], y_all[train_idx]
        self.x_val, self.y_val = x_all[val_idx], y_all[val_idx]
        self.x_test, self.y_test = to_t(xte, yte)
        n = min(cfg["train_eval_size"], len(self.y_train))
        self.x_train_eval, self.y_train_eval = self.x_train[:n], self.y_train[:n]
        self.name, self.device = name, device
        self.batch_size = cfg["batch_size"]
        self.flip = info["flip"]
        self.num_classes = info["classes"]
        self.n_train = len(self.y_train)

    def train_batches(self, generator):
        n = len(self.y_train)
        perm = torch.randperm(n, generator=generator).to(self.device)
        for i in range(0, n, self.batch_size):
            idx = perm[i: i + self.batch_size]
            yield augment(self.x_train[idx], generator, self.flip), self.y_train[idx]


def augment(x, generator, flip=True):
    """Random 32x32 crop from the 4-pixel zero-padded image (+ horizontal flip)."""
    b = x.shape[0]
    ix = torch.randint(0, 9, (b,), generator=generator)
    iy = torch.randint(0, 9, (b,), generator=generator)
    fl = torch.rand(b, generator=generator) < 0.5
    if not flip:
        fl = torch.zeros_like(fl)
    return crop_flip(x, ix, iy, fl)


def crop_flip(x, ix, iy, flip):
    b, dev = x.shape[0], x.device
    ix, iy, flip = ix.to(dev), iy.to(dev), flip.to(dev)
    padded = F.pad(x, (4, 4, 4, 4))
    base = torch.arange(32, device=dev)
    rows = (iy[:, None] + base)[:, None, :, None]
    cols = (ix[:, None] + base)[:, None, None, :]
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(3, device=dev)[None, :, None, None]
    out = padded[bi, ci, rows, cols]
    out[flip] = out[flip].flip(3)
    return out


# --------------------------------------------------------------------------
# Models
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


class CifarResNet(nn.Module):
    """He et al. 2016 CIFAR ResNet, depth 6n+2 (20 -> n=3, 56 -> n=9), 16/32/64 channels.
    Module names match common.ResNet20."""

    def __init__(self, depth=20, num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        layers, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(n):
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
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


VGG11_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGG(nn.Module):
    """VGG with BatchNorm, CIFAR version (one 512 -> classes linear layer)."""

    def __init__(self, cfg=VGG11_CFG, num_classes=10):
        super().__init__()
        layers, cin = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(cin, v, 3, 1, 1, bias=False), nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                cin = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


MODELS = {
    "resnet20": lambda c: CifarResNet(20, c),
    "resnet32": lambda c: CifarResNet(32, c),
    "resnet44": lambda c: CifarResNet(44, c),
    "resnet56": lambda c: CifarResNet(56, c),
    "vgg11": lambda c: VGG(VGG11_CFG, c),
}


def build_model(name, num_classes):
    if name not in MODELS:
        raise ValueError(f"unknown model {name}; choose from {list(MODELS)}")
    return MODELS[name](num_classes)


# --------------------------------------------------------------------------
# Pruning
# --------------------------------------------------------------------------

def prunable_names(model):
    """Every conv weight is prunable. FC, BN and biases stay dense."""
    return [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def prunable_params(model):
    mods = dict(model.named_modules())
    return {n: mods[n].weight for n in prunable_names(model)}


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
    """Remove `frac` of the surviving prunable weights, globally by |w|."""
    params = prunable_params(model)
    alive = torch.cat([params[n].detach().abs()[mask[n].bool()] for n in params])
    k = int(round(frac * alive.numel()))
    if k == 0:
        return {n: m.clone() for n, m in mask.items()}
    threshold = torch.kthvalue(alive.cpu(), k).values.item()
    return {n: ((w.detach().abs() > threshold) & mask[n].bool()).float()
            for n, w in params.items()}


def flat_prunable(model):
    return torch.cat([w.detach().flatten() for w in prunable_params(model).values()])


@torch.no_grad()
def conv_flops(model, device):
    """Multiply-accumulates of every conv layer for one 32x32 image."""
    flops, hooks = {}, []
    mods = dict(model.named_modules())
    for n in prunable_names(model):
        m = mods[n]

        def hook(mod, inp, out, name=n):
            kh, kw = mod.kernel_size
            flops[name] = (out.shape[2] * out.shape[3] * mod.out_channels
                           * (mod.in_channels // mod.groups) * kh * kw)
        hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    model(torch.zeros(1, 3, 32, 32, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return flops


def density_measures(mask, flops):
    """Global kept fraction d plus three structure-aware effective densities."""
    sizes = {n: m.numel() for n, m in mask.items()}
    dens = {n: max(m.sum().item() / m.numel(), 1e-12) for n, m in mask.items()}
    tot = sum(sizes.values())
    ftot = sum(flops.values())
    return dict(
        d_global=sum(dens[n] * sizes[n] for n in mask) / tot,
        d_flops=sum(dens[n] * flops[n] for n in mask) / ftot,     # share of conv MACs kept
        d_min=min(dens.values()),                                  # most-pruned layer
        d_lgeo=math.exp(np.mean([math.log(v) for v in dens.values()])),  # layer geo-mean
    ), dens


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------

def make_optimizer(model, cfg):
    return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                           weight_decay=cfg["weight_decay"])


def kd_loss(out, y, t_logits, kd_T, kd_lambda):
    """(1 - lambda) CE + lambda T^2 KL(teacher_T || student_T)  (Hinton et al. 2015)."""
    ce = F.cross_entropy(out, y)
    if t_logits is None:
        return ce
    kl = F.kl_div(F.log_softmax(out / kd_T, 1), F.log_softmax(t_logits / kd_T, 1),
                  reduction="batchmean", log_target=True) * (kd_T ** 2)
    return (1 - kd_lambda) * ce + kd_lambda * kl


def train_one_epoch(model, data, opt, epoch, generator, cfg, mask=None,
                    teacher=None, kd_T=4.0, kd_lambda=0.9):
    """One epoch. With `teacher`, the loss is the KD loss on the same augmented batch.
    Returns (running train loss, running train acc, mean gradient norm)."""
    lr = lr_at_epoch(epoch, cfg)
    for g in opt.param_groups:
        g["lr"] = lr
    model.train()
    if teacher is not None:
        teacher.eval()
    params = prunable_params(model) if mask is not None else None
    tot_loss = torch.zeros((), device=data.device)
    tot_correct = torch.zeros((), device=data.device)
    tot_n, grad_norms = 0, []
    THROTTLE.start()
    for x, y in data.train_batches(generator):
        t_logits = None
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(x)
        out = model(x)
        loss = kd_loss(out, y, t_logits, kd_T, kd_lambda)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if mask is not None:
            for n, w in params.items():
                w.grad.mul_(mask[n])
        grad_norms.append(torch.linalg.vector_norm(torch.stack(
            [p.grad.detach().norm() for p in model.parameters() if p.grad is not None])))
        opt.step()
        if mask is not None:
            apply_mask(model, mask)
        tot_loss += loss.detach() * len(y)
        tot_correct += (out.argmax(1) == y).sum()
        tot_n += len(y)
        THROTTLE.tick()
    if not math.isfinite(tot_loss.item()):
        raise FloatingPointError(f"training loss became non-finite in epoch {epoch}")
    return (tot_loss.item() / tot_n, 100.0 * tot_correct.item() / tot_n,
            torch.stack(grad_norms).mean().item())


@torch.no_grad()
def logits_of(model, x, batch_size=1000):
    model.eval()
    outs = []
    THROTTLE.start()
    for i in range(0, len(x), batch_size):
        outs.append(model(x[i: i + batch_size]))
        THROTTLE.tick()
    return torch.cat(outs)


def kl_teacher_student(t_logits, s_logits):
    """Mean KL(teacher || student) at temperature 1, in nats."""
    lt, ls = F.log_softmax(t_logits, 1), F.log_softmax(s_logits, 1)
    return (lt.exp() * (lt - ls)).sum(1).mean().item()


def metrics_from_logits(logits, y, ref_logits=None, prefix=""):
    m = {f"{prefix}loss": F.cross_entropy(logits, y).item(),
         f"{prefix}acc": 100.0 * (logits.argmax(1) == y).float().mean().item()}
    if ref_logits is not None:
        m[f"{prefix}kl"] = kl_teacher_student(ref_logits, logits)
        m[f"{prefix}agree"] = 100.0 * (logits.argmax(1) == ref_logits.argmax(1)).float().mean().item()
    return m


def teacher_reference(model, data):
    """Logits of the dense (reference) model on the fixed eval sets, computed once."""
    return dict(val=logits_of(model, data.x_val), test=logits_of(model, data.x_test),
                train=logits_of(model, data.x_train_eval))


def evaluate_all(model, data, ref=None, splits=("val", "train", "test")):
    """Loss/acc on val, the clean train subset and test. With `ref` (teacher_reference
    of the dense model) also KL to the dense model and top-1 agreement."""
    sets = dict(val=(data.x_val, data.y_val), train=(data.x_train_eval, data.y_train_eval),
                test=(data.x_test, data.y_test))
    out = {}
    for s in splits:
        x, y = sets[s]
        out.update(metrics_from_logits(logits_of(model, x), y,
                                       None if ref is None else ref[s], prefix=f"{s}_"))
    return out


@torch.no_grad()
def recalibrate_bn(model, data, generator, num_batches=100):
    """Re-estimate BatchNorm running stats for the current weights (no weight update)."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    THROTTLE.start()
    for i, (x, _) in enumerate(data.train_batches(generator)):
        if i >= num_batches:
            break
        model(x)
        THROTTLE.tick()
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()


@torch.no_grad()
def teacher_entropy(logits):
    lp = F.log_softmax(logits, 1)
    return -(lp.exp() * lp).sum(1).mean().item()


# --------------------------------------------------------------------------
# Files (writes are atomic, so a crash never leaves a half-written file)
# --------------------------------------------------------------------------

def _atomic(path, write):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def save_json(path, obj):
    def w(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=float)
    _atomic(path, w)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_torch(path, obj):
    _atomic(path, lambda tmp: torch.save(obj, tmp))


def load_torch(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def write_csv(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]

    def w(tmp):
        with open(tmp, "w", newline="") as f:
            dw = csv.DictWriter(f, fieldnames=fields)
            dw.writeheader()
            dw.writerows(rows)
    _atomic(path, w)


def truncate_csv(path, max_epoch):
    """Keep rows with epoch <= max_epoch (used when resuming)."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(float(r["epoch"])) <= max_epoch]
    if rows:
        write_csv(path, rows)
    else:
        os.remove(path)


def setting_dir(runs_dir, dataset, model):
    return os.path.join(runs_dir, f"{dataset}_{model}")


def seed_dir(runs_dir, dataset, model, seed):
    return os.path.join(setting_dir(runs_dir, dataset, model), f"seed{seed}")


def dense_ckpt(sd, epoch):
    return os.path.join(sd, "dense", "checkpoints", f"epoch_{epoch:03d}.pt")
