"""
v2 step 3 - train the field-robust species classifier.

    python -m v2.train --backbone efficientnet_v2_s
    python -m v2.train --backbone convnext_tiny
    python -m v2.train --backbone efficientnet_b0

Trains one backbone on final_datasets_v2 and writes it to
results/v2/<backbone>/. v2.evaluate then ensembles whichever backbones exist
and calibrates an abstention threshold.

What is different from v1, and why
----------------------------------
* The data now contains real smartphone field photographs, so the synthetic
  smartphone augmentation is DAMPED. It was compensating for a domain the
  dataset lacked; over-applying it on top of genuine field variation just
  destroys signal.
* Preprocessing is plain resize + ImageNet normalise. The v1 chain had a
  Gaussian blur measured to change the final tensor by 0.11% (a no-op after
  the 2.86x downscale) and a bounding-box crop that fired on only 31% of
  images and, when ablated, improved every metric. Both are configurable, and
  both default OFF.
* Checkpoint selection is driven by the FIELD validation subset, because field
  accuracy is the actual objective. Lab metrics are tracked but do not select.
* Balanced batches, as in the v1 repair, since Anopheles is still smallest.
"""

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from training.smartphone_aug import SmartphoneAugment, load_config

Image.MAX_IMAGE_PIXELS = None

DATA = "final_datasets_v2"
OUT_ROOT = os.path.join("results", "v2")
AUG_CFG = os.path.join("configs", "augmentation_v2.yaml")
CLASSES = ["aedes", "anopheles", "culex", "unknown_objects"]
SPECIES = ["aedes", "anopheles", "culex"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BACKBONES = {
    "efficientnet_b0": (torchvision.models.efficientnet_b0, "classifier"),
    "efficientnet_v2_s": (torchvision.models.efficientnet_v2_s, "classifier"),
    "convnext_tiny": (torchvision.models.convnext_tiny, "classifier"),
}


@dataclass
class Cfg:
    backbone: str = "efficientnet_v2_s"
    seed: int = 42
    image_size: int = 224
    per_class_in_batch: int = 8
    num_workers: int = 4
    # preprocessing - both default OFF, see module docstring
    bbox_crop: bool = False
    blur_enabled: bool = False
    blur_kernel: int = 3
    blur_sigma: float = 0.6
    # regularisation
    dropout: float = 0.30
    label_smoothing: float = 0.05
    weight_decay: float = 1e-4
    loss: str = "cb_focal"          # "ce" | "cb_focal"
    cb_beta: float = 0.999
    focal_gamma: float = 1.5
    # optimisation
    backbone_lr: float = 5e-5
    classifier_lr: float = 5e-4
    warmup_epochs: int = 3
    freeze_backbone_epochs: int = 2
    epochs: int = 35
    early_stopping_patience: int = 8
    amp: bool = True
    grad_clip: float = 1.0
    # selection
    select_on: str = "field"        # "field" | "all"
    macro_f1_tolerance: float = 0.004
    min_anopheles_val_recall: float = 0.55


def set_all_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init(_):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s); random.seed(s)


class V2Dataset(Dataset):
    def __init__(self, split, cfg, aug_cfg):
        self.rows = list(csv.DictReader(open(
            os.path.join(DATA, "manifests", split + ".csv"), encoding="utf-8")))
        self.split = split
        self.labels = [CLASSES.index(r["class"]) for r in self.rows]
        self.domains = [r["domain"] for r in self.rows]
        self.cfg = cfg
        self.train = split == "train"
        self.aug = (SmartphoneAugment(aug_cfg, seed=cfg.seed, final_resize=False)
                    if self.train else None)
        if cfg.bbox_crop or cfg.blur_enabled:
            from preprocessing import BoundingBoxCropTransform, GaussianBlurTransform
            self.bbox = BoundingBoxCropTransform(padding=10) if cfg.bbox_crop else None
            self.blur = (GaussianBlurTransform(kernel_size=cfg.blur_kernel,
                                               sigma=cfg.blur_sigma)
                         if cfg.blur_enabled else None)
        else:
            self.bbox = self.blur = None

    def __len__(self):
        return len(self.rows)

    def class_counts(self):
        c = Counter(r["class"] for r in self.rows)
        return {k: c.get(k, 0) for k in CLASSES}

    def __getitem__(self, i):
        img = Image.open(self.rows[i]["final_path"]).convert("RGB")
        if self.train:
            img = self.aug(img)
        if self.bbox is not None:
            img = self.bbox(img)
        if self.blur is not None:
            img = self.blur(img)
        img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        import torchvision.transforms.functional as TF
        x = TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
        return x, self.labels[i], i


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, per_class, seed=42):
        self.by = defaultdict(list)
        for i, l in enumerate(labels):
            self.by[l].append(i)
        self.classes = sorted(self.by)
        self.per = per_class
        self.rng = random.Random(seed)
        self.batches = max(1, max(len(v) for v in self.by.values()) // per_class)

    def __len__(self):
        return self.batches

    def __iter__(self):
        pools = {c: [] for c in self.classes}
        for _ in range(self.batches):
            b = []
            for c in self.classes:
                if len(pools[c]) < self.per:
                    f = self.by[c][:]; self.rng.shuffle(f); pools[c].extend(f)
                b.extend(pools[c][:self.per]); pools[c] = pools[c][self.per:]
            self.rng.shuffle(b)
            yield b


def effective_number_weights(counts, beta):
    w = np.array([(1 - beta) / (1 - beta ** max(counts[c], 1)) for c in CLASSES],
                 dtype=np.float64)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class CBFocalLoss(nn.Module):
    def __init__(self, weight, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma, self.ls = gamma, label_smoothing

    def forward(self, logits, target):
        logp = torch.log_softmax(logits.float(), 1)
        n = logits.size(1)
        with torch.no_grad():
            true = torch.zeros_like(logp).fill_(self.ls / (n - 1))
            true.scatter_(1, target.unsqueeze(1), 1.0 - self.ls)
        pt = logp.exp().gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1)
        per = -(true * logp).sum(1)
        return (((1 - pt) ** self.gamma) * self.weight.to(logits.device)[target] * per).mean()


def build_model(name, device, dropout):
    fn, head = BACKBONES[name]
    m = fn(weights="DEFAULT")
    cls = getattr(m, head)
    # every one of these backbones ends in Sequential(..., Linear); reconfigure
    # the existing dropout rather than stacking another
    for mod in cls.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = dropout
    for i in range(len(cls) - 1, -1, -1):
        if isinstance(cls[i], nn.Linear):
            cls[i] = nn.Linear(cls[i].in_features, len(CLASSES))
            break
    return m.to(device)


def feature_params(model, name):
    pre = "features" if name.startswith("efficientnet") or name.startswith("convnext") else ""
    bb = [p for n, p in model.named_parameters() if n.startswith(pre)]
    hd = [p for n, p in model.named_parameters() if not n.startswith(pre)]
    return bb, hd


def set_frozen(model, name, frozen):
    bb, _ = feature_params(model, name)
    for p in bb:
        p.requires_grad = not frozen
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def metrics(y, p, loss, mask=None):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, precision_recall_fscore_support)
    if mask is not None:
        y = [v for v, m in zip(y, mask) if m]
        p = [v for v, m in zip(p, mask) if m]
    if not y:
        return None
    pr, rc, f1, s = precision_recall_fscore_support(
        y, p, labels=list(range(len(CLASSES))), zero_division=0)
    _, _, fm, _ = precision_recall_fscore_support(
        y, p, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
    sp_idx = [CLASSES.index(c) for c in SPECIES]
    sp_mask = [i for i, t in enumerate(y) if t in sp_idx]
    sp_acc = (sum(1 for i in sp_mask if y[i] == p[i]) / len(sp_mask)) if sp_mask else 0.0
    return {"loss": loss, "n": len(y),
            "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "f1_macro": float(fm),
            "species_accuracy": float(sp_acc),
            "min_class_recall": float(min(rc)),
            "per_class": {CLASSES[i]: {"precision": float(pr[i]), "recall": float(rc[i]),
                                       "f1": float(f1[i]), "support": int(s[i])}
                          for i in range(len(CLASSES))},
            "confusion_matrix": confusion_matrix(
                y, p, labels=list(range(len(CLASSES)))).tolist()}


def run_epoch(model, loader, crit, opt, device, train, scaler, cfg, domains=None):
    model.train() if train else model.eval()
    tot, n, ys, ps, idxs = 0.0, 0, [], [], []
    for x, y, idx in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    out = model(x)
                loss = crit(out.float(), y)
            else:
                out = model(x); loss = crit(out, y)
        if train:
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
        tot += float(loss) * y.size(0); n += y.size(0)
        ys += y.detach().cpu().tolist()
        ps += out.detach().float().argmax(1).cpu().tolist()
        idxs += idx.tolist()
    all_m = metrics(ys, ps, tot / max(n, 1))
    out = {"all": all_m}
    if domains is not None:
        for dom in ("field", "lab"):
            mask = [domains[i] == dom for i in idxs]
            out[dom] = metrics(ys, ps, tot / max(n, 1), mask)
    return out


def better(c, b, cfg):
    if b is None:
        return True, "first epoch"
    d = c["f1_macro"] - b["f1_macro"]
    if d > cfg.macro_f1_tolerance:
        return True, "macro-F1 +%.4f" % d
    if d < -cfg.macro_f1_tolerance:
        return False, "macro-F1 %.4f" % d
    if abs(c["balanced_accuracy"] - b["balanced_accuracy"]) > 1e-3:
        return c["balanced_accuracy"] > b["balanced_accuracy"], "tied; balanced acc"
    if abs(c["min_class_recall"] - b["min_class_recall"]) > 1e-6:
        return c["min_class_recall"] > b["min_class_recall"], "tied; min class recall"
    return c["loss"] < b["loss"] - 1e-6, "tied; loss"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), default="efficientnet_v2_s")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--loss", choices=["ce", "cb_focal"], default=None)
    ap.add_argument("--image-size", type=int, default=None,
                    help="input resolution; 320 makes the Aedes/Culex siphon "
                         "resolvable in field photos where 224 does not")
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    a = ap.parse_args()

    cfg = Cfg(backbone=a.backbone)
    if a.epochs: cfg.epochs = a.epochs
    if a.num_workers is not None: cfg.num_workers = a.num_workers
    if a.loss: cfg.loss = a.loss
    if a.image_size:
        cfg.image_size = a.image_size
        if a.image_size >= 320 and not a.__dict__.get("_kept_batch"):
            # 6 per class x 4 = 24 keeps peak memory at 3.45 GB on a 6 GB card
            cfg.per_class_in_batch = 6

    out_dir = os.path.join(OUT_ROOT, cfg.backbone + (("_" + a.tag) if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(cfg.seed)
    aug_cfg = load_config(AUG_CFG if os.path.exists(AUG_CFG)
                          else os.path.join("configs", "augmentation_stage2.yaml"))

    ds = {sp: V2Dataset(sp, cfg, aug_cfg)
          for sp in ("train", "val", "test_field", "test_lab")}
    counts = ds["train"].class_counts()
    sampler = BalancedBatchSampler(ds["train"].labels, cfg.per_class_in_batch, cfg.seed)
    g = torch.Generator(); g.manual_seed(cfg.seed)
    loaders = {
        "train": DataLoader(ds["train"], batch_sampler=sampler,
                            num_workers=cfg.num_workers, pin_memory=True,
                            worker_init_fn=worker_init, generator=g,
                            persistent_workers=cfg.num_workers > 0),
        "val": DataLoader(ds["val"], batch_size=32, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=cfg.num_workers > 0),
    }

    print("=" * 84)
    print("V2 TRAIN - %s" % cfg.backbone)
    print("=" * 84)
    for sp in ds:
        d = Counter(ds[sp].domains)
        print("  %-11s %5d  %s  domains=%s" % (sp, len(ds[sp]),
                                               ds[sp].class_counts(), dict(d)))
    print("  batch: %d/class x %d = %d   batches/epoch %d"
          % (cfg.per_class_in_batch, len(CLASSES),
             cfg.per_class_in_batch * len(CLASSES), len(sampler)))
    print("  selection driven by: %s validation" % cfg.select_on.upper())

    model = build_model(cfg.backbone, device, cfg.dropout)
    cb_w = effective_number_weights(counts, cfg.cb_beta)
    crit = (nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
            if cfg.loss == "ce"
            else CBFocalLoss(cb_w, cfg.focal_gamma, cfg.label_smoothing).to(device))
    bb, hd = feature_params(model, cfg.backbone)
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg.backbone_lr},
                             {"params": hd, "lr": cfg.classifier_lr}],
                            weight_decay=cfg.weight_decay)
    import math

    def lr_l(e):
        if e < cfg.warmup_epochs:
            return (e + 1) / max(1, cfg.warmup_epochs)
        p = (e - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_l)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None
    set_frozen(model, cfg.backbone, True)

    hist, best, best_ep, best_why, stale = [], None, 0, "", 0
    ckpt = os.path.join(out_dir, "best_model.pth")
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        if ep == cfg.freeze_backbone_epochs + 1:
            n_tr = set_frozen(model, cfg.backbone, False)
            print("  -> backbone unfrozen (%d trainable)" % n_tr)
        e0 = time.time()
        tr = run_epoch(model, loaders["train"], crit, opt, device, True, scaler, cfg)
        va = run_epoch(model, loaders["val"], crit, None, device, False, scaler,
                       cfg, domains=ds["val"].domains)
        sched.step()
        sel = va[cfg.select_on] or va["all"]
        ok, why = better(sel, best, cfg)
        if ok and sel["per_class"]["anopheles"]["recall"] < cfg.min_anopheles_val_recall:
            ok, why = False, ("BLOCKED anopheles %s recall %.3f < %.2f"
                              % (cfg.select_on, sel["per_class"]["anopheles"]["recall"],
                                 cfg.min_anopheles_val_recall))
        if ok:
            best, best_ep, best_why, stale = sel, ep, why, 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
        f = va.get("field") or {}
        l = va.get("lab") or {}
        hist.append({"epoch": ep, "train_f1": tr["all"]["f1_macro"],
                     "val_f1_all": va["all"]["f1_macro"],
                     "val_f1_field": f.get("f1_macro"), "val_f1_lab": l.get("f1_macro"),
                     "val_species_acc_field": f.get("species_accuracy"),
                     "val_ano_recall_field": (f.get("per_class") or {}).get("anopheles", {}).get("recall"),
                     "seconds": time.time() - e0, "selected": bool(ok), "why": why})
        print("  ep %2d/%d | tr F1 %.4f | val(all) %.4f | FIELD F1 %.4f spAcc %.4f ano %.3f "
              "| lab F1 %.4f | %.0fs%s"
              % (ep, cfg.epochs, tr["all"]["f1_macro"], va["all"]["f1_macro"],
                 f.get("f1_macro", 0), f.get("species_accuracy", 0),
                 (f.get("per_class") or {}).get("anopheles", {}).get("recall", 0),
                 l.get("f1_macro", 0), hist[-1]["seconds"],
                 "  *best*" if ok else ("  " + why if why.startswith("BLOCKED") else "")))
        if cfg.early_stopping_patience and stale >= cfg.early_stopping_patience:
            print("  early stop"); break

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    va = run_epoch(model, loaders["val"], crit, None, device, False, scaler, cfg,
                   domains=ds["val"].domains)
    import hashlib
    sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    meta = {"backbone": cfg.backbone, "seed": cfg.seed,
            "class_mapping": {str(i): c for i, c in enumerate(CLASSES)},
            "counts": {sp: ds[sp].class_counts() for sp in ds},
            "domains": {sp: dict(Counter(ds[sp].domains)) for sp in ds},
            "best_epoch": best_ep, "best_reason": best_why,
            "epochs_run": len(hist), "train_minutes": (time.time() - t0) / 60,
            "validation": {k: v for k, v in va.items()},
            "checkpoint": ckpt.replace("\\", "/"), "checkpoint_sha256": sha,
            "test_note": "TEST NOT EVALUATED HERE - v2.evaluate does that once, "
                         "after the ensemble is fixed on validation.",
            "environment": {"python": platform.python_version(),
                            "torch": torch.__version__,
                            "torchvision": torchvision.__version__,
                            "gpu": torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None},
            "config": asdict(cfg)}
    json.dump(meta, open(os.path.join(out_dir, "metadata.json"), "w",
                         encoding="utf-8"), indent=2)
    with open(os.path.join(out_dir, "history.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist)
    fv = va.get("field") or va["all"]
    print("\n  best epoch %d (%s)" % (best_ep, best_why))
    print("  FIELD val: macroF1 %.4f  species-acc %.4f  balacc %.4f"
          % (fv["f1_macro"], fv["species_accuracy"], fv["balanced_accuracy"]))
    for c in CLASSES:
        p = fv["per_class"][c]
        print("     %-16s P %.3f R %.3f F1 %.3f n=%d"
              % (c, p["precision"], p["recall"], p["f1"], p["support"]))
    print("  sha %s" % sha[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
