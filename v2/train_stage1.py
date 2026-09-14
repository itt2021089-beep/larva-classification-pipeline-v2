"""
v2 step 3b - retrain Stage 1 (mosquito / non-mosquito) on lab AND field images.

    python -m v2.train_stage1

Why Stage 1 has to be retrained
-------------------------------
The deployed Stage 1 ResNet-50 was trained purely on laboratory photographs.
Run over the 2,211 iNaturalist field photographs it assigns p(larva) < 0.5 to

    aedes 42%   anopheles 36%   culex 37%

of GENUINE mosquito larva photos. Stage 1 gates the pipeline, so in deployment
it would silently discard more than a third of real field photos before the
species classifier ever saw them. A perfect Stage 2 cannot rescue that.

So Stage 1 is rebuilt from the v2 data:

    mosquito     = aedes + anopheles + culex   (field and lab)
    non_mosquito = unknown_objects             (chironomid larvae from the
                   field, plus the v1 non-larva images) topped up with the v1
                   binary dataset's non_mosquito images for volume

Splits are inherited from final_datasets_v2 so an observation that is in
Stage 2's test_field is also in Stage 1's test_field. Without that, a specimen
could train Stage 1 and test Stage 2, which is leakage across the pipeline.
"""

import argparse
import csv
import hashlib
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

V2 = "final_datasets_v2"
V1_BIN = os.path.join("final_datasets", "stage_1_binary")
OUT = os.path.join("results", "v2", "stage1_resnet50")
AUG_CFG = os.path.join("configs", "augmentation_v2.yaml")
CLASSES = ["mosquito", "non_mosquito"]
SPECIES = {"aedes", "anopheles", "culex"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Cfg:
    seed: int = 42
    image_size: int = 224
    per_class_in_batch: int = 16
    num_workers: int = 4
    dropout: float = 0.4
    label_smoothing: float = 0.05
    weight_decay: float = 1e-4
    backbone_lr: float = 5e-5
    classifier_lr: float = 5e-4
    warmup_epochs: int = 2
    freeze_backbone_epochs: int = 2
    epochs: int = 22
    early_stopping_patience: int = 6
    amp: bool = True
    grad_clip: float = 1.0
    v1_negatives_per_split: int = 900


def seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_rows(cfg):
    """Binary rows per split, inheriting the v2 split assignment."""
    out = defaultdict(list)
    for sp in ("train", "val", "test_field", "test_lab"):
        p = os.path.join(V2, "manifests", sp + ".csv")
        for r in csv.DictReader(open(p, encoding="utf-8")):
            out[sp].append({
                "path": r["final_path"],
                "label": "mosquito" if r["class"] in SPECIES else "non_mosquito",
                "domain": r["domain"], "group_id": r["group_id"],
                "origin": "v2_" + r["class"]})
    # top up negatives from the v1 binary dataset, keeping its own split
    rng = random.Random(cfg.seed)
    for v1sp, tgt in (("train", "train"), ("val", "val"), ("test", "test_lab")):
        f = os.path.join(V1_BIN, "manifests", v1sp + ".csv")
        if not os.path.exists(f):
            continue
        neg = [r for r in csv.DictReader(open(f, encoding="utf-8"))
               if r["class"] == "non_mosquito"]
        rng.shuffle(neg)
        for r in neg[:cfg.v1_negatives_per_split]:
            out[tgt].append({"path": r["final_path"], "label": "non_mosquito",
                             "domain": "lab", "group_id": r.get("group_id", ""),
                             "origin": "v1_binary"})
    return out


class S1Dataset(Dataset):
    def __init__(self, rows, split, cfg, aug_cfg):
        self.rows = rows
        self.split = split
        self.train = split == "train"
        self.cfg = cfg
        self.labels = [CLASSES.index(r["label"]) for r in rows]
        self.domains = [r["domain"] for r in rows]
        self.aug = (SmartphoneAugment(aug_cfg, seed=cfg.seed, final_resize=False)
                    if self.train else None)

    def __len__(self):
        return len(self.rows)

    def counts(self):
        c = Counter(r["label"] for r in self.rows)
        return {k: c.get(k, 0) for k in CLASSES}

    def __getitem__(self, i):
        img = Image.open(self.rows[i]["path"]).convert("RGB")
        if self.train:
            img = self.aug(img)
        img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        import torchvision.transforms.functional as TF
        return TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD), \
            self.labels[i], i


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, per_class, seed=42):
        self.by = defaultdict(list)
        for i, l in enumerate(labels):
            self.by[l].append(i)
        self.cls = sorted(self.by); self.per = per_class
        self.rng = random.Random(seed)
        self.batches = max(1, max(len(v) for v in self.by.values()) // per_class)

    def __len__(self):
        return self.batches

    def __iter__(self):
        pools = {c: [] for c in self.cls}
        for _ in range(self.batches):
            b = []
            for c in self.cls:
                if len(pools[c]) < self.per:
                    f = self.by[c][:]; self.rng.shuffle(f); pools[c].extend(f)
                b.extend(pools[c][:self.per]); pools[c] = pools[c][self.per:]
            self.rng.shuffle(b); yield b


def score(y, p, mask=None):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, precision_recall_fscore_support)
    if mask is not None:
        y = [a for a, m in zip(y, mask) if m]; p = [a for a, m in zip(p, mask) if m]
    if not y:
        return None
    pr, rc, f1, s = precision_recall_fscore_support(
        y, p, labels=[0, 1], zero_division=0)
    _, _, fm, _ = precision_recall_fscore_support(
        y, p, labels=[0, 1], average="macro", zero_division=0)
    return {"n": len(y), "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "f1_macro": float(fm),
            "mosquito_recall": float(rc[0]), "non_mosquito_recall": float(rc[1]),
            "per_class": {CLASSES[i]: {"precision": float(pr[i]),
                                       "recall": float(rc[i]), "f1": float(f1[i]),
                                       "support": int(s[i])} for i in (0, 1)},
            "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist()}


def run(model, loader, crit, opt, dev, train, scaler, cfg, domains=None):
    model.train() if train else model.eval()
    tot, n, ys, ps, ix = 0.0, 0, [], [], []
    for x, y, idx in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.set_grad_enabled(train):
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    o = model(x)
                loss = crit(o.float(), y)
            else:
                o = model(x); loss = crit(o, y)
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
        ys += y.cpu().tolist(); ps += o.float().argmax(1).cpu().tolist()
        ix += idx.tolist()
    res = {"all": score(ys, ps), "loss": tot / max(n, 1)}
    if domains is not None:
        for d in ("field", "lab"):
            res[d] = score(ys, ps, [domains[i] == d for i in ix])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    a = ap.parse_args()
    cfg = Cfg()
    if a.epochs: cfg.epochs = a.epochs
    if a.num_workers is not None: cfg.num_workers = a.num_workers

    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds(cfg.seed)
    aug_cfg = load_config(AUG_CFG)
    rows = build_rows(cfg)
    ds = {sp: S1Dataset(rows[sp], sp, cfg, aug_cfg) for sp in rows}
    sampler = BalancedBatchSampler(ds["train"].labels, cfg.per_class_in_batch, cfg.seed)
    ld = {"train": DataLoader(ds["train"], batch_sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0),
          "val": DataLoader(ds["val"], batch_size=32, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=cfg.num_workers > 0)}

    print("=" * 80)
    print("V2 STAGE 1 - ResNet50, lab + field")
    print("=" * 80)
    for sp in ("train", "val", "test_field", "test_lab"):
        print("  %-11s %5d  %s  %s" % (sp, len(ds[sp]), ds[sp].counts(),
                                       dict(Counter(ds[sp].domains))))

    from stage1 import model as s1m
    model = s1m.build_resnet50(num_classes=2, dropout=cfg.dropout,
                               pretrained=True, device=dev)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    bb = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    hd = [p for n, p in model.named_parameters() if n.startswith("fc")]
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg.backbone_lr},
                             {"params": hd, "lr": cfg.classifier_lr}],
                            weight_decay=cfg.weight_decay)
    import math

    def lr_l(e):
        if e < cfg.warmup_epochs:
            return (e + 1) / max(1, cfg.warmup_epochs)
        p = (e - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_l)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and dev.type == "cuda") else None
    for p in bb:
        p.requires_grad = False

    hist, best, best_ep, stale = [], None, 0, 0
    ck = os.path.join(OUT, "best_model.pth")
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        if ep == cfg.freeze_backbone_epochs + 1:
            for p in bb:
                p.requires_grad = True
            print("  -> backbone unfrozen")
        e0 = time.time()
        tr = run(model, ld["train"], crit, opt, dev, True, scaler, cfg)
        va = run(model, ld["val"], crit, None, dev, False, scaler, cfg,
                 ds["val"].domains)
        sch.step()
        sel = va["field"] or va["all"]
        imp = best is None or sel["f1_macro"] > best["f1_macro"] + 1e-4
        if imp:
            best, best_ep, stale = sel, ep, 0
            torch.save(model.state_dict(), ck)
        else:
            stale += 1
        hist.append({"epoch": ep, "train_f1": tr["all"]["f1_macro"],
                     "val_f1_all": va["all"]["f1_macro"],
                     "val_f1_field": sel["f1_macro"],
                     "val_mosq_recall_field": sel["mosquito_recall"],
                     "seconds": time.time() - e0, "selected": bool(imp)})
        print("  ep %2d/%d | tr F1 %.4f | val all %.4f | FIELD F1 %.4f "
              "mosq-recall %.4f | %.0fs%s"
              % (ep, cfg.epochs, tr["all"]["f1_macro"], va["all"]["f1_macro"],
                 sel["f1_macro"], sel["mosquito_recall"], hist[-1]["seconds"],
                 "  *best*" if imp else ""))
        if cfg.early_stopping_patience and stale >= cfg.early_stopping_patience:
            print("  early stop"); break

    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
    res = {}
    for sp in ("val", "test_field", "test_lab"):
        dl = DataLoader(ds[sp], batch_size=32, shuffle=False,
                        num_workers=cfg.num_workers)
        res[sp] = run(model, dl, crit, None, dev, False, scaler, cfg,
                      ds[sp].domains)
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
    json.dump({"architecture": "resnet50", "classes": CLASSES,
               "class_mapping": {"0": "mosquito", "1": "non_mosquito"},
               "counts": {sp: ds[sp].counts() for sp in ds},
               "best_epoch": best_ep, "epochs_run": len(hist),
               "train_minutes": (time.time() - t0) / 60,
               "results": res, "checkpoint": ck.replace("\\", "/"),
               "checkpoint_sha256": sha,
               "why_retrained": ("the v1 Stage 1 rejected 36-42% of genuine "
                                 "field mosquito-larva photos"),
               "environment": {"python": platform.python_version(),
                               "torch": torch.__version__},
               "config": asdict(cfg)},
              open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
              indent=2)
    print("\n  best epoch %d" % best_ep)
    for sp in ("test_field", "test_lab"):
        s = res[sp]["all"]
        print("  %-11s acc %.4f  macroF1 %.4f  mosquito-recall %.4f  n=%d"
              % (sp, s["accuracy"], s["f1_macro"], s["mosquito_recall"], s["n"]))
    print("  sha %s" % sha[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
