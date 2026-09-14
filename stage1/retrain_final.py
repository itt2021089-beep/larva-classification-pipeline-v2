"""
Retrain Stage 1 ResNet50 on the REPAIRED binary dataset.

    python -m stage1.retrain_final                  # both recipes
    python -m stage1.retrain_final --only original_recipe

Why retrain
-----------
finalize.verify found 31 evaluation images that were the same photograph as a
training image, and finalize.fix_leakage removed 33 of them from the Stage 1
training and validation sides. The previously reported 99.92% test accuracy
was therefore measured with 29 duplicates of test images sitting in the
training set. It has to be re-measured.

Data: final_datasets/stage_1_binary/manifests/{train,val,test}.csv
      train 6,066 | val 1,155 | test 1,245   (test unchanged by the repair)

Two recipes, same data, same budget, same seed:

  original_recipe   gamma(1.5) -> resize -> hflip -> rot15 -> jitter(0.2,0.2)
                    Exactly the augmentation the previous checkpoint used, so
                    the test number is directly comparable to 99.92%.

  smartphone_aug    gamma(1.5) -> the on-the-fly smartphone field pipeline
                    (configs/augmentation_config.yaml). Measures what
                    field-robustness training costs on the clean lab test set.

Both apply gamma(1.5) in TRAIN, VAL and TEST, matching demo/preprocessing.py,
so either checkpoint is a drop-in replacement for the deployed one.

Protocol, unchanged from the rest of the project:
  * augmentation on TRAIN only; val and test are deterministic
  * checkpoint selected on VALIDATION macro F1
  * test evaluated exactly once, after the best checkpoint is restored
"""

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from preprocessing import GammaCorrectionTransform
from stage1 import model as model_mod
from stage2 import metrics as metrics_mod
from stage2.engine import build_optimizer, run_epoch, set_seed
from training.smartphone_aug import SmartphoneAugment, load_config

STAGE = os.path.join("final_datasets", "stage_1_binary")
OUT_ROOT = os.path.join("results", "stage1", "resnet50_repaired")
CLASSES = ["mosquito", "non_mosquito"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Cfg:
    name: str = "original_recipe"
    recipe: str = "original"          # "original" | "smartphone"
    img_size: int = 224
    batch_size: int = 32
    gamma: float = 1.5
    dropout: float = 0.4
    pretrained: bool = True
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 15
    early_stopping: int = 5
    label_smoothing: float = 0.0
    num_workers: int = 4
    seed: int = 42


class ManifestImages(Dataset):
    """Reads a finalised split manifest. Augments TRAIN only, by construction."""

    def __init__(self, split, cfg, aug_cfg):
        path = os.path.join(STAGE, "manifests", split + ".csv")
        self.rows = list(csv.DictReader(open(path, encoding="utf-8")))
        self.split = split
        self.labels = [CLASSES.index(r["class"]) for r in self.rows]
        self.gamma = GammaCorrectionTransform(gamma=cfg.gamma)
        self.train = split == "train"
        self.recipe = cfg.recipe
        self.sp = (SmartphoneAugment(aug_cfg, seed=cfg.seed)
                   if (self.train and cfg.recipe == "smartphone") else None)
        if self.train and cfg.recipe == "original":
            self.geo = transforms.Compose([
                transforms.Resize((cfg.img_size, cfg.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.geo = transforms.Resize((cfg.img_size, cfg.img_size))
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.rows)

    def counts(self):
        from collections import Counter
        c = Counter(r["class"] for r in self.rows)
        return {k: c[k] for k in CLASSES}

    def __getitem__(self, i):
        img = Image.open(self.rows[i]["final_path"]).convert("RGB")
        img = self.gamma(img)                    # applied in every split
        if self.sp is not None:
            img = self.sp(img)                   # smartphone pipeline (train only)
        img = self.geo(img)
        return self.to_tensor(img), self.labels[i]


def _add_binary_auc(m, y_true, y_prob):
    """
    stage2.metrics.compute_metrics computes AUC only for the multiclass
    one-vs-rest case; with 2 classes sklearn needs the positive-class score as
    a 1-D array, so it bails out. Filled in here rather than changing that
    shared module, whose behaviour the existing Stage 2 reports depend on.
    Positive class = index 1 = non_mosquito.
    """
    if y_prob is None:
        return m
    try:
        from sklearn.metrics import roc_auc_score
        p = np.asarray(y_prob)
        if p.ndim == 2 and p.shape[1] == 2 and len(set(y_true)) == 2:
            m["auc_binary"] = float(roc_auc_score(y_true, p[:, 1]))
            m["auc_positive_class"] = CLASSES[1]
            m.pop("auc_note", None)
    except Exception as exc:
        m["auc_note"] = "binary AUC skipped: " + str(exc)
    return m


def loaders_for(cfg, aug_cfg):
    ds = {sp: ManifestImages(sp, cfg, aug_cfg) for sp in ("train", "val", "test")}
    out = {
        "train": DataLoader(ds["train"], batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=cfg.num_workers > 0),
        "val": DataLoader(ds["val"], batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=cfg.num_workers > 0),
        "test": DataLoader(ds["test"], batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True),
    }
    return out, ds


def train_one(cfg, aug_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(OUT_ROOT, cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "best_model.pth")

    print("\n" + "=" * 78)
    print("STAGE 1 ResNet50 - " + cfg.name + "  (repaired dataset)")
    print("=" * 78)

    set_seed(cfg.seed)
    loaders, ds = loaders_for(cfg, aug_cfg)
    for sp in ("train", "val", "test"):
        print("  " + sp.ljust(6) + str(len(ds[sp])).rjust(6) + "  "
              + str(ds[sp].counts())
              + ("  [augmented]" if ds[sp].train else "  [deterministic]"))

    set_seed(cfg.seed)
    model = model_mod.build_resnet50(num_classes=len(CLASSES),
                                     dropout=cfg.dropout,
                                     pretrained=cfg.pretrained, device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    opt = build_optimizer(model, cfg.optimizer, cfg.learning_rate,
                          cfg.weight_decay, 0.9)

    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        e0 = time.time()
        trl, tra, trf, *_ = run_epoch(model, loaders["train"], criterion, opt,
                                      device, True, scaler)
        val, vaa, vaf, *_ = run_epoch(model, loaders["val"], criterion, None,
                                      device, False, scaler)
        history.append({"epoch": ep, "train_loss": trl, "train_acc": tra,
                        "train_f1_macro": trf, "val_loss": val, "val_acc": vaa,
                        "val_f1_macro": vaf, "seconds": time.time() - e0})
        imp = vaf > best_f1 + 1e-6
        if imp:
            best_f1, best_epoch, stale = vaf, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
        print("  epoch %2d/%d | train loss %.4f acc %5.2f%% F1 %.4f | "
              "val loss %.4f acc %5.2f%% F1 %.4f | %.0fs%s"
              % (ep, cfg.epochs, trl, tra, trf, val, vaa, vaf,
                 history[-1]["seconds"], "  *best*" if imp else ""))
        if cfg.early_stopping and stale >= cfg.early_stopping:
            print("  early stop: no val macro-F1 gain for "
                  + str(cfg.early_stopping) + " epochs")
            break
    secs = time.time() - t0

    # restore best, then touch test exactly once
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    ev = {}
    for sp in ("val", "test"):
        _, _, _, yt, yp, ypr = run_epoch(model, loaders[sp], criterion, None,
                                         device, False, scaler, collect_probs=True)
        ev[sp] = metrics_mod.compute_metrics(yt, yp, ypr, CLASSES)
        _add_binary_auc(ev[sp], yt, ypr)

    rec = {
        "run_name": cfg.name, "config": asdict(cfg),
        "dataset": STAGE, "dataset_state": "repaired (38 leakage images removed)",
        "class_names": CLASSES,
        "class_mapping": {str(i): c for i, c in enumerate(CLASSES)},
        "split_sizes": {sp: len(ds[sp]) for sp in ("train", "val", "test")},
        "best_epoch": best_epoch, "best_val_f1_macro": best_f1,
        "epochs_run": len(history), "train_minutes": secs / 60.0,
        "checkpoint": ckpt.replace("\\", "/"), "metrics": ev,
        "selection_rule": "best VALIDATION macro F1; test evaluated once after restore",
    }
    json.dump(rec, open(os.path.join(out_dir, "metrics.json"), "w",
                        encoding="utf-8"), indent=2)
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w",
                            encoding="utf-8"), indent=2)
    print("\n[" + cfg.name + "] best val macro-F1 %.4f @ epoch %d  (%.1f min)"
          % (best_f1, best_epoch, secs / 60))
    print(metrics_mod.format_metrics("TEST", ev["test"], CLASSES))
    return rec


RUNS = [
    # lr 1e-3 -- the project's Stage 1 BASELINE config (98.31% before repair).
    Cfg(name="original_recipe", recipe="original"),
    # Same data and budget, smartphone field augmentation instead of the light
    # geometric/colour augmentation.
    Cfg(name="smartphone_aug", recipe="smartphone"),
    # lr 1e-4 -- the config of the DEPLOYED checkpoint
    # (demo_models/stage1_resnet50/best_model.pth, training_run "tuning/lr_1e-4",
    # sha256 04bd6283...). This is the only like-for-like comparison with the
    # 99.92% currently quoted in demo/config.py: that figure came from the
    # lr_1e-4 tuning run, NOT from the lr 1e-3 baseline. It peaked at epoch 1.
    Cfg(name="deployed_recipe_lr1e-4", recipe="original", learning_rate=1e-4),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    a = ap.parse_args()

    aug_cfg = load_config()
    os.makedirs(OUT_ROOT, exist_ok=True)
    out = []
    for cfg in RUNS:
        if a.only and cfg.name != a.only:
            continue
        if a.epochs:
            cfg.epochs = a.epochs
        if a.num_workers is not None:
            cfg.num_workers = a.num_workers
        done = os.path.join(OUT_ROOT, cfg.name, "metrics.json")
        if os.path.exists(done):
            print("[skip] " + cfg.name + " already has metrics.json")
            out.append(json.load(open(done, encoding="utf-8")))
            continue
        out.append(train_one(cfg, aug_cfg))

    print("\n" + "=" * 78)
    print("%-20s %10s %10s %10s %10s" % ("run", "val F1", "test acc",
                                         "test F1", "epochs"))
    print("-" * 78)
    for r in out:
        m = r["metrics"]["test"]
        print("%-20s %10.4f %9.2f%% %10.4f %10d"
              % (r["run_name"], r["best_val_f1_macro"],
                 m["accuracy"] * 100 if m["accuracy"] <= 1 else m["accuracy"],
                 m["f1_macro"], r["epochs_run"]))
    json.dump(out, open(os.path.join(OUT_ROOT, "summary.json"), "w",
                        encoding="utf-8"), indent=2)
    print("\n[out] " + OUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
