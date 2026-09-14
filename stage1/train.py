"""
Stage 1 ResNet50 training.

    python -m stage1.train                 # baseline only
    python -m stage1.train --tune          # baseline + a small LR/dropout/batch probe

Reuses the project's existing, already-validated generic machinery rather than
reinventing it:
  * `stage2.engine.run_epoch / set_seed / build_optimizer / build_scheduler`
    — dataset-agnostic; the InceptionV3 aux-logit branch simply doesn't fire
    for ResNet50 (its forward returns a plain tensor).
  * `stage2.metrics.compute_metrics / format_metrics` — generic over class count.
  * `preprocessing.GammaCorrectionTransform` — the project's own preprocessing.

Baseline configuration comes from the research presentation's Stage 1 ResNet50
table (its best row, Test 11: batch 32, LR 0.001, dropout 0.4, Adam, 10 epochs,
"with preprocessing"), combined with `configs/config_binary.py` for the settings
the presentation does not state (224x224 input, seed 42).

Rules held throughout:
  * The leakage-free manifest split is used; training refuses to start if any
    duplicate-group spans splits.
  * Augmentation is applied to TRAIN only. Val and test use a deterministic
    transform and are never randomised.
  * The checkpoint is selected on VALIDATION macro F1. Test is evaluated once,
    after the checkpoint is restored, and never used for selection.
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from preprocessing import GammaCorrectionTransform
from stage1 import dataset as data_mod
from stage1 import model as model_mod
from stage2 import metrics as metrics_mod
from stage2.engine import build_optimizer, build_scheduler, run_epoch, set_seed

OUT_ROOT = os.path.join("results", "stage1", "resnet50")
CLASS_NAMES = data_mod.CLASS_NAMES          # ["Larva", "Non_larva"]


@dataclass
class Stage1Config:
    name: str = "baseline"
    changed: str = "(reference configuration)"
    # data
    img_size: int = 224                     # configs/config_binary.py
    batch_size: int = 32                    # presentation Test 11
    augment: bool = True                    # TRAIN only
    preprocessing: str = "gamma"            # project's GammaCorrectionTransform
    gamma: float = 1.5                      # preprocessing.py default
    num_workers: int = 4
    # model
    dropout: float = 0.4                    # presentation Test 11
    pretrained: bool = True
    # optimisation
    optimizer: str = "adam"                 # presentation
    learning_rate: float = 1e-3             # presentation Test 11
    weight_decay: float = 0.0
    momentum: float = 0.9                   # unused for Adam
    scheduler: str = None
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    epochs: int = 10                        # presentation Test 11
    early_stopping: int = 4                 # on val macro F1; 0 disables
    label_smoothing: float = 0.0
    seed: int = 42                          # configs/config_binary.py


def _preprocess_chain(cfg):
    if cfg.preprocessing == "gamma":
        return [GammaCorrectionTransform(gamma=cfg.gamma)]
    if cfg.preprocessing in (None, "none"):
        return None
    raise ValueError(f"unknown preprocessing {cfg.preprocessing!r}")


def _add_binary_auc(m, y_true, y_prob):
    """
    Add binary AUC-ROC.

    stage2.metrics.compute_metrics computes AUC only for the multiclass
    one-vs-rest case; for 2 classes sklearn needs the positive-class score as a
    1-D array, so it bails out. Rather than change that shared module (the
    existing Stage 2 reports depend on its behaviour), the binary value is
    filled in here. Positive class = index 1 = Non_larva.
    """
    if y_prob is None:
        return m
    try:
        from sklearn.metrics import roc_auc_score
        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 2 and y_prob.shape[1] == 2 and len(set(y_true)) == 2:
            m["auc_binary"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            m["auc_positive_class"] = CLASS_NAMES[1]
            m.pop("auc_note", None)
    except Exception as exc:
        m["auc_note"] = f"binary AUC skipped: {exc}"
    return m


def train_one(cfg, out_dir, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_model.pth")

    if verbose:
        print(f"\n{'='*78}\nSTAGE 1 ResNet50 — {cfg.name}\n  {cfg.changed}\n{'='*78}")

    set_seed(cfg.seed)
    loaders, ds = data_mod.build_dataloaders(
        img_size=cfg.img_size, batch_size=cfg.batch_size, augment=cfg.augment,
        preprocess=_preprocess_chain(cfg), num_workers=cfg.num_workers,
        seed=cfg.seed, verbose=verbose,
    )

    set_seed(cfg.seed)
    model = model_mod.build_resnet50(dropout=cfg.dropout,
                                     pretrained=cfg.pretrained, device=device)
    model_info = model_mod.verify(model, device, cfg.img_size, verbose=verbose)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    optimizer = build_optimizer(model, cfg.optimizer, cfg.learning_rate,
                                cfg.weight_decay, cfg.momentum)
    scheduler, step_on = build_scheduler(optimizer, cfg.scheduler, cfg.epochs,
                                         cfg.plateau_factor, cfg.plateau_patience)

    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        e0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, tr_f1, *_ = run_epoch(model, loaders["train"], criterion,
                                               optimizer, device, True, scaler)
        va_loss, va_acc, va_f1, *_ = run_epoch(model, loaders["val"], criterion,
                                               None, device, False, scaler)
        if scheduler is not None:
            scheduler.step(va_loss) if step_on == "val_loss" else scheduler.step()

        history.append({"epoch": epoch, "lr": lr_now,
                        "train_loss": tr_loss, "train_acc": tr_acc, "train_f1_macro": tr_f1,
                        "val_loss": va_loss, "val_acc": va_acc, "val_f1_macro": va_f1,
                        "seconds": time.time() - e0})
        improved = va_f1 > best_f1 + 1e-6
        if improved:
            best_f1, best_epoch, stale = va_f1, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale += 1
        if verbose:
            print(f"  epoch {epoch:2d}/{cfg.epochs} lr={lr_now:.2e} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:5.2f}% F1 {tr_f1:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:5.2f}% F1 {va_f1:.4f} | "
                  f"{history[-1]['seconds']:.1f}s{'  *best*' if improved else ''}")
        if cfg.early_stopping and stale >= cfg.early_stopping:
            if verbose:
                print(f"  early stop: no val macro-F1 gain for {cfg.early_stopping} epochs")
            break
    train_seconds = time.time() - t0

    # restore best checkpoint, then evaluate (test touched exactly once)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    ev = {}
    for split in ("val", "test"):
        _, _, _, yt, yp, ypr = run_epoch(model, loaders[split], criterion, None,
                                         device, False, scaler, collect_probs=True)
        ev[split] = metrics_mod.compute_metrics(yt, yp, ypr, CLASS_NAMES)
        _add_binary_auc(ev[split], yt, ypr)
    _, _, _, yt, yp, _ = run_epoch(model, loaders["train_eval"], criterion, None,
                                   device, False, scaler)
    ev["train"] = metrics_mod.compute_metrics(yt, yp, None, CLASS_NAMES)

    record = {
        "run_name": cfg.name, "changed": cfg.changed, "config": asdict(cfg),
        "best_epoch": best_epoch, "best_val_f1_macro": best_f1,
        "epochs_run": len(history), "train_seconds": train_seconds,
        "train_minutes": train_seconds / 60.0,
        "seconds_per_epoch": train_seconds / max(len(history), 1),
        "class_names": CLASS_NAMES,
        "class_mapping": {str(i): n for i, n in enumerate(CLASS_NAMES)},
        "class_dirs": {str(i): d for i, d in enumerate(data_mod.CLASS_DIRS)},
        "split_sizes": {k: len(v) for k, v in ds.items() if k != "train_eval"},
        "checkpoint": ckpt_path.replace("\\", "/"),
        "model": model_info, "metrics": ev,
    }
    _write(out_dir, cfg, record, history)
    if verbose:
        print(f"\n[{cfg.name}] best val macro-F1 {best_f1:.4f} @ epoch {best_epoch}")
        print(metrics_mod.format_metrics("TEST", ev["test"], CLASS_NAMES))
        print(f"[{cfg.name}] {train_seconds/60:.1f} min -> {out_dir}")
    return record


def _write(out_dir, cfg, record, history):
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    fields = ["epoch", "lr", "train_loss", "train_acc", "train_f1_macro",
              "val_loss", "val_acc", "val_f1_macro", "seconds"]
    with open(os.path.join(out_dir, "history.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(fields) + "\n")
        for h in history:
            fh.write(",".join(str(h[f]) for f in fields) + "\n")

    m = record["metrics"]
    lines = [f"Stage 1 ResNet50 — {cfg.name}", f"Changed: {cfg.changed}",
             f"Checkpoint: {record['checkpoint']}", "",
             "CLASS MAPPING", json.dumps(record["class_mapping"], indent=2),
             f"(source folders: {record['class_dirs']})", "",
             "CONFIGURATION", json.dumps(asdict(cfg), indent=2), "",
             f"Split sizes : {record['split_sizes']}",
             f"Epochs run  : {record['epochs_run']} (best epoch {record['best_epoch']})",
             f"Best val macro F1: {record['best_val_f1_macro']:.4f}",
             f"Training time: {record['train_minutes']:.2f} min", ""]
    for s in ("train", "val", "test"):
        lines.append(metrics_mod.format_metrics(s.upper(), m[s], CLASS_NAMES))
        lines.append("")
    t = m["test"]
    lines += ["BINARY RECALL BREAKDOWN (test)",
              f"  {CLASS_NAMES[0]} recall     : {t['per_class'][CLASS_NAMES[0]]['recall']:.4f}",
              f"  {CLASS_NAMES[1]} recall : {t['per_class'][CLASS_NAMES[1]]['recall']:.4f}", ""]
    with open(os.path.join(out_dir, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    _plot_curves(out_dir, cfg, history, record["best_epoch"])
    _plot_cm(out_dir, cfg, m["test"])


def _plot_curves(out_dir, cfg, history, best_epoch):
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[0].plot(ep, [h["val_acc"] for h in history], label="val")
    ax[0].set_title("accuracy (%)")
    ax[1].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[1].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[1].set_title("loss")
    ax[2].plot(ep, [h["train_f1_macro"] for h in history], label="train")
    ax[2].plot(ep, [h["val_f1_macro"] for h in history], label="val")
    ax[2].set_title("macro F1")
    for a in ax:
        a.axvline(best_epoch, color="green", ls=":", lw=1)
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
    fig.suptitle(f"Stage 1 ResNet50 — {cfg.name} (best epoch {best_epoch})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130)
    plt.close(fig)


def _plot_cm(out_dir, cfg, tm):
    cm = np.asarray(tm["confusion_matrix"], float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for a, data, is_norm, title in ((axes[0], cm, False, "counts"),
                                    (axes[1], norm, True, "row-normalised (recall)")):
        a.imshow(data, cmap="Blues", vmin=0, vmax=data.max() or 1)
        a.set_xticks(range(len(CLASS_NAMES))); a.set_yticks(range(len(CLASS_NAMES)))
        a.set_xticklabels(CLASS_NAMES); a.set_yticklabels(CLASS_NAMES)
        a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(title, fontsize=10)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                a.text(j, i, f"{data[i,j]:.3f}" if is_norm else f"{int(cm[i,j])}",
                       ha="center", va="center", fontsize=10,
                       color="white" if data[i, j] > data.max()*0.6 else "black")
    fig.suptitle(f"Stage 1 ResNet50 — {cfg.name} test confusion matrix "
                 f"(acc {tm['accuracy']:.4f}, macro F1 {tm['f1_macro']:.4f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────
# Small, prioritised probe (LR -> dropout -> batch), only if requested
# ─────────────────────────────────────────────
TUNING = [
    ("lr_1e-4", "learning_rate: 1e-3 -> 1e-4", dict(learning_rate=1e-4)),
    ("lr_3e-4", "learning_rate: 1e-3 -> 3e-4", dict(learning_rate=3e-4)),
    ("dropout_02", "dropout: 0.4 -> 0.2", dict(dropout=0.2)),
    ("batch_16", "batch_size: 32 -> 16", dict(batch_size=16)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true",
                    help="after the baseline, run the small LR/dropout/batch probe")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    base = Stage1Config()
    records = {}
    base_dir = os.path.join(OUT_ROOT, "baseline")
    if args.skip_existing and os.path.exists(os.path.join(base_dir, "metrics.json")):
        with open(os.path.join(base_dir, "metrics.json"), encoding="utf-8") as fh:
            records["baseline"] = json.load(fh)
        print("[stage1] baseline already complete, skipping (resume)")
    else:
        records["baseline"] = train_one(base, base_dir)

    if args.tune:
        for name, changed, over in TUNING:
            d = os.path.join(OUT_ROOT, "tuning", name)
            if args.skip_existing and os.path.exists(os.path.join(d, "metrics.json")):
                with open(os.path.join(d, "metrics.json"), encoding="utf-8") as fh:
                    records[name] = json.load(fh)
                print(f"[stage1] {name} already complete, skipping (resume)")
                continue
            try:
                records[name] = train_one(replace(base, name=name, changed=changed, **over), d)
            except Exception as exc:
                print(f"[stage1] {name} FAILED: {type(exc).__name__}: {exc}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "FAILED.json"), "w", encoding="utf-8") as fh:
                    json.dump({"error": f"{type(exc).__name__}: {exc}"}, fh, indent=2)

    # class mapping is a first-class demo artefact
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, "class_mapping.json"), "w", encoding="utf-8") as fh:
        json.dump({str(i): n for i, n in enumerate(CLASS_NAMES)}, fh, indent=2)
    return records


if __name__ == "__main__":
    main()
