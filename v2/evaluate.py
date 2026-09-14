"""
v2 step 4 - ensemble, calibrate abstention on validation, then evaluate once.

    python -m v2.evaluate

Everything that could be tuned is tuned on VALIDATION:
  * which backbones enter the ensemble,
  * whether test-time augmentation helps,
  * the confidence threshold below which the app says "retake the photo".

Only then are test_field and test_lab scored, once.

Why abstention matters here
---------------------------
A PHI officer is standing next to the specimen. If the model says "not sure,
move closer and retake", that costs ten seconds. If it confidently says Culex
when the larva is Anopheles, that is a wrong surveillance record. So the
deployable quantity is not raw accuracy but accuracy AT A COVERAGE LEVEL -
"on the 80% of photos it will answer, it is right 93% of the time" is both
more useful and more honest than a single headline number.

The risk-coverage curve is reported in full so the operating point can be
moved without re-running anything.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

Image.MAX_IMAGE_PIXELS = None

DATA = "final_datasets_v2"
ROOT = os.path.join("results", "v2")
OUT = os.path.join("results", "v2", "final")
CLASSES = ["aedes", "anopheles", "culex", "unknown_objects"]
SPECIES = ["aedes", "anopheles", "culex"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BACKBONES = {
    "efficientnet_b0": torchvision.models.efficientnet_b0,
    "efficientnet_v2_s": torchvision.models.efficientnet_v2_s,
    "convnext_tiny": torchvision.models.convnext_tiny,
}


def discover():
    """
    Find every trained v2 run: directory name is free-form (e.g.
    convnext_tiny_320), so the backbone and the input resolution are read from
    each run's own metadata rather than inferred from the folder name. Models
    trained at different resolutions can therefore be ensembled - each is fed
    the size it was trained at.
    """
    out = {}
    for d in sorted(os.listdir(ROOT)):
        ck = os.path.join(ROOT, d, "best_model.pth")
        mj = os.path.join(ROOT, d, "metadata.json")
        if not (os.path.exists(ck) and os.path.exists(mj)):
            continue
        m = json.load(open(mj, encoding="utf-8"))
        bb = m.get("backbone")
        if bb not in BACKBONES:
            continue
        out[d] = {"ckpt": ck, "backbone": bb,
                  "image_size": int(m.get("config", {}).get("image_size", 224))}
    return out


def load_backbone(name, ckpt, device):
    m = BACKBONES[name](weights=None)
    cls = m.classifier
    for i in range(len(cls) - 1, -1, -1):
        if isinstance(cls[i], nn.Linear):
            cls[i] = nn.Linear(cls[i].in_features, len(CLASSES))
            break
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True),
                      strict=True)
    return m.to(device).eval()


def rows_of(split):
    return list(csv.DictReader(open(os.path.join(DATA, "manifests", split + ".csv"),
                                    encoding="utf-8")))


def probs_for(models, rows, device, tta=True, batch=24):
    """
    Mean softmax over models and (optionally) TTA views.

    `models` is a list of (module, input_size) so a 224px and a 320px model can
    sit in the same ensemble; each sees the resolution it was trained for.
    """
    import torchvision.transforms.functional as TF
    views = [lambda im: im]
    if tta:
        views += [lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
                  lambda im: im.resize((int(im.width * 1.12), int(im.height * 1.12)),
                                       Image.BILINEAR)]
    acc = np.zeros((len(rows), len(CLASSES)), np.float64)
    n_passes = 0
    for model, size in models:
        for v in views:
            n_passes += 1
            buf, idx = [], []

            def flush():
                if not buf:
                    return
                x = torch.stack(buf).to(device)
                with torch.no_grad():
                    p = torch.softmax(model(x).float(), 1).cpu().numpy()
                for k, i in enumerate(idx):
                    acc[i] += p[k]
                buf.clear(); idx.clear()

            for i, r in enumerate(rows):
                im = Image.open(r["final_path"]).convert("RGB")
                im = v(im).resize((size, size), Image.BILINEAR)
                buf.append(TF.normalize(TF.to_tensor(im), IMAGENET_MEAN,
                                        IMAGENET_STD))
                idx.append(i)
                if len(buf) >= batch:
                    flush()
            flush()
    return acc / max(n_passes, 1)


def score(y, p, mask=None):
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
    sp = [CLASSES.index(c) for c in SPECIES]
    si = [i for i, t in enumerate(y) if t in sp]
    return {"n": len(y), "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "f1_macro": float(fm),
            "species_n": len(si),
            "species_accuracy": float(sum(1 for i in si if y[i] == p[i]) / len(si)) if si else 0.0,
            "per_class": {CLASSES[i]: {"precision": float(pr[i]), "recall": float(rc[i]),
                                       "f1": float(f1[i]), "support": int(s[i])}
                          for i in range(len(CLASSES))},
            "confusion_matrix": confusion_matrix(
                y, p, labels=list(range(len(CLASSES)))).tolist()}


def risk_coverage(y, probs, grid=None):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = np.array([int(a == b) for a, b in zip(pred, y)])
    out = []
    for t in (grid if grid is not None else np.arange(0.0, 0.991, 0.01)):
        keep = conf >= t
        n = int(keep.sum())
        if n == 0:
            continue
        out.append({"threshold": round(float(t), 3),
                    "coverage": round(n / len(y), 4),
                    "n_answered": n,
                    "accuracy_on_answered": round(float(correct[keep].mean()), 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-accuracy", type=float, default=0.90)
    ap.add_argument("--min-coverage", type=float, default=0.70)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avail = discover()
    if not avail:
        sys.exit("no trained v2 runs found under " + ROOT)
    print("available runs:")
    for k, v in avail.items():
        print("   %-26s %s @ %dpx" % (k, v["backbone"], v["image_size"]))

    val = rows_of("val")
    y_val = [CLASSES.index(r["class"]) for r in val]
    field_val = [r["domain"] == "field" for r in val]

    # ---- 1. choose the ensemble on VALIDATION (field subset) --------
    print("\n" + "=" * 78)
    print("STEP 1 - ensemble selection on FIELD validation")
    print("=" * 78)
    singles = {}
    for name, info in avail.items():
        m = (load_backbone(info["backbone"], info["ckpt"], device),
             info["image_size"])
        p = probs_for([m], val, device, tta=False)
        s = score(y_val, p.argmax(1), field_val)
        singles[name] = {"model": m, "probs": p, "score": s}
        print("  %-26s field macroF1 %.4f  species-acc %.4f"
              % (name, s["f1_macro"], s["species_accuracy"]))

    best_combo, best_f1 = None, -1
    import itertools
    for r in range(1, len(avail) + 1):
        for combo in itertools.combinations(sorted(avail), r):
            p = np.mean([singles[n]["probs"] for n in combo], axis=0)
            s = score(y_val, p.argmax(1), field_val)
            if s["f1_macro"] > best_f1:
                best_f1, best_combo = s["f1_macro"], combo
            if r > 1 and s["f1_macro"] > best_f1 - 0.02:
                print("  %-44s field macroF1 %.4f"
                      % ("+".join(combo)[:44], s["f1_macro"]))
    print("\n  chosen ensemble: %s  (field val macroF1 %.4f)"
          % ("+".join(best_combo), best_f1))
    models = [singles[n]["model"] for n in best_combo]

    # ---- 2. does TTA help? decided on validation -------------------
    print("\n" + "=" * 78)
    print("STEP 2 - test-time augmentation, decided on FIELD validation")
    print("=" * 78)
    p_no = np.mean([singles[n]["probs"] for n in best_combo], axis=0)
    p_tta = probs_for(models, val, device, tta=True)
    s_no = score(y_val, p_no.argmax(1), field_val)
    s_tta = score(y_val, p_tta.argmax(1), field_val)
    use_tta = s_tta["f1_macro"] >= s_no["f1_macro"]
    print("  without TTA: macroF1 %.4f   with TTA: macroF1 %.4f  -> %s"
          % (s_no["f1_macro"], s_tta["f1_macro"], "USE TTA" if use_tta else "no TTA"))
    p_val = p_tta if use_tta else p_no

    # ---- 3. abstention threshold, calibrated on validation ---------
    print("\n" + "=" * 78)
    print("STEP 3 - abstention threshold from FIELD validation")
    print("=" * 78)
    fv = [i for i, m in enumerate(field_val) if m]
    rc = risk_coverage([y_val[i] for i in fv], p_val[fv])
    chosen = None
    for row in rc:
        if (row["accuracy_on_answered"] >= a.target_accuracy
                and row["coverage"] >= a.min_coverage):
            chosen = row
            break
    if chosen is None:
        feasible = [r for r in rc if r["accuracy_on_answered"] >= a.target_accuracy]
        chosen = (max(feasible, key=lambda r: r["coverage"]) if feasible
                  else max(rc, key=lambda r: r["accuracy_on_answered"]))
    print("  target >= %.0f%% accuracy at >= %.0f%% coverage"
          % (100 * a.target_accuracy, 100 * a.min_coverage))
    print("  chosen threshold %.2f -> coverage %.1f%%, accuracy on answered %.1f%%"
          % (chosen["threshold"], 100 * chosen["coverage"],
             100 * chosen["accuracy_on_answered"]))
    thr = chosen["threshold"]

    # ---- 4. TEST, once ---------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 4 - LOCKED TEST EVALUATION (once, after everything above)")
    print("=" * 78)
    results = {}
    for split in ("test_field", "test_lab"):
        rows = rows_of(split)
        if not rows:
            continue
        y = [CLASSES.index(r["class"]) for r in rows]
        p = probs_for(models, rows, device, tta=use_tta)
        full = score(y, p.argmax(1))
        conf = p.max(1)
        keep = conf >= thr
        gated = score([y[i] for i in range(len(y)) if keep[i]],
                      [int(p[i].argmax()) for i in range(len(y)) if keep[i]])
        results[split] = {"full": full, "gated": gated,
                          "coverage": float(keep.mean()),
                          "threshold": thr,
                          "risk_coverage": risk_coverage(y, p)}
        print("\n  %s  (n=%d)" % (split, len(rows)))
        print("    ALL images     : acc %.4f  macroF1 %.4f  balacc %.4f  species-acc %.4f"
              % (full["accuracy"], full["f1_macro"], full["balanced_accuracy"],
                 full["species_accuracy"]))
        if gated:
            print("    with abstention: coverage %.1f%%  acc %.4f  macroF1 %.4f  species-acc %.4f"
                  % (100 * keep.mean(), gated["accuracy"], gated["f1_macro"],
                     gated["species_accuracy"]))
        for c in CLASSES:
            pc = full["per_class"][c]
            print("      %-16s P %.3f R %.3f F1 %.3f n=%d"
                  % (c, pc["precision"], pc["recall"], pc["f1"], pc["support"]))

    out = {"ensemble": list(best_combo), "use_tta": bool(use_tta),
           "abstention_threshold": thr,
           "calibration": {"target_accuracy": a.target_accuracy,
                           "min_coverage": a.min_coverage,
                           "chosen_on": "field validation", "row": chosen},
           "validation_field": {"single": {n: singles[n]["score"] for n in singles},
                                "ensemble_no_tta": s_no, "ensemble_tta": s_tta},
           "test": results,
           "checkpoints": {n: avail[n] for n in best_combo},
           "available_runs": avail,
           "note": "Ensemble membership, TTA and the abstention threshold were "
                   "all chosen on validation. Test was scored once afterwards."}
    json.dump(out, open(os.path.join(OUT, "v2_results.json"), "w",
                        encoding="utf-8"), indent=2)
    print("\n[out] " + os.path.join(OUT, "v2_results.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
