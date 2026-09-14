"""
v2 deployable pipeline - the end product.

    from v2.pipeline import classify
    result = classify("photo.jpg")

    python -m v2.pipeline photo1.jpg photo2.jpg      # command line

    Smartphone photo
          |
    Stage 1  ResNet-50 (lab + field)        mosquito / non_mosquito
          |--- non_mosquito -> STOP
          |
    Stage 2  ensemble of 3 CNNs, multi-resolution
          |
    confidence >= threshold ? -> aedes / anopheles / culex / unknown_objects
                              -> otherwise: RETAKE THE PHOTO

Why there is a "retake" answer
------------------------------
A PHI officer is standing next to the specimen. Being told "move closer and
try again" costs ten seconds; a confident wrong genus becomes a wrong
surveillance record. On the held-out field test set the pipeline answers 49%
of photos at 89.9% accuracy, versus 72.2% if it is forced to answer every
time. The threshold is read from the calibration file and can be moved without
retraining - see results/v2/final/v2_results.json for the full
risk-coverage curve.

Everything here loads from results/v2/. Nothing is hard-coded: the ensemble
membership, the input resolutions and the abstention threshold all come from
the calibration produced by v2.evaluate.
"""

import json
import os
import sys
import threading
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

Image.MAX_IMAGE_PIXELS = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "v2")
CALIB = os.path.join(RESULTS, "final", "v2_results.json")
S1_CKPT = os.path.join(RESULTS, "stage1_resnet50", "best_model.pth")

S1_CLASSES = ["mosquito", "non_mosquito"]
S2_CLASSES = ["aedes", "anopheles", "culex", "unknown_objects"]
DISPLAY = {"aedes": "Aedes", "anopheles": "Anopheles", "culex": "Culex",
           "unknown_objects": "Unknown object"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
S1_SIZE = 224
BACKBONES = {
    "efficientnet_b0": torchvision.models.efficientnet_b0,
    "efficientnet_v2_s": torchvision.models.efficientnet_v2_s,
    "convnext_tiny": torchvision.models.convnext_tiny,
}

_STATE = None
_LOCK = threading.Lock()


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prep(img, size):
    import torchvision.transforms.functional as TF
    im = img.convert("RGB").resize((size, size), Image.BILINEAR)
    return TF.normalize(TF.to_tensor(im), IMAGENET_MEAN, IMAGENET_STD)


def load():
    """Load Stage 1, the Stage 2 ensemble and the calibration. Cached."""
    global _STATE
    with _LOCK:
        if _STATE is not None:
            return _STATE
        dev = _device()
        if not os.path.exists(CALIB):
            raise FileNotFoundError("calibration missing: " + CALIB
                                    + " (run: python -m v2.evaluate)")
        cal = json.load(open(CALIB, encoding="utf-8"))

        # LITE mode: one backbone instead of three. 196 MB instead of 352 MB
        # and ~3x faster, for 0.003 field macro-F1. Selected with
        # SAFEZONE_MODE=lite. Its abstention threshold is calibrated
        # separately (v2.calibrate_lite) because a single model's confidence
        # distribution is not the ensemble's.
        # A packaged build bakes its choice into the calibration file, so it
        # needs no environment variable; report that mode rather than the
        # env default, or a lite package would mislabel itself "full".
        mode = os.environ.get("SAFEZONE_MODE",
                              cal.get("packaged_mode", "full")).lower()
        if mode == "lite" and "lite" in cal:
            lite = cal["lite"]
            cal = dict(cal)
            cal["ensemble"] = lite["ensemble"]
            cal["use_tta"] = lite["use_tta"]
            cal["abstention_threshold"] = lite["abstention_threshold"]
            cal["mode"] = "lite"
        else:
            cal = dict(cal)
            cal["mode"] = cal.get("packaged_mode", "full")

        from stage1 import model as s1m
        s1 = s1m.build_resnet50(num_classes=2, dropout=0.4, pretrained=False,
                                device=dev)
        s1.load_state_dict(torch.load(S1_CKPT, map_location=dev,
                                      weights_only=True), strict=True)
        s1.eval()

        runs = cal["available_runs"]
        members = []
        for name in cal["ensemble"]:
            info = runs[name]
            m = BACKBONES[info["backbone"]](weights=None)
            cls = m.classifier
            for i in range(len(cls) - 1, -1, -1):
                if isinstance(cls[i], nn.Linear):
                    cls[i] = nn.Linear(cls[i].in_features, len(S2_CLASSES))
                    break
            # The calibration file was written on Windows, so its paths carry
            # backslashes. os.path.join would not resolve those on macOS or
            # Linux, which is where teammates may run this.
            rel = info["ckpt"].replace("\\", "/")
            ck = rel if os.path.isabs(rel) else os.path.join(ROOT, *rel.split("/"))
            m.load_state_dict(torch.load(ck, map_location=dev,
                                         weights_only=True), strict=True)
            members.append({"name": name, "model": m.to(dev).eval(),
                            "size": info["image_size"],
                            "backbone": info["backbone"]})
        _STATE = {"device": dev, "stage1": s1, "stage2": members,
                  "mode": cal.get("mode", "full"),
                  "threshold": float(cal["abstention_threshold"]),
                  "use_tta": bool(cal["use_tta"]), "calibration": cal}
        return _STATE


def _tta_views(img, use_tta):
    views = [img]
    if use_tta:
        views.append(img.transpose(Image.FLIP_LEFT_RIGHT))
        views.append(img.resize((int(img.width * 1.12), int(img.height * 1.12)),
                                Image.BILINEAR))
    return views


def classify(source, threshold=None, force_answer=False):
    """
    Run the full pipeline on one image.

    threshold      override the calibrated abstention threshold
    force_answer   return the best guess even when below threshold
                   (the confidence and `abstained` flag are still reported)
    """
    st = load()
    dev = st["device"]
    t0 = time.perf_counter()
    img = (source if isinstance(source, Image.Image)
           else Image.open(source)).convert("RGB")

    with torch.no_grad():
        p1 = torch.softmax(
            st["stage1"](_prep(img, S1_SIZE).unsqueeze(0).to(dev)).float(),
            1)[0].cpu().numpy()
    i1 = int(p1.argmax())
    stage1 = {"model": "ResNet-50 (v2, lab+field)",
              "class": S1_CLASSES[i1], "confidence": float(p1[i1]),
              "probabilities": {c: float(v) for c, v in zip(S1_CLASSES, p1)},
              "is_mosquito": S1_CLASSES[i1] == "mosquito"}

    if not stage1["is_mosquito"]:
        return {"stage1": stage1, "stage2": None, "stage2_executed": False,
                "final_class": "non_mosquito",
                "final_label": "Not a mosquito larva",
                "confidence": stage1["confidence"], "abstained": False,
                "message": "No mosquito larva detected — Stage 2 not executed.",
                "total_ms": (time.perf_counter() - t0) * 1000}

    acc = np.zeros(len(S2_CLASSES), np.float64)
    n = 0
    for mem in st["stage2"]:
        for v in _tta_views(img, st["use_tta"]):
            with torch.no_grad():
                q = torch.softmax(
                    mem["model"](_prep(v, mem["size"]).unsqueeze(0).to(dev)).float(),
                    1)[0].cpu().numpy()
            acc += q
            n += 1
    probs = acc / max(n, 1)
    i2 = int(probs.argmax())
    conf = float(probs[i2])
    thr = st["threshold"] if threshold is None else float(threshold)
    abstain = conf < thr

    stage2 = {"model": "ensemble: " + ", ".join(
                  "%s@%dpx" % (m["backbone"], m["size"]) for m in st["stage2"]),
              "class": S2_CLASSES[i2], "class_index": i2, "confidence": conf,
              "probabilities": {c: float(v) for c, v in zip(S2_CLASSES, probs)},
              "threshold": thr, "abstained": bool(abstain)}

    if abstain and not force_answer:
        return {"stage1": stage1, "stage2": stage2, "stage2_executed": True,
                "final_class": "uncertain", "final_label": "Uncertain",
                "confidence": conf, "abstained": True,
                "best_guess": S2_CLASSES[i2],
                "message": ("Not confident enough (%.0f%% < %.0f%%). Move closer, "
                            "steady the phone and retake the photo."
                            % (100 * conf, 100 * thr)),
                "total_ms": (time.perf_counter() - t0) * 1000}

    return {"stage1": stage1, "stage2": stage2, "stage2_executed": True,
            "final_class": S2_CLASSES[i2],
            "final_label": DISPLAY[S2_CLASSES[i2]],
            "confidence": conf, "abstained": bool(abstain),
            "message": "Classified as %s (%.0f%% confidence)."
                       % (DISPLAY[S2_CLASSES[i2]], 100 * conf),
            "total_ms": (time.perf_counter() - t0) * 1000}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python -m v2.pipeline <image> [<image> ...]")
        return 1
    st = load()
    print("Stage 1 : ResNet-50 (v2)")
    print("Stage 2 : " + ", ".join("%s@%dpx" % (m["backbone"], m["size"])
                                   for m in st["stage2"]))
    print("abstain below confidence %.2f   TTA=%s\n"
          % (st["threshold"], st["use_tta"]))
    for p in sys.argv[1:]:
        r = classify(p)
        s1 = r["stage1"]
        print(os.path.basename(p))
        print("  Stage 1 : %-13s %.3f" % (s1["class"], s1["confidence"]))
        if r["stage2_executed"]:
            s2 = r["stage2"]
            top = sorted(s2["probabilities"].items(), key=lambda kv: -kv[1])[:2]
            print("  Stage 2 : %-13s %.3f   (next: %s %.3f)"
                  % (s2["class"], s2["confidence"], top[1][0], top[1][1]))
        else:
            print("  Stage 2 : not executed (gate)")
        print("  RESULT  : %s" % r["message"])
        print("  %.0f ms\n" % r["total_ms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
