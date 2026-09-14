"""
Verify the trained Stage 1 model is genuinely usable by the demo.

    python -m stage1.demo_test

Runs the checks in Step 12 of the task and writes
`results/stage1/resnet50/demo_test_results.csv`:

    image, true_class, predicted_class, confidence, correct

Checks performed:
  1. checkpoint loads
  2. inference runs on real held-out TEST images
  3. predictions are valid class names
  4. probabilities sum to ~1
  5. JPG and PNG both work
  6. CPU inference works
  7. GPU inference works (if available) and agrees with CPU
  8. confidence values are in [0, 1] and equal the max probability
  9. class mapping matches the training dataset ordering
 10. no dataset file was modified (SHA-256 of a sample verified before/after)
"""

import csv
import hashlib
import json
import os
import random

import numpy as np
import torch

from stage1 import dataset as data_mod
from stage1 import inference as inf

OUT_DIR = os.path.join("results", "stage1", "resnet50")
CSV_PATH = os.path.join(OUT_DIR, "demo_test_results.csv")
N_SAMPLE = 40


def _sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    checks, failures = [], []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    rows = data_mod.load_manifest()
    test_rows = [r for r in rows if r["split"] == "test"]
    rng = random.Random(42)

    # stratified sample, and make sure PNGs are represented if any exist
    by_cls = {}
    for r in test_rows:
        by_cls.setdefault(r["class_dir"], []).append(r)
    sample = []
    for cls, items in by_cls.items():
        rng.shuffle(items)
        sample += items[:N_SAMPLE // 2]
    pngs = [r for r in test_rows if r["path"].lower().endswith(".png")]
    for p in pngs[:4]:
        if p not in sample:
            sample.append(p)
    rng.shuffle(sample)

    print(f"\n[demo_test] {len(sample)} test images sampled "
          f"({sum(1 for r in sample if r['path'].lower().endswith('.png'))} PNG)")

    before = {r["path"]: _sha(r["path"]) for r in sample}

    # ---- 1. checkpoint loads ----
    print("\n[demo_test] checks")
    try:
        model, meta = inf.load_model(verbose=False)
        check("1. checkpoint loads", True, meta["checkpoint"])
    except Exception as exc:
        check("1. checkpoint loads", False, str(exc))
        raise SystemExit("cannot continue without a checkpoint")

    # ---- 9. class mapping matches training ordering ----
    map_path = os.path.join(OUT_DIR, "class_mapping.json")
    mapping = json.load(open(map_path, encoding="utf-8")) if os.path.exists(map_path) else {}
    expected = {str(i): n for i, n in enumerate(data_mod.CLASS_NAMES)}
    check("9. class mapping matches dataset ordering",
          mapping == expected and meta["class_names"] == data_mod.CLASS_NAMES,
          f"{mapping} (dirs {data_mod.CLASS_DIRS})")

    # ---- 2,3,4,8: run inference ----
    results, prob_sums, bad_names, bad_conf = [], [], 0, 0
    for r in sample:
        out = inf.predict_stage1(r["path"])
        true_name = data_mod.CLASS_NAMES[r["label"]]
        s = sum(out["probabilities"].values())
        prob_sums.append(s)
        if out["class"] not in data_mod.CLASS_NAMES:
            bad_names += 1
        if not (0.0 <= out["confidence"] <= 1.0) or \
           abs(out["confidence"] - max(out["probabilities"].values())) > 1e-6:
            bad_conf += 1
        results.append({
            "image": r["path"], "true_class": true_name,
            "predicted_class": out["class"],
            "confidence": round(out["confidence"], 6),
            "correct": int(out["class"] == true_name),
        })
    check("2. inference runs on test images", len(results) == len(sample),
          f"{len(results)} predictions")
    check("3. predictions are valid class names", bad_names == 0)
    check("4. probabilities sum to ~1",
          all(abs(s - 1.0) < 1e-4 for s in prob_sums),
          f"min={min(prob_sums):.6f} max={max(prob_sums):.6f}")
    check("8. confidence valid and equals max probability", bad_conf == 0)

    # ---- 5. JPG and PNG ----
    jpg = [r for r in results if r["image"].lower().endswith((".jpg", ".jpeg"))]
    png = [r for r in results if r["image"].lower().endswith(".png")]
    check("5. JPG and PNG both work",
          len(jpg) > 0 and (len(png) > 0 or len(pngs) == 0),
          f"{len(jpg)} JPG, {len(png)} PNG in sample")

    # ---- 6/7. CPU and GPU ----
    probe = sample[0]["path"]
    inf._MODEL, inf._META = None, None                 # force a clean CPU load
    cpu_out = inf.predict_stage1(probe, device="cpu")
    check("6. CPU inference works", cpu_out["class"] in data_mod.CLASS_NAMES,
          f"{cpu_out['class']} {cpu_out['confidence']:.4f}")
    if torch.cuda.is_available():
        inf._MODEL, inf._META = None, None
        gpu_out = inf.predict_stage1(probe, device="cuda")
        agree = (gpu_out["class"] == cpu_out["class"] and
                 abs(gpu_out["confidence"] - cpu_out["confidence"]) < 1e-2)
        check("7. GPU inference works and agrees with CPU", agree,
              f"cpu={cpu_out['confidence']:.4f} gpu={gpu_out['confidence']:.4f}")
    else:
        check("7. GPU inference", True, "skipped — CUDA not available")

    # ---- 10. dataset untouched ----
    after = {p: _sha(p) for p in before}
    check("10. no dataset file modified", before == after,
          f"{len(before)} files hashed before/after")

    # model cached (not reloaded per image)
    inf._MODEL, inf._META = None, None
    inf.load_model(verbose=False)
    first = id(inf._MODEL)
    inf.predict_stage1(probe); inf.predict_stage1(probe)
    check("bonus: model loaded once (cached across calls)", id(inf._MODEL) == first)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "true_class", "predicted_class",
                                           "confidence", "correct"])
        w.writeheader()
        w.writerows(results)

    acc = sum(r["correct"] for r in results) / max(len(results), 1)
    print(f"\n[demo_test] sample accuracy {acc:.4f} ({sum(r['correct'] for r in results)}"
          f"/{len(results)}) — this is a SPOT CHECK, not the official test metric")
    print(f"[demo_test] wrote {CSV_PATH}")
    print(f"[demo_test] {len(checks)-len(failures)}/{len(checks)} checks passed")
    if failures:
        print(f"[demo_test] FAILED: {failures}")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
