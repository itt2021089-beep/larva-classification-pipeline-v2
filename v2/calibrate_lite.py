"""
Calibrate an abstention threshold for LITE mode (single backbone).

    python -m v2.calibrate_lite

The deployed app can run in two modes:

    full  convnext_tiny + efficientnet_v2_s + efficientnet_v2_s@320   352 MB
    lite  convnext_tiny only                                          196 MB

Lite exists because 352 MB of checkpoints is awkward to host for free and
because a single 106 MB file already exceeds GitHub's per-file limit. On field
validation the two are within 0.003 macro-F1 of each other, so lite is a
reasonable deployment choice - but its confidence distribution is NOT the
ensemble's, so reusing the ensemble threshold would silently change the
operating point. This computes lite's own threshold on the same field
validation subset and writes it alongside the ensemble's.

Validation only. The test splits are not touched.
"""

import json
import os
import sys

import numpy as np

from v2.evaluate import (CLASSES, discover, load_backbone, probs_for,
                         risk_coverage, rows_of, score)

CALIB = os.path.join("results", "v2", "final", "v2_results.json")


def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cal = json.load(open(CALIB, encoding="utf-8"))
    avail = discover()

    lite_name = "convnext_tiny"
    if lite_name not in avail:
        sys.exit("lite backbone %s not found" % lite_name)
    info = avail[lite_name]
    print("lite model: %s (%s @ %dpx)"
          % (lite_name, info["backbone"], info["image_size"]))

    val = rows_of("val")
    y = [CLASSES.index(r["class"]) for r in val]
    field = [r["domain"] == "field" for r in val]
    fv = [i for i, m in enumerate(field) if m]

    m = (load_backbone(info["backbone"], info["ckpt"], device), info["image_size"])
    p = probs_for([m], val, device, tta=False)
    s = score(y, p.argmax(1), field)
    print("  field val: macroF1 %.4f  species-acc %.4f"
          % (s["f1_macro"], s["species_accuracy"]))

    rc = risk_coverage([y[i] for i in fv], p[fv])
    target, min_cov = 0.90, 0.70
    chosen = None
    for row in rc:
        if row["accuracy_on_answered"] >= target and row["coverage"] >= min_cov:
            chosen = row
            break
    if chosen is None:
        feasible = [r for r in rc if r["accuracy_on_answered"] >= target]
        chosen = (max(feasible, key=lambda r: r["coverage"]) if feasible
                  else max(rc, key=lambda r: r["accuracy_on_answered"]))
    print("  chosen threshold %.2f -> coverage %.1f%%, accuracy on answered %.1f%%"
          % (chosen["threshold"], 100 * chosen["coverage"],
             100 * chosen["accuracy_on_answered"]))

    cal["lite"] = {
        "ensemble": [lite_name],
        "use_tta": False,
        "abstention_threshold": chosen["threshold"],
        "validation_field": s,
        "calibration_row": chosen,
        "risk_coverage_field_validation": rc,
        "note": ("Single-backbone deployment mode. Threshold calibrated on the "
                 "SAME field validation subset as the ensemble, because lite's "
                 "confidence distribution differs. Test splits untouched."),
    }
    json.dump(cal, open(CALIB, "w", encoding="utf-8"), indent=2)
    print("\n[out] " + CALIB + "  (added 'lite' block)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
