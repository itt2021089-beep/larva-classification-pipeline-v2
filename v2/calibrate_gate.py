"""
Choose Stage 1's rejection threshold on validation.

    python -m v2.calibrate_gate

Writes a "gate" block into results/v2/final/v2_results.json, which
v2.pipeline reads. Uses validation only; test is never touched here.

The problem
-----------
The pipeline first shipped with a plain argmax gate: Stage 1 rejected an image
whenever p(mosquito) < 0.5, and a rejection was always delivered as a
confident "not a mosquito larva". It could never say "retake". Measured end to
end on the field test set, that gate rejected 12 of 289 real larvae — each
one a confident wrong answer that bypassed the abstention the rest of the
pipeline relies on.

The fix
-------
v2.pipeline.gate_decision adds a deferral band: when reject_below <=
p(mosquito) < 0.5, Stage 1 is unsure, and Stage 2 decides instead — with its
own unknown_objects class and its own calibrated abstention.

The rule, fixed before looking at the outcome
---------------------------------------------
Every answer the app gives should be held to the same bar. Stage 2's
threshold was chosen so that its genus answers are right at least 90% of the
time on field validation. So Stage 1's "not a larva" answers must also be
right at least 90% of the time on field validation:

    rejection precision = (true unknown_objects among rejected) / (rejected)

Among thresholds meeting that bar, take the LARGEST — the one closest to the
original gate, deferring as few images as necessary. If no threshold with at
least one rejection meets it, reject_below = 0 and Stage 2 decides everything.

Field validation is used, as for Stage 2's threshold, because field
performance is the objective; laboratory validation is reported alongside for
information and does not select. The rule depends on Stage 1 alone, so one
threshold serves both the full and the lite builds.
"""

import csv
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2 import pipeline  # noqa: E402

VAL = os.path.join("final_datasets_v2", "manifests", "val.csv")
CAL = pipeline.CALIB
TARGET_PRECISION = 0.90
GRID = sorted({round(0.01 * i, 2) for i in range(0, 51)}
              | {0.001, 0.002, 0.005}, reverse=True)
SPECIES = {"aedes", "anopheles", "culex"}


def sweep(recs, thr):
    rows = []
    for t in GRID:
        rej = [r for r in recs if r["pm"] < t]
        dfr = [r for r in recs if t <= r["pm"] < 0.5]
        ok_rej = sum(1 for r in rej if r["true"] == "unknown_objects")
        # end-to-end outcome at this gate, for information
        n_ok = n_ans = n_ok_ans = 0
        for r in recs:
            dec, ans, answered, ok = _outcome(r, t, thr)
            n_ok += ok
            n_ans += answered
            n_ok_ans += ok and answered
        rows.append({
            "reject_below": t,
            "rejected": len(rej),
            "rejection_precision": (ok_rej / len(rej)) if rej else None,
            "larvae_rejected": sum(1 for r in rej if r["true"] in SPECIES),
            "deferred": len(dfr),
            "pipeline_accuracy": n_ok / len(recs),
            "pipeline_coverage": n_ans / len(recs),
            "pipeline_confident_accuracy": n_ok_ans / max(n_ans, 1)})
    return rows


def _outcome(r, t, thr):
    dec = pipeline.gate_decision(r["pm"], t)
    if dec == "reject":
        ans, answered = "non_mosquito", True
    else:
        ans, answered = r["s2"], r["s2_conf"] >= thr
    if r["true"] in SPECIES:
        ok = ans == r["true"]
    else:
        ok = ans in ("non_mosquito", "unknown_objects")
    return dec, ans, answered, ok


def main():
    st = pipeline.load()
    thr = st["threshold"]
    rows = list(csv.DictReader(open(VAL, encoding="utf-8")))
    recs = {"field": [], "lab": []}
    for row in rows:
        img = Image.open(row["final_path"]).convert("RGB")
        p1 = pipeline.stage1_probs(st, img)
        p2 = pipeline.stage2_probs(st, img)
        i2 = int(p2.argmax())
        recs[row["domain"]].append({
            "true": row["class"],
            "pm": float(p1[pipeline.S1_CLASSES.index("mosquito")]),
            "s2": pipeline.S2_CLASSES[i2], "s2_conf": float(p2[i2])})
    print("validation: field %d, lab %d   (Stage 2 threshold %.2f, mode %s)"
          % (len(recs["field"]), len(recs["lab"]), thr, st["mode"]))

    field = sweep(recs["field"], thr)
    lab = sweep(recs["lab"], thr)

    feasible = [r for r in field if r["rejected"] > 0
                and r["rejection_precision"] >= TARGET_PRECISION]
    chosen = max(feasible, key=lambda r: r["reject_below"]) if feasible else \
        next(r for r in field if r["reject_below"] == 0.0)
    t = chosen["reject_below"]
    orig = next(r for r in field if r["reject_below"] == 0.5)
    lab_at = next(r for r in lab if r["reject_below"] == t)
    lab_orig = next(r for r in lab if r["reject_below"] == 0.5)

    print("\n reject_below  rejected  precision  larvae_rej  deferred   "
          "all    confident @ coverage   (field validation)")
    for r in field:
        if r["reject_below"] in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01,
                                 0.005, 0.002, 0.001, 0.0) or r is chosen:
            print("   %6.3f      %4d      %s     %4d       %4d    %5.1f%%   "
                  "%5.1f%% @ %5.1f%%%s"
                  % (r["reject_below"], r["rejected"],
                     ("%5.1f%%" % (100 * r["rejection_precision"]))
                     if r["rejection_precision"] is not None else "   -  ",
                     r["larvae_rejected"], r["deferred"],
                     100 * r["pipeline_accuracy"],
                     100 * r["pipeline_confident_accuracy"],
                     100 * r["pipeline_coverage"],
                     "   <- chosen" if r is chosen else ""))

    print("\nchosen reject_below = %.3f" % t)
    print("  field val  rejection precision %s -> %s, larvae rejected %d -> %d"
          % ("%.1f%%" % (100 * orig["rejection_precision"]),
             ("%.1f%%" % (100 * chosen["rejection_precision"]))
             if chosen["rejection_precision"] is not None else "n/a",
             orig["larvae_rejected"], chosen["larvae_rejected"]))
    print("  field val  pipeline confident %.1f%% @ %.1f%% -> %.1f%% @ %.1f%%"
          % (100 * orig["pipeline_confident_accuracy"],
             100 * orig["pipeline_coverage"],
             100 * chosen["pipeline_confident_accuracy"],
             100 * chosen["pipeline_coverage"]))
    print("  lab val    pipeline all %.1f%% -> %.1f%%  (information only)"
          % (100 * lab_orig["pipeline_accuracy"], 100 * lab_at["pipeline_accuracy"]))

    cal = json.load(open(CAL, encoding="utf-8"))
    cal["gate"] = {
        "reject_below": t,
        "rule": ("largest reject_below whose Stage 1 rejections are correct "
                 ">= %.0f%% of the time on field validation — the same bar "
                 "as Stage 2's genus answers" % (100 * TARGET_PRECISION)),
        "chosen_on": "field validation",
        "target_rejection_precision": TARGET_PRECISION,
        "stage2_threshold_at_calibration": thr,
        "field_validation": {"n": len(recs["field"]), "chosen": chosen,
                             "original_gate": orig, "sweep": field},
        "lab_validation": {"n": len(recs["lab"]), "chosen": lab_at,
                           "original_gate": lab_orig},
    }
    json.dump(cal, open(CAL, "w", encoding="utf-8"), indent=2)
    print("\nwrote gate block to", CAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
