"""
End-to-end evaluation of the deployed two-stage pipeline.

    python -m v2.eval_end_to_end                        # full ensemble
    SAFEZONE_MODE=lite python -m v2.eval_end_to_end     # lite build

Results are written to results/v2/final/end_to_end.json under the mode's own
key, so the full and lite builds each report their own numbers. The app and
the package README read them from there.

Why this exists
---------------
v2.evaluate scores the Stage 2 ensemble on every test image. It never runs
Stage 1, so its headline numbers describe the species classifier in isolation.
The number that answers "does the system work?" is the pipeline's: an image
first has to get past the Stage 1 gate, and a real larva that Stage 1 rejects
is lost no matter how good Stage 2 is.

Three views are reported from the same forward passes:

  stage2_only   Stage 2 on every image. Must reproduce v2.evaluate exactly,
                which is what validates this script.
  hard_gate     the pipeline as first shipped: Stage 1 rejects whenever
                p(mosquito) < 0.5, and a rejection is never an abstention.
  pipeline      the pipeline as deployed now, with the calibrated deferral band
                (see v2.pipeline.gate_decision and v2.calibrate_gate).

Scoring rule
------------
* true aedes / anopheles / culex: correct only if the pipeline names the right
  genus.
* true unknown_objects: correct if Stage 1 rejects it ("not a larva") OR
  Stage 2 calls it unknown_objects. Both are the right answer for the officer.
* "confident" / gated: a Stage 1 rejection is always an answer; past the gate,
  Stage 2 abstains below the calibrated threshold, as v2.pipeline.classify does.

Nothing is tuned here. Test is only scored.
"""

import json
import math
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2 import pipeline  # noqa: E402

DATA = "final_datasets_v2"
OUT = os.path.join("results", "v2", "final", "end_to_end.json")
SPLITS = ["test_field", "test_lab"]
SPECIES = {"aedes", "anopheles", "culex"}


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def outcome(true, p_mosquito, p2, reject_below, thr):
    """
    (decision, forced_answer, answered, correct) for one image under one gate.

    forced_answer is what the pipeline would say if it were never allowed to
    abstain; `answered` is whether it actually answers at the threshold.
    """
    decision = pipeline.gate_decision(p_mosquito, reject_below)
    if decision == "reject":
        answer, answered = "non_mosquito", True
    else:
        i = int(p2.argmax())
        answer, answered = pipeline.S2_CLASSES[i], float(p2[i]) >= thr
    if true in SPECIES:
        correct = answer == true
    else:
        correct = answer in ("non_mosquito", "unknown_objects")
    return decision, answer, answered, correct


def score(recs, key):
    """All-photo and confident-only accuracy for one view."""
    n = len(recs)
    k = sum(r[key]["correct"] for r in recs)
    ans = [r for r in recs if r[key]["answered"]]
    ka = sum(r[key]["correct"] for r in ans)
    return {"accuracy": k / n, "ci95": wilson(k, n),
            "gated_accuracy": ka / max(len(ans), 1),
            "gated_ci95": wilson(ka, len(ans)),
            "coverage": len(ans) / n, "n_answered": len(ans)}


def gate_breakdown(recs, key):
    sp = [r for r in recs if r["true"] in SPECIES]
    un = [r for r in recs if r["true"] == "unknown_objects"]

    def count(rs, pred):
        return sum(1 for r in rs if pred(r))

    return {
        "larvae": len(sp),
        "larvae_rejected": count(sp, lambda r: r[key]["decision"] == "reject"),
        "larvae_deferred": count(sp, lambda r: r[key]["decision"] == "defer"),
        "larvae_deferred_then_named_correctly_and_confidently": count(
            sp, lambda r: r[key]["decision"] == "defer"
            and r[key]["answered"] and r[key]["correct"]),
        "larvae_deferred_then_retake": count(
            sp, lambda r: r[key]["decision"] == "defer"
            and not r[key]["answered"]),
        "unknown": len(un),
        "unknown_rejected": count(un, lambda r: r[key]["decision"] == "reject"),
        "unknown_deferred": count(un, lambda r: r[key]["decision"] == "defer"),
        "unknown_passed_or_deferred_and_caught_by_stage2": count(
            un, lambda r: r[key]["decision"] != "reject"
            and r[key]["answer"] == "unknown_objects"),
    }


def main():
    st = pipeline.load()
    thr = st["threshold"]
    rb = st["gate_reject_below"]
    mode = st["mode"]
    print("mode %s | Stage 2 %s | threshold %.2f | TTA %s | gate rejects below %.2f"
          % (mode, [m["name"] for m in st["stage2"]], thr, st["use_tta"], rb))

    block = {"mode": mode, "ensemble": [m["name"] for m in st["stage2"]],
             "use_tta": st["use_tta"], "threshold": thr,
             "gate_reject_below": rb, "splits": {}}

    for split in SPLITS:
        recs = []
        t0 = time.perf_counter()
        for cls in pipeline.S2_CLASSES:
            d = os.path.join(DATA, split, cls)
            for fn in sorted(os.listdir(d)):
                img = Image.open(os.path.join(d, fn)).convert("RGB")
                p1 = pipeline.stage1_probs(st, img)
                p2 = pipeline.stage2_probs(st, img)
                pm = float(p1[pipeline.S1_CLASSES.index("mosquito")])
                i2 = int(p2.argmax())
                rec = {"file": fn, "true": cls, "p_mosquito": pm,
                       "s2": {"answer": pipeline.S2_CLASSES[i2],
                              "answered": float(p2[i2]) >= thr}}
                rec["s2"]["correct"] = rec["s2"]["answer"] == cls
                for key, r_below in (("hard", 0.5), ("deployed", rb)):
                    dec, ans, answered, ok = outcome(cls, pm, p2, r_below, thr)
                    rec[key] = {"decision": dec, "answer": ans,
                                "answered": answered, "correct": ok}
                recs.append(rec)

        n = len(recs)
        s = {"n": n,
             "stage2_only": score(recs, "s2"),
             "hard_gate": score(recs, "hard"),
             "pipeline": score(recs, "deployed"),
             "gate_hard": gate_breakdown(recs, "hard"),
             "gate_deployed": gate_breakdown(recs, "deployed"),
             "per_class_pipeline": {},
             "seconds": time.perf_counter() - t0}
        for c in pipeline.S2_CLASSES:
            rs = [r for r in recs if r["true"] == c]
            k = sum(r["deployed"]["correct"] for r in rs)
            s["per_class_pipeline"][c] = {"n": len(rs),
                                          "recall": k / max(len(rs), 1),
                                          "ci95": wilson(k, len(rs))}
        block["splits"][split] = s

        print("\n== %s  (n=%d, %.0f s)" % (split, n, s["seconds"]))
        for label, key in (("Stage 2 only", "stage2_only"),
                           ("hard gate", "hard_gate"),
                           ("PIPELINE", "pipeline")):
            v = s[key]
            print("  %-13s all %6.2f%% [%4.1f-%4.1f]   confident %6.2f%% "
                  "[%4.1f-%4.1f] @ %5.1f%% coverage"
                  % (label, 100 * v["accuracy"], 100 * v["ci95"][0],
                     100 * v["ci95"][1], 100 * v["gated_accuracy"],
                     100 * v["gated_ci95"][0], 100 * v["gated_ci95"][1],
                     100 * v["coverage"]))
        print("  gate (deployed):", json.dumps(s["gate_deployed"]))

    allres = {}
    if os.path.exists(OUT):
        prev = json.load(open(OUT, encoding="utf-8"))
        # An earlier single-mode layout had "splits" at the top level; start
        # afresh rather than mixing the two.
        if "splits" not in prev:
            allres = prev
    allres[mode] = block
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(allres, open(OUT, "w", encoding="utf-8"), indent=2)
    print("\nwrote %s [%s]" % (OUT, mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
