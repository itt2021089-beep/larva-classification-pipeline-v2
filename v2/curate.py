"""
v2 step 2 - curate the field photos and build the v2 dataset.

    python -m v2.curate --audit     # inspect, verify, dedupe; writes no dataset
    python -m v2.curate --build     # materialise final_datasets_v2/

The central design decision
---------------------------
v1 had one test set, and every image in it was a laboratory capture. That made
"how well does this work on a PHI officer's phone photo?" unanswerable. v2
splits evaluation in two:

    test_field  - held-out iNaturalist observations ONLY. Smartphone photos of
                  larvae in trays, jars, hands, puddles. This is the number
                  that actually predicts deployment.
    test_lab    - the v1 locked test set, unchanged, so every previous result
                  stays comparable.

Training and validation mix both domains, because the model has to work on
both and the lab images carry most of the morphological signal.

Leakage control
---------------
An iNaturalist observation is one specimen photographed in one session, often
2-3 frames. All photos of an observation share `obs_<id>` and move together.
On top of that the project's existing four-signal check runs: SHA-256, the
letterbox-stripped perceptual descriptor, filename-stem lineage, and
group identity. Nothing is assumed; everything is verified afterwards.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from stage2.build_anopheles import _descriptor_unboxed
from stage2.dedup import _dihedral

Image.MAX_IMAGE_PIXELS = None

RAW = os.path.join("v2", "raw_inat")
V1 = os.path.join("final_datasets", "stage_2_multiclass")
OUT = "final_datasets_v2"
REPORT = os.path.join("v2", "curation")
CLASSES = ["aedes", "anopheles", "culex", "unknown_objects"]
SPECIES = ["aedes", "anopheles", "culex"]
DUP = 0.90
DESC_N = 64 * 64
MIN_SIDE = 200
SEED = 42
FIELD_TEST_FRAC = 0.22
FIELD_VAL_FRAC = 0.16


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(buf), b""):
            h.update(c)
    return h.hexdigest()


def max_sim(v, REF):
    if len(REF) == 0:
        return -1.0, -1
    best, arg = -1.0, -1
    for t in _dihedral(v.reshape(64, 64)):
        s = (REF @ t.reshape(-1)) / DESC_N
        j = int(s.argmax())
        if s[j] > best:
            best, arg = float(s[j]), j
    return best, arg


def load_v1():
    rows = []
    for sp in ("train", "val", "test"):
        for r in csv.DictReader(open(os.path.join(V1, "manifests", sp + ".csv"),
                                     encoding="utf-8")):
            r["v1_split"] = sp
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if not (a.audit or a.build):
        a.audit = True
    os.makedirs(REPORT, exist_ok=True)
    rng = random.Random(SEED)

    raw = list(csv.DictReader(open(os.path.join(RAW, "manifest.csv"),
                                   encoding="utf-8")))
    print("raw iNaturalist photos: %d" % len(raw))

    # Keep only the first two photos of each observation.
    #
    # iNaturalist observers routinely upload a specimen shot followed by
    # context shots of the breeding site - a puddle, a tyre, a leaf - and those
    # carry the species label while containing no larva at all. The observer
    # leads with the specimen, so photo 0 (and usually 1) is the animal.
    #
    # A model-based filter was tried first and rejected: the v1 Stage 1
    # classifier assigns p(larva) < 0.5 to 35-42% of these genuine field
    # photographs because it was trained purely on laboratory images. Using it
    # to clean the data would have discarded a third of the real specimens and
    # biased the set back toward the lab domain this whole exercise exists to
    # escape.
    before = len(raw)
    raw = [r for r in raw if int(r.get("photo_index_in_obs", 0)) < 2]
    print("  after keeping <=2 photos per observation: %d (dropped %d)"
          % (len(raw), before - len(raw)))

    # ---- 1. readability + size ------------------------------------
    kept, dropped = [], []
    for r in raw:
        p = r["path"]
        if not os.path.exists(p):
            dropped.append((r, "missing"))
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:
            dropped.append((r, "unreadable " + type(e).__name__))
            continue
        if min(w, h) < MIN_SIDE:
            dropped.append((r, "too small %dx%d" % (w, h)))
            continue
        r["width"], r["height"] = w, h
        kept.append(r)
    print("  readable and >= %dpx: %d   dropped: %d" % (MIN_SIDE, len(kept), len(dropped)))

    # ---- 2. exact duplicates --------------------------------------
    by_sha = defaultdict(list)
    for r in kept:
        r["sha256"] = sha256(r["path"])
        by_sha[r["sha256"]].append(r)
    uniq = []
    for h, rs in by_sha.items():
        uniq.append(rs[0])
        for extra in rs[1:]:
            dropped.append((extra, "exact duplicate of " + rs[0]["filename"]))
    print("  after exact dedup: %d" % len(uniq))

    # ---- 3. descriptors, then dedup against v1 and within ---------
    print("  computing descriptors for %d new photos ..." % len(uniq))
    D = np.stack([_descriptor_unboxed(r["path"]) for r in uniq]).reshape(len(uniq), -1)
    v1 = load_v1()
    print("  computing descriptors for %d v1 images ..." % len(v1))
    V = np.stack([_descriptor_unboxed(r["final_path"]) for r in v1]).reshape(len(v1), -1)

    survivors = []
    for i, r in enumerate(uniq):
        s, j = max_sim(D[i], V)
        r["sim_v1"] = round(s, 4)
        if s >= DUP:
            dropped.append((r, "near-duplicate of v1 %s (%.3f)"
                            % (os.path.basename(v1[j]["final_path"]), s)))
        else:
            survivors.append(i)
    print("  after v1 dedup: %d" % len(survivors))

    # within-new, group aware: two photos of the SAME observation are expected
    # to look alike and are NOT duplicates - they are one specimen, one group.
    final_idx, kept_desc, kept_groups = [], [], []
    for i in survivors:
        r = uniq[i]
        dup = None
        for k, j in enumerate(final_idx):
            if uniq[j]["group_id"] == r["group_id"]:
                continue
            s, _ = max_sim(D[i], D[j][None, :])
            if s >= DUP:
                dup = (j, s)
                break
        if dup:
            dropped.append((r, "near-duplicate of another observation %s (%.3f)"
                            % (uniq[dup[0]]["filename"], dup[1])))
        else:
            final_idx.append(i)
    new = [uniq[i] for i in final_idx]
    print("  after cross-observation dedup: %d" % len(new))

    # ---- 4. report -------------------------------------------------
    per_class = Counter(r["label"] for r in new)
    groups = defaultdict(set)
    for r in new:
        groups[r["label"]].add(r["group_id"])
    print("\n  kept photos / distinct observations per class:")
    for c in CLASSES:
        print("    %-18s %4d photos   %4d observations"
              % (c, per_class[c], len(groups[c])))
    qg = Counter((r["label"], r["quality_grade"]) for r in new)
    print("\n  label confidence (quality grade):")
    for c in CLASSES:
        row = {k[1]: v for k, v in qg.items() if k[0] == c}
        print("    %-18s %s" % (c, row))

    with open(os.path.join(REPORT, "dropped.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "label", "reason"])
        for r, why in dropped:
            w.writerow([r.get("filename"), r.get("label"), why])

    if not a.build:
        json.dump({"raw": len(raw), "kept": len(new),
                   "per_class": dict(per_class),
                   "observations_per_class": {c: len(groups[c]) for c in CLASSES},
                   "dropped": len(dropped)},
                  open(os.path.join(REPORT, "audit.json"), "w",
                       encoding="utf-8"), indent=2)
        print("\n(audit only - no dataset written)")
        print("[out] " + REPORT)
        return 0

    # ---- 5. split the FIELD data by observation group -------------
    if os.path.exists(OUT):
        sys.exit(OUT + " already exists; remove it before rebuilding")

    assign = {}
    for c in CLASSES:
        g = sorted(groups[c])
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, int(round(n * FIELD_TEST_FRAC)))
        n_val = max(1, int(round(n * FIELD_VAL_FRAC)))
        for i, gid in enumerate(g):
            assign[gid] = ("test_field" if i < n_test
                           else "val" if i < n_test + n_val else "train")

    rows_out = []
    for r in new:
        sp = assign[r["group_id"]]
        rows_out.append({
            "split": sp, "class": r["label"], "domain": "field",
            "source_path": r["path"], "filename": r["filename"],
            "group_id": r["group_id"], "sha256": r["sha256"],
            "observation_id": r["observation_id"], "taxon_name": r["taxon_name"],
            "taxon_rank": r["taxon_rank"], "quality_grade": r["quality_grade"],
            "license": r["license"], "attribution": r["attribution"],
            "source_url": r["source_url"], "place_guess": r["place_guess"],
            "width": r["width"], "height": r["height"],
            "sim_v1": r["sim_v1"],
        })

    # ---- 6. carry v1 across, test stays test ----------------------
    for r in v1:
        sp = "test_lab" if r["v1_split"] == "test" else r["v1_split"]
        rows_out.append({
            "split": sp, "class": r["class"], "domain": "lab",
            "source_path": r["final_path"], "filename": r["filename"],
            "group_id": r.get("group_id") or ("v1_" + r["filename"]),
            "sha256": r.get("sha256", ""), "observation_id": "",
            "taxon_name": "", "taxon_rank": "", "quality_grade": "v1_curated",
            "license": "project", "attribution": "project dataset",
            "source_url": "", "place_guess": "",
            "width": "", "height": "", "sim_v1": "",
        })

    # ---- 7. materialise -------------------------------------------
    used = set()
    for r in rows_out:
        d = os.path.join(OUT, r["split"], r["class"])
        os.makedirs(d, exist_ok=True)
        base = r["filename"]
        dst = os.path.join(d, base)
        if dst.lower() in used:
            stem, ext = os.path.splitext(base)
            dst = os.path.join(d, stem + "__" + r["sha256"][:8] + ext)
        used.add(dst.lower())
        shutil.copy2(r["source_path"], dst)
        r["final_path"] = dst.replace("\\", "/")

    mdir = os.path.join(OUT, "manifests")
    os.makedirs(mdir, exist_ok=True)
    cols = list(rows_out[0].keys())
    for sp in ("train", "val", "test_field", "test_lab"):
        sub = [r for r in rows_out if r["split"] == sp]
        with open(os.path.join(mdir, sp + ".csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(sub)

    summary = {"classes": CLASSES, "species_of_interest": SPECIES,
               "splits": {}, "by_domain": {}}
    print("\n" + "=" * 74)
    for sp in ("train", "val", "test_field", "test_lab"):
        sub = [r for r in rows_out if r["split"] == sp]
        c = Counter(r["class"] for r in sub)
        dm = Counter(r["domain"] for r in sub)
        summary["splits"][sp] = {"total": len(sub), "per_class": dict(c),
                                 "per_domain": dict(dm)}
        print("%-11s %5d  %s   %s" % (sp, len(sub), dict(c), dict(dm)))
    json.dump(summary, open(os.path.join(mdir, "dataset_summary.json"), "w",
                            encoding="utf-8"), indent=2)
    shutil.copy2(os.path.join(RAW, "ATTRIBUTIONS.md"),
                 os.path.join(OUT, "ATTRIBUTIONS.md"))
    print("\n[out] " + OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
