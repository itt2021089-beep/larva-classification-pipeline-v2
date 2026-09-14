"""
Build a self-contained zip of the v2 app + models for the team.

    python -m v2.package                # full ensemble  (~355 MB)
    python -m v2.package --mode lite    # single backbone (~200 MB)
    python -m v2.package --mode both

Read-only with respect to the project: it copies out of v2/, stage1/ and
results/v2/. Nothing is modified.

Before zipping, the packager RUNS THE PIPELINE from inside the built folder.
The v1 packaging exercise shipped a folder that crashed on first use because a
module the import chain needed was missing; a self-test catches that here
instead of on a teammate's laptop.

Output: dist_v2/safezone_v2[_lite]/ and the matching .zip.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist_v2")

# Everything the runtime imports. Verified by grepping the import chain of
# v2/pipeline.py, v2/app.py and v2/api.py: torch, torchvision, PIL, numpy,
# streamlit, fastapi -- plus stage1.model for the Stage 1 architecture.
CODE_DIRS = ["v2", "stage1"]
CODE_FILES = ["requirements-app.txt"]

LITE_MODEL = "convnext_tiny"
SAMPLES_PER_CLASS = 3


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(buf), b""):
            h.update(c)
    return h.hexdigest()


def copytree(src, dst):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.pyo", "raw_inat", "curation", "_check_v1.py"))


RUN_BAT = """@echo off
cd /d "%~dp0"
if not exist venv (
  echo Creating a virtual environment ^(one time^)...
  python -m venv venv
  venv\\Scripts\\python.exe -m pip install --upgrade pip
  venv\\Scripts\\python.exe -m pip install -r requirements-app.txt
)
echo.
echo Starting Safe Zone AI - a browser tab will open shortly.
venv\\Scripts\\python.exe -m streamlit run v2\\app.py
pause
"""

RUN_SH = """#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  echo "Creating a virtual environment (one time)..."
  python3 -m venv venv
  venv/bin/python -m pip install --upgrade pip
  venv/bin/python -m pip install -r requirements-app.txt
fi
echo
echo "Starting Safe Zone AI - a browser tab will open shortly."
venv/bin/python -m streamlit run v2/app.py
"""

API_BAT = """@echo off
cd /d "%~dp0"
venv\\Scripts\\python.exe -m uvicorn v2.api:app --host 0.0.0.0 --port 8000
pause
"""

VERIFY = '''"""Re-hash every file and compare against MANIFEST.json."""
import hashlib, json, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))

def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(buf), b""):
            h.update(c)
    return h.hexdigest()

man = json.load(open(os.path.join(ROOT, "MANIFEST.json"), encoding="utf-8"))
bad = []
for e in man["files"]:
    p = os.path.join(ROOT, e["path"].replace("/", os.sep))
    if not os.path.exists(p):
        bad.append((e["path"], "MISSING")); continue
    if os.path.getsize(p) != e["bytes"]:
        bad.append((e["path"], "wrong size")); continue
    if sha256(p) != e["sha256"]:
        bad.append((e["path"], "sha256 mismatch"))
print("checked %d files" % len(man["files"]))
if bad:
    for p, why in bad:
        print("  [FAIL] %s - %s" % (p, why))
    print("%d problem(s). Re-download or re-copy the package." % len(bad))
    sys.exit(1)
print("OK - every file matches the manifest.")
'''


def readme(mode, models, cal, n_samples):
    tf = (cal.get("test") or {}).get("test_field") or {}
    tl = (cal.get("test") or {}).get("test_lab") or {}
    thr = cal["abstention_threshold"]
    return """# Safe Zone AI v2 — mosquito larva classifier

Photograph a larva with a phone, copy the photos to a laptop, upload them here,
get the genus. Runs entirely on your machine — no internet needed after setup.

## Install and run

**Windows** — double-click `run_app.bat`
**macOS / Linux** — `bash run_app.sh`

The first run creates a virtual environment and downloads PyTorch
(~2 GB, several minutes, needs internet). Every run after that is offline and
starts in about 20 seconds. Requires **Python 3.10–3.12** installed and on PATH.

A browser tab opens at `http://localhost:8501`.

## Using it

Upload **one photo** for the full breakdown, or **many at once** for a table
plus a CSV you can keep as the surveillance record. `sample_photos/` has %d
images to try immediately.

Three possible outcomes per photo:

| result | what it means | what to do |
|---|---|---|
| **Aedes / Anopheles / Culex** | confident identification | record it |
| **Retake the photo** | below %.0f%% confidence | photograph the specimen again, closer and steadier |
| **Not a mosquito larva** | Stage 1 rejected it | check it really is a mosquito larva |

**"Retake" is a real answer, not a failure.** Forced to answer every field
photo the system is %.1f%% accurate; on the photos it is confident about it is
%.1f%% accurate. A wrong genus recorded confidently is worse than a ten-second
retake.

## How accurate is it, honestly

| | accuracy |
|---|---|
| Real smartphone field photos — all images | **%.1f%%** |
| Real smartphone field photos — when confident (%.0f%% of them) | **%.1f%%** |
| Laboratory / microscope images | **%.1f%%** |

Measured on %d held-out field photographs and %d laboratory images, neither
used for training or for setting the confidence threshold.

### What it cannot do

* **No Sri Lankan specimens** were used in training or evaluation. The field
  photographs are mostly North American and European. Genus-level features are
  shared, but this is untested on local vectors.
* **Aedes and Culex are the main confusion.** Both have a siphon and differ
  mainly in its length; in a small or blurry photo that is often not
  resolvable. Anopheles is the reliable one.
* Labels come from citizen-science identifications, not from an entomologist.
* **This is a research prototype, not a diagnostic device.**

## Getting good photos

Put the larva in a plain white tray, on tissue paper, or in a clear container.
Hold the phone 10–15 cm away, use macro mode if the phone has it, and avoid
glare off the water. Take **two or three shots of each specimen** — if one is
uncertain, another usually is not, and you cannot retake once you are back at
the desk.

## Keep the photos

Save every photo with what it actually was. The single biggest limitation of
this model is that no Sri Lankan larvae were used to build it — a few hundred
locally collected, expert-identified specimens would improve it more than any
change to the software.

## Also included

* `run_api.bat` — HTTP API at `http://localhost:8000` (docs at `/docs`) for
  scripting or integration.
* `verify_package.py` — re-hashes every file against `MANIFEST.json`. Worth
  running after any transfer; the model files are large and a truncated copy
  produces confusing errors rather than an obvious failure.

## What is in here

| path | what |
|---|---|
| `v2/app.py` | the web app |
| `v2/api.py` | the HTTP API |
| `v2/pipeline.py` | Stage 1 -> Stage 2 -> abstention |
| `stage1/model.py` | Stage 1 architecture |
| `results/v2/` | trained weights and the calibration |
| `sample_photos/` | %d photos to test with |
| `MANIFEST.json` | SHA-256 of every file |

Mode: **%s** (%s).

Built %s · ICT 4808 Group 08, Rajarata University of Sri Lanka.
Field photographs sourced from iNaturalist under CC licences — see
`ATTRIBUTIONS.md`.
""" % (n_samples, 100 * thr,
       100 * (tf.get("full") or {}).get("accuracy", 0),
       100 * (tf.get("gated") or {}).get("accuracy", 0),
       100 * (tf.get("full") or {}).get("accuracy", 0),
       100 * tf.get("coverage", 0),
       100 * (tf.get("gated") or {}).get("accuracy", 0),
       100 * (tl.get("full") or {}).get("accuracy", 0),
       (tf.get("full") or {}).get("n", 0), (tl.get("full") or {}).get("n", 0),
       n_samples, mode, ", ".join(models), date.today().isoformat())


def build(mode, args):
    name = "safezone_v2" + ("_lite" if mode == "lite" else "")
    pkg = os.path.join(DIST, name)
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    os.makedirs(pkg)
    print("\n=== building %s (%s) ===" % (name, mode))

    for d in CODE_DIRS:
        copytree(os.path.join(ROOT, d), os.path.join(pkg, d))
        print("  [code] %s/" % d)
    for f in CODE_FILES:
        shutil.copy2(os.path.join(ROOT, f), os.path.join(pkg, f))
        print("  [code] %s" % f)

    cal = json.load(open(os.path.join(ROOT, "results", "v2", "final",
                                      "v2_results.json"), encoding="utf-8"))
    if mode == "lite":
        # Rewrite the top-level calibration so the package needs no env var:
        # the lite block becomes THE configuration, with its own threshold.
        lite = cal["lite"]
        cal["ensemble"] = lite["ensemble"]
        cal["use_tta"] = lite["use_tta"]
        cal["abstention_threshold"] = lite["abstention_threshold"]
        cal["packaged_mode"] = "lite"
        cal["available_runs"] = {k: v for k, v in cal["available_runs"].items()
                                 if k in lite["ensemble"]}
    else:
        cal["packaged_mode"] = "full"
    keep = set(cal["ensemble"])

    # checkpoints + metadata for the members actually used, plus Stage 1
    os.makedirs(os.path.join(pkg, "results", "v2", "final"), exist_ok=True)
    for sub in ["stage1_resnet50"] + sorted(keep):
        src = os.path.join(ROOT, "results", "v2", sub)
        dst = os.path.join(pkg, "results", "v2", sub)
        os.makedirs(dst, exist_ok=True)
        for f in ("best_model.pth", "metadata.json"):
            s = os.path.join(src, f)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, f))
        print("  [model] %s" % sub)
    # normalise separators so the package works on macOS and Linux
    for k, v in cal["available_runs"].items():
        v["ckpt"] = v["ckpt"].replace("\\", "/")
    json.dump(cal, open(os.path.join(pkg, "results", "v2", "final",
                                     "v2_results.json"), "w",
                        encoding="utf-8"), indent=2)

    for f in ("ATTRIBUTIONS.md",):
        s = os.path.join(ROOT, "final_datasets_v2", f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(pkg, f))

    # sample photos
    n_samples = 0
    if not args.no_samples:
        sd = os.path.join(pkg, "sample_photos")
        os.makedirs(sd, exist_ok=True)
        import glob
        for c in ("aedes", "anopheles", "culex", "unknown_objects"):
            fs = sorted(glob.glob(os.path.join(
                ROOT, "final_datasets_v2", "test_field", c, "*")))
            random.Random(7).shuffle(fs)
            for i, f in enumerate(fs[:SAMPLES_PER_CLASS]):
                shutil.copy2(f, os.path.join(
                    sd, "%s_%d%s" % (c, i + 1, os.path.splitext(f)[1])))
                n_samples += 1
        print("  [samples] %d photos" % n_samples)

    for fn, body, nl in (("run_app.bat", RUN_BAT, "\r\n"),
                         ("run_api.bat", API_BAT, "\r\n"),
                         ("run_app.sh", RUN_SH, "\n"),
                         ("verify_package.py", VERIFY, "\n")):
        with open(os.path.join(pkg, fn), "w", encoding="utf-8", newline=nl) as fh:
            fh.write(body)
    with open(os.path.join(pkg, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme(mode, sorted(keep), cal, n_samples))

    # manifest
    files = []
    for dp, dns, fns in os.walk(pkg):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in sorted(fns):
            if fn == "MANIFEST.json":
                continue
            full = os.path.join(dp, fn)
            files.append({"path": os.path.relpath(full, pkg).replace("\\", "/"),
                          "bytes": os.path.getsize(full), "sha256": sha256(full)})
    total = sum(f["bytes"] for f in files)
    json.dump({"package": name, "mode": mode,
               "built": date.today().isoformat(),
               "file_count": len(files), "total_bytes": total,
               "files": files},
              open(os.path.join(pkg, "MANIFEST.json"), "w", encoding="utf-8"),
              indent=2)
    print("  %d files, %.0f MB" % (len(files), total / 1e6))

    # --- self-test: run the pipeline from INSIDE the package -----------
    if not args.no_self_test and n_samples:
        import glob
        probe = sorted(glob.glob(os.path.join(pkg, "sample_photos", "*")))[0]
        print("  self-test: classifying %s from inside the package …"
              % os.path.basename(probe))
        code = ("import sys; sys.path.insert(0,'.');"
                "from v2.pipeline import classify, load;"
                "st=load();"
                "r=classify(%r);"
                "print('MODE', st['mode'], '| MODELS',"
                "[m['name'] for m in st['stage2']], '| THR', st['threshold']);"
                "print('RESULT', r['final_label'], round(r['confidence'],3),"
                "'abstained', r['abstained'])" % probe)
        p = subprocess.run([sys.executable, "-c", code], cwd=pkg,
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-2000:]); print(p.stderr[-2000:])
            raise SystemExit("  SELF-TEST FAILED — the package is incomplete.")
        for line in p.stdout.strip().splitlines()[-2:]:
            print("    " + line)
        print("  self-test PASSED")

    if not args.no_zip:
        z = os.path.join(DIST, name + ".zip")
        if os.path.exists(z):
            os.remove(z)
        print("  zipping (a few minutes) …")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for dp, dns, fns in os.walk(pkg):
                dns[:] = [d for d in dns if d != "__pycache__"]
                for fn in fns:
                    full = os.path.join(dp, fn)
                    zf.write(full, os.path.join(name,
                                                os.path.relpath(full, pkg)))
        print("  [zip] %s  %.0f MB" % (z, os.path.getsize(z) / 1e6))
    return pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lite", "both"], default="full")
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-self-test", action="store_true")
    a = ap.parse_args()
    os.makedirs(DIST, exist_ok=True)
    modes = ["full", "lite"] if a.mode == "both" else [a.mode]
    for m in modes:
        build(m, a)
    print("\ndone — send the .zip from %s" % DIST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
