---
title: Safe Zone AI v2
emoji: 🦟
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.63.0"
app_file: v2/app.py
pinned: false
---

# Safe Zone AI v2 — mosquito larva classifier

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
plus a CSV you can keep as the surveillance record. `sample_photos/` has 12
images to try immediately.

Three possible outcomes per photo:

| result | what it means | what to do |
|---|---|---|
| **Aedes / Anopheles / Culex** | confident identification | record it |
| **Retake the photo** | below 91% confidence | photograph the specimen again, closer and steadier |
| **Not a mosquito larva** | Stage 1 rejected it | check it really is a mosquito larva |

**"Retake" is a real answer, not a failure.** Forced to answer every field
photo the system is 72.2% accurate; on the photos it is confident about it is
89.9% accurate. A wrong genus recorded confidently is worse than a ten-second
retake.

## How accurate is it, honestly

| | accuracy |
|---|---|
| Real smartphone field photos — all images | **72.2%** |
| Real smartphone field photos — when confident (49% of them) | **89.9%** |
| Laboratory / microscope images | **93.6%** |

Measured on 381 held-out field photographs and 280 laboratory images, neither
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
| `sample_photos/` | 12 photos to test with |
| `MANIFEST.json` | SHA-256 of every file |

Mode: **full** (convnext_tiny, efficientnet_v2_s, efficientnet_v2_s_320).

Built 2026-09-14 · ICT 4808 Group 08, Rajarata University of Sri Lanka.
Field photographs sourced from iNaturalist under CC licences — see
`ATTRIBUTIONS.md`.
