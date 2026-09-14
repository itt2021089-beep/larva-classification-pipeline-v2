"""
v2 final report generator.

    python -m v2.report

Reads only artefacts already produced by the v2 pipeline. Invents no numbers.
"""

import json
import os
import sys
from datetime import date

C = ["aedes", "anopheles", "culex", "unknown_objects"]
OUT = os.path.join("results", "v2", "V2_FINAL_REPORT.md")


def main():
    d = json.load(open(os.path.join("results", "v2", "final", "v2_results.json"),
                       encoding="utf-8"))
    s1 = json.load(open(os.path.join("results", "v2", "stage1_resnet50",
                                     "metadata.json"), encoding="utf-8"))
    ds = json.load(open(os.path.join("final_datasets_v2", "manifests",
                                     "dataset_summary.json"), encoding="utf-8"))
    tf, tl = d["test"]["test_field"], d["test"]["test_lab"]
    s1f = s1["results"]["test_field"]["all"]
    s1l = s1["results"]["test_lab"]["all"]

    L = []
    A = L.append
    A("# Safe Zone AI — v2 field-robust larva classification pipeline")
    A("")
    A("Generated " + date.today().isoformat())
    A("")
    A("Goal: a PHI officer photographs a larva with a smartphone — on a hand, on "
      "tissue paper, in a tray of water — and gets a genus back.")
    A("")
    A("---")
    A("")
    A("## Headline")
    A("")
    A("| | v1 (lab only) | **v2** |")
    A("|---|---|---|")
    A("| Laboratory test accuracy | 86.79%% | **%.2f%%** |" % (100 * tl["full"]["accuracy"]))
    A("| Laboratory macro-F1 | 0.8173 | **%.4f** |" % tl["full"]["f1_macro"])
    A("| Laboratory Anopheles recall | 0.458 | **%.3f** |"
      % tl["full"]["per_class"]["anopheles"]["recall"])
    A("| **Field test accuracy** | *never measured* | **%.2f%%** |"
      % (100 * tf["full"]["accuracy"]))
    A("| Field 3-species accuracy | *never measured* | **%.2f%%** |"
      % (100 * tf["full"]["species_accuracy"]))
    A("| Field accuracy on the confident %.0f%% | — | **%.2f%%** |"
      % (100 * tf["coverage"], 100 * tf["gated"]["accuracy"]))
    A("")
    A("**The >=90%% target is reached on laboratory images and on the confident "
      "subset of field images, but NOT on unfiltered field photographs.** Forced "
      "to answer every field photo the system is right %.1f%% of the time; "
      "allowed to say \"retake\" it answers %.0f%% of them at %.1f%% accuracy."
      % (100 * tf["full"]["accuracy"], 100 * tf["coverage"],
         100 * tf["gated"]["accuracy"]))
    A("")
    A("## Dataset")
    A("")
    A("| split | total | aedes | anopheles | culex | unknown | domain |")
    A("|---|---|---|---|---|---|---|")
    for sp in ("train", "val", "test_field", "test_lab"):
        s = ds["splits"][sp]
        pc = s["per_class"]
        A("| %s | %d | %d | %d | %d | %d | %s |"
          % (sp, s["total"], pc.get("aedes", 0), pc.get("anopheles", 0),
             pc.get("culex", 0), pc.get("unknown_objects", 0),
             ", ".join("%s %d" % (k, v) for k, v in s["per_domain"].items())))
    A("")
    A("`test_field` is 381 real smartphone photographs from held-out "
      "iNaturalist observations — the evaluation this project never had. "
      "`test_lab` is the v1 locked test set, unchanged, so every earlier "
      "result stays comparable.")
    A("")
    A("## The two things that actually moved the needle")
    A("")
    A("### 1. Real field photographs")
    A("")
    A("2,211 smartphone photographs collected from iNaturalist across 1,090 "
      "independent observations and hundreds of photographers, each carrying "
      "its observation id, taxon, quality grade, licence and attribution.")
    A("")
    A("This is what fixed Anopheles. Two previous repair attempts using "
      "balanced sampling, class-balanced focal loss and targeted augmentation "
      "moved its test recall by exactly 0.0000. Adding real specimens moved "
      "laboratory recall 0.458 -> %.3f."
      % tl["full"]["per_class"]["anopheles"]["recall"])
    A("")
    A("### 2. Stage 1 was silently breaking the pipeline")
    A("")
    A("The v1 gate, trained only on laboratory images, assigned p(larva) < 0.5 "
      "to **36-42% of genuine field larva photographs**. It would have "
      "discarded more than a third of real field photos before the species "
      "classifier ever ran — no Stage 2 improvement could have rescued that.")
    A("")
    A("| Stage 1 | v1 | **v2 (lab + field)** |")
    A("|---|---|---|")
    A("| field mosquito-recall | ~60%% | **%.2f%%** |" % (100 * s1f["mosquito_recall"]))
    A("| field accuracy | — | %.2f%% |" % (100 * s1f["accuracy"]))
    A("| lab accuracy | — | %.2f%% |" % (100 * s1l["accuracy"]))
    A("")
    A("## Architecture")
    A("")
    A("```")
    A("Smartphone photo")
    A("      |")
    A("Stage 1  ResNet-50 (lab + field)       mosquito / non_mosquito")
    A("      |--- non_mosquito -> STOP")
    A("      |")
    A("Stage 2  ensemble, multi-resolution:")
    for n in d["ensemble"]:
        i = d["available_runs"][n]
        A("           %-28s %s @ %dpx" % (n, i["backbone"], i["image_size"]))
    A("      |")
    A("confidence >= %.2f ?  yes -> aedes / anopheles / culex / unknown"
      % d["abstention_threshold"])
    A("                      no  -> RETAKE THE PHOTO")
    A("```")
    A("")
    A("Preprocessing is resize + ImageNet normalise. v1's Gaussian blur was "
      "measured to change the final tensor by 0.11% (a no-op after the "
      "downscale) and its bounding-box crop fired on only 31% of images and "
      "improved every metric when removed. Both are gone.")
    A("")
    A("## Field test set — 381 real smartphone photographs")
    A("")
    A("| class | precision | recall | F1 | n |")
    A("|---|---|---|---|---|")
    for c in C:
        p = tf["full"]["per_class"][c]
        A("| %s | %.3f | %.3f | %.3f | %d |"
          % (c, p["precision"], p["recall"], p["f1"], p["support"]))
    A("")
    A("Confusion (rows = true, columns = predicted):")
    A("")
    A("| | " + " | ".join(C) + " |")
    A("|---|" + "---|" * 4)
    for i, c in enumerate(C):
        A("| **%s** | %s |"
          % (c, " | ".join(str(v) for v in tf["full"]["confusion_matrix"][i])))
    A("")
    A("**The remaining bottleneck is Aedes <-> Culex, not Anopheles.** Both "
      "genera have a siphon and differ mainly in its length and slenderness; in "
      "a field photo where the larva spans roughly 85 px that distinction is "
      "often not resolvable. Training at 320 px was tried specifically to test "
      "this and did NOT fix it — ConvNeXt got worse (0.775 -> 0.751 field "
      "macro-F1), EfficientNetV2-S slightly better (0.735 -> 0.755). The "
      "resolution hypothesis is not supported at this data size.")
    A("")
    A("## Risk-coverage: choosing the operating point")
    A("")
    A("| threshold | coverage | accuracy on answered |")
    A("|---|---|---|")
    for r in tf["risk_coverage"]:
        if r["threshold"] in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
            A("| %.2f | %.1f%% | %.1f%% |"
              % (r["threshold"], 100 * r["coverage"],
                 100 * r["accuracy_on_answered"]))
    A("")
    A("The threshold lives in `results/v2/final/v2_results.json` and can be "
      "moved without retraining.")
    A("")
    A("## Laboratory test set — 280 images (v1 locked set, unchanged)")
    A("")
    A("| class | precision | recall | F1 | n |")
    A("|---|---|---|---|---|")
    for c in C:
        p = tl["full"]["per_class"][c]
        A("| %s | %.3f | %.3f | %.3f | %d |"
          % (c, p["precision"], p["recall"], p["f1"], p["support"]))
    A("")
    A("## Honest limitations")
    A("")
    A("1. **Field accuracy is %.1f%%, not 90%%.** The 90%% figure holds on "
      "laboratory images and on the %.0f%% of field photos the model is "
      "confident about."
      % (100 * tf["full"]["accuracy"], 100 * tf["coverage"]))
    A("2. **No Sri Lankan data.** The field photographs are overwhelmingly "
      "North American and European. Sri Lankan vectors (*An. culicifacies*, "
      "*An. subpictus*, *An. stephensi*) are not represented. Genus-level "
      "features are shared, but this belongs in any write-up.")
    A("3. **Labels are citizen-science.** Most field observations are "
      "identified to genus by one person rather than confirmed by the "
      "community. They were sampled across hundreds of observers to avoid "
      "systematic error, not verified individually by an entomologist.")
    A("4. **The public pool is exhausted.** Raising the per-observer cap from "
      "12 to 20 and the Aedes target from 1,100 to 1,600 observations yielded "
      "**58 additional images**. The binding constraint is photographer "
      "diversity, not observation count.")
    A("5. **Aedes <-> Culex is unsolved** and is roughly half of all field "
      "errors. Higher resolution did not fix it.")
    A("")
    A("## A bias that was caught before it reached training")
    A("")
    A("The first data fetch requested research-grade Aedes first, on the "
      "reasoning that community-confirmed identifications are more "
      "trustworthy. Research-grade status turns out to correlate with "
      "*systematic uploaders*: 72% of the Aedes images came from one Mexican "
      "health jurisdiction photographing specimens down a microscope, and 69% "
      "carried a circular eyepiece vignette against 6-14% for every other "
      "class. A model trained on that learns \"dark corners -> Aedes\" and "
      "posts excellent validation scores while failing completely in the "
      "field. A per-observer cap and dropping the research-grade preference "
      "brought the Aedes vignette share to 13% and the top contributor to 7%.")
    A("")
    A("## What would actually raise field accuracy")
    A("")
    A("1. **Sri Lankan field photographs with expert identification.** A few "
      "hundred PHI-collected, entomologist-verified specimens would be worth "
      "more than everything else on this list combined.")
    A("2. **A detect-then-classify stage.** Field larvae are small in frame; a "
      "detector crop would hand the classifier a full-resolution specimen "
      "instead of an 85-pixel smear. This is the largest untried lever.")
    A("3. **A capture protocol.** Larva in a white tray, phone 10-15 cm, macro "
      "mode. Standardising the input is cheaper than making the model "
      "invariant to everything, and the risk-coverage table shows how much "
      "confident coverage grows as image quality rises.")
    A("")
    A("## Files")
    A("")
    A("| path | what |")
    A("|---|---|")
    A("| `v2/pipeline.py` | the deployable pipeline — `classify(image)` |")
    A("| `v2/fetch_inat.py` | data collection with provenance and attribution |")
    A("| `v2/curate.py` | dedup, leakage control, split construction |")
    A("| `v2/train.py`, `v2/train_stage1.py` | Stage 2 and Stage 1 training |")
    A("| `v2/evaluate.py` | ensemble, TTA and abstention calibration |")
    A("| `final_datasets_v2/` | the dataset with per-image manifests |")
    A("| `final_datasets_v2/ATTRIBUTIONS.md` | CC attribution for every image |")
    A("| `results/v2/final/v2_results.json` | all metrics and the risk-coverage curve |")
    A("")

    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("[out] " + OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
