"""
Assemble a self-contained model package for the demo application.

    python -m stage1.package_demo

Creates:

    demo_models/
    ├── stage1_resnet50/
    │   ├── best_model.pth          (copy of the selected Stage 1 checkpoint)
    │   ├── class_mapping.json      (verified index -> class name)
    │   └── model_config.json       (architecture + preprocessing contract)
    ├── stage2_efficientnet/
    │   └── best_tuning_run.h5      (copy; the model itself is NEVER modified)
    └── PACKAGE_INFO.json           (SHA-256 of every file, provenance, pipeline rules)

Copies are byte-for-byte and verified by SHA-256 against the source, so the
Stage 2 Keras model is provably unaltered.
"""

import hashlib
import json
import os
import shutil

STAGE1_SRC_DIR = os.path.join("results", "stage1", "resnet50")
STAGE2_SRC = os.path.join(os.path.expanduser("~"), "Downloads", "best_tuning_run.h5")

PKG = "demo_models"
S1 = os.path.join(PKG, "stage1_resnet50")
S2 = os.path.join(PKG, "stage2_efficientnet")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _copy_verified(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    a, b = sha(src), sha(dst)
    if a != b:
        raise RuntimeError(f"copy verification failed for {src}")
    return {"source": src.replace("\\", "/"), "path": dst.replace("\\", "/"),
            "sha256": a, "bytes": os.path.getsize(dst)}


def main(stage1_run="baseline", stage2_src=STAGE2_SRC, verbose=True):
    os.makedirs(S1, exist_ok=True)
    os.makedirs(S2, exist_ok=True)
    info = {"files": {}, "pipeline": {}, "provenance": {}}

    # ---- Stage 1 ----
    run_dir = os.path.join(STAGE1_SRC_DIR, stage1_run)
    ckpt = os.path.join(run_dir, "best_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"{ckpt} not found — train Stage 1 first")
    info["files"]["stage1_checkpoint"] = _copy_verified(
        ckpt, os.path.join(S1, "best_model.pth"))

    mapping_src = os.path.join(STAGE1_SRC_DIR, "class_mapping.json")
    info["files"]["stage1_class_mapping"] = _copy_verified(
        mapping_src, os.path.join(S1, "class_mapping.json"))
    with open(mapping_src, encoding="utf-8") as fh:
        mapping = json.load(fh)

    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        train_cfg = json.load(fh)
    with open(os.path.join(run_dir, "metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)
    test_m = metrics["metrics"]["test"]

    model_config = {
        "stage": 1,
        "task": "binary classification: Larva vs Non-larva",
        "architecture": "ResNet50 (torchvision), ImageNet pretrained, "
                        "fc replaced by Dropout + Linear(2048 -> 2)",
        "framework": "pytorch",
        "input_size": train_cfg["img_size"],
        "input_layout": "NCHW, RGB",
        "num_classes": 2,
        "class_mapping": mapping,
        "source_folders": {"0": "larvae", "1": "non_larvae"},
        "preprocessing": {
            "order": ["gamma_correction", "resize", "to_tensor", "imagenet_normalize"]
            if train_cfg.get("preprocessing") == "gamma"
            else ["resize", "to_tensor", "imagenet_normalize"],
            "gamma": train_cfg.get("gamma"),
            "resize": [train_cfg["img_size"], train_cfg["img_size"]],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "note": "No random augmentation at inference. Identical to the "
                    "validation/test transform used in training.",
        },
        "training_run": stage1_run,
        "measured_test_metrics_this_model": {
            "accuracy": test_m["accuracy"],
            "macro_f1": test_m["f1_macro"],
            "larva_recall": test_m["per_class"]["Larva"]["recall"],
            "non_larva_recall": test_m["per_class"]["Non_larva"]["recall"],
            "n_test_images": test_m["n"],
        },
        "IMPORTANT": "These metrics belong to THIS locally trained model only. "
                     "They are not the 99.88% figure quoted in the research "
                     "presentation, which came from a different earlier "
                     "experiment on a different (leaky) split.",
    }
    p = os.path.join(S1, "model_config.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(model_config, fh, indent=2)
    info["files"]["stage1_model_config"] = {
        "path": p.replace("\\", "/"), "sha256": sha(p), "bytes": os.path.getsize(p)}

    # ---- Stage 2 (copied, never modified) ----
    if os.path.exists(stage2_src):
        info["files"]["stage2_model"] = _copy_verified(
            stage2_src, os.path.join(S2, "best_tuning_run.h5"))
        info["files"]["stage2_model"]["modified"] = False
        info["files"]["stage2_model"]["note"] = (
            "Byte-for-byte copy, SHA-256 verified against the source. Not "
            "retrained and not modified.")
        info["provenance"]["stage2"] = {
            "framework": "keras 3.13.2 / tensorflow",
            "architecture": "EfficientNet-B0 (Functional, 243 layers)",
            "input_shape": [224, 224, 3],
            "output": "Dense(4, softmax)",
            "classes_expected": ["Aedes", "Anopheles", "Culex", "Non_larvae"],
            "note": "Class ORDER must be confirmed against the Stage 2 training "
                    "code before wiring the demo — it is not recoverable from "
                    "the .h5 weights alone.",
        }
    else:
        info["files"]["stage2_model"] = {"path": None,
                                         "error": f"not found at {stage2_src}"}

    info["pipeline"] = {
        "order": ["input image", "Stage 1 ResNet50 (binary)",
                  "if Larva -> Stage 2 EfficientNet-B0 (4-class)", "final result"],
        "gate_rule": "If Stage 1 predicts Non_larva, STOP. Do not run Stage 2. "
                     "Display: 'Non-larva detected. Stage 2 was not executed.'",
        "stage1_entrypoint": "from stage1.inference import predict_stage1",
        "shared_input_size": 224,
        "note": "Stage 1 and Stage 2 both take 224x224 RGB, so the demo can "
                "decode the image once. Stage 2 expects NHWC (Keras); Stage 1 "
                "expects NCHW (PyTorch) — the Stage 1 module handles its own "
                "preprocessing internally.",
    }
    info["provenance"]["stage1"] = {
        "trained_locally": True,
        "dataset": "dataset_binary/ (re-split leakage-free; see "
                   "results/stage1/dataset_audit.md)",
        "checkpoint_selected_on": "validation macro F1",
        "run": stage1_run,
    }

    pi = os.path.join(PKG, "PACKAGE_INFO.json")
    with open(pi, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    if verbose:
        print(f"[package] wrote {PKG}/")
        for k, v in info["files"].items():
            if v.get("path"):
                print(f"  {k}: {v['path']} ({v.get('bytes',0)/1e6:.1f} MB)")
            else:
                print(f"  {k}: MISSING — {v.get('error')}")
    return info


if __name__ == "__main__":
    main()
