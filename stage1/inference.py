"""
Stage 1 inference for the demo application.

    from stage1.inference import predict_stage1
    result = predict_stage1("some_image.jpg")
    # {"class": "Larva", "confidence": 0.98, "probabilities": {...}, ...}

Design points that matter for a demo:
  * The model is loaded ONCE, lazily, and cached (`_MODEL`). Calling
    `predict_stage1` in a loop does not reload the checkpoint.
  * Preprocessing is imported from the training module, not re-implemented, so
    inference cannot silently drift from how the model was trained
    (gamma correction -> resize 224 -> ToTensor -> ImageNet normalise).
    No random augmentation is ever applied at inference.
  * Class mapping is read from `class_mapping.json`, which is written by
    training from the actual dataset ordering — never hardcoded here.
  * Runs on CPU by default if CUDA is unavailable; uses GPU when present.
  * Accepts a path, a PIL image, or a numpy array; handles JPG/PNG/RGBA.

Pipeline contract (see the research presentation): if this returns
`Non_larva`, the caller must STOP and not run Stage 2.
"""

import json
import os
import threading

import numpy as np
import torch
from PIL import Image

from stage1 import dataset as data_mod
from stage1 import model as model_mod

Image.MAX_IMAGE_PIXELS = None

_S1_ROOT = os.path.join("results", "stage1", "resnet50")
_SELECTED = os.path.join(_S1_ROOT, "selected_model.json")

DEFAULT_MAPPING = os.path.join(_S1_ROOT, "class_mapping.json")


def _selected():
    """
    Resolve the model chosen on VALIDATION performance.

    `selected_model.json` is written after training and records which run won on
    validation macro F1. Reading it here means the demo always uses the
    validation-selected model rather than whichever directory happens to be
    named "baseline".
    """
    if os.path.exists(_SELECTED):
        try:
            with open(_SELECTED, encoding="utf-8") as fh:
                s = json.load(fh)
            if os.path.exists(s.get("checkpoint", "")):
                return s["checkpoint"], s.get("config", "")
        except Exception:
            pass
    return (os.path.join(_S1_ROOT, "baseline", "best_model.pth"),
            os.path.join(_S1_ROOT, "baseline", "config.json"))


DEFAULT_CKPT, DEFAULT_CONFIG = _selected()

_MODEL = None
_META = None
_LOCK = threading.Lock()


def _resolve(path, fallbacks):
    if path and os.path.exists(path):
        return path
    for f in fallbacks:
        if os.path.exists(f):
            return f
    return path


def load_model(checkpoint=None, mapping_path=None, config_path=None, device=None,
               verbose=False):
    """
    Load (once) and cache the Stage 1 ResNet50. Returns (model, meta).

    Safe to call repeatedly; subsequent calls return the cached model.
    """
    global _MODEL, _META
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _META

        ckpt = _resolve(checkpoint, [
            DEFAULT_CKPT,
            os.path.join("demo_models", "stage1_resnet50", "best_model.pth"),
        ])
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found at {ckpt}. Train it first with "
                "`python -m stage1.train`.")

        cfg_path = _resolve(config_path, [
            DEFAULT_CONFIG,
            os.path.join("demo_models", "stage1_resnet50", "model_config.json"),
        ])
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)

        map_path = _resolve(mapping_path, [
            DEFAULT_MAPPING,
            os.path.join("demo_models", "stage1_resnet50", "class_mapping.json"),
        ])
        if os.path.exists(map_path):
            with open(map_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            class_names = [raw[str(i)] for i in range(len(raw))]
        else:
            # Never guess silently: fall back to the training module's own order.
            class_names = list(data_mod.CLASS_NAMES)

        dev = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        img_size = int(cfg.get("img_size", model_mod.INPUT_SIZE))
        dropout = float(cfg.get("dropout", 0.4))

        model = model_mod.build_resnet50(num_classes=len(class_names),
                                         dropout=dropout, pretrained=False, device=dev)
        state = torch.load(ckpt, map_location=dev, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Exactly the transform used for val/test — deterministic, no augmentation.
        preprocess = None
        if cfg.get("preprocessing") == "gamma":
            from preprocessing import GammaCorrectionTransform
            preprocess = [GammaCorrectionTransform(gamma=float(cfg.get("gamma", 1.5)))]
        transform = data_mod.eval_transform(img_size, preprocess)

        _MODEL = model
        _META = {"device": dev, "class_names": class_names, "transform": transform,
                 "img_size": img_size, "checkpoint": ckpt.replace("\\", "/"),
                 "preprocessing": cfg.get("preprocessing", "none")}
        if verbose:
            print(f"[stage1/inference] loaded {ckpt} on {dev}; classes={class_names}")
        return _MODEL, _META


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as im:
            return im.convert("RGB")     # handles JPG, PNG, RGBA, palette
    raise TypeError(f"unsupported image type: {type(image)}")


def predict_stage1(image, checkpoint=None, device=None):
    """
    Classify one image as Larva or Non_larva.

    Accepts a file path, a PIL.Image, or a HxWx3 numpy array.

    Returns:
        {
            "class": "Larva" | "Non_larva",
            "confidence": float,               # probability of the predicted class
            "probabilities": {"Larva": float, "Non_larva": float},
            "class_index": int,
            "is_larva": bool,                  # convenience gate for the pipeline
            "run_stage2": bool,                # False -> demo must stop here
        }
    """
    model, meta = load_model(checkpoint=checkpoint, device=device)
    img = _to_pil(image)
    x = meta["transform"](img).unsqueeze(0).to(meta["device"])

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

    idx = int(probs.argmax())
    names = meta["class_names"]
    return {
        "class": names[idx],
        "confidence": float(probs[idx]),
        "probabilities": {n: float(p) for n, p in zip(names, probs)},
        "class_index": idx,
        "is_larva": names[idx] == "Larva",
        "run_stage2": names[idx] == "Larva",
    }


def predict_batch(images, checkpoint=None, device=None):
    """Convenience wrapper; the model is still loaded only once."""
    return [predict_stage1(im, checkpoint=checkpoint, device=device) for im in images]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m stage1.inference <image> [<image> ...]")
        raise SystemExit(1)
    for p in sys.argv[1:]:
        r = predict_stage1(p)
        gate = "-> run Stage 2" if r["run_stage2"] else "-> STOP (Stage 2 not executed)"
        print(f"{p}: {r['class']} ({r['confidence']:.4f}) {gate}")
