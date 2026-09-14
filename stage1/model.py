"""
ResNet50 backbone for Stage 1 binary classification (Larva vs Non-larva).

Architecture choice is fixed by the research presentation
(`ICT4808_90%_Progress_Presentation.pptx`, slide 13: "STAGE 1 — BINARY
CLASSIFICATION, Model -> ResNet50"). It is not changed here.

Head
----
torchvision's ResNet50 ends in `fc: Linear(2048 -> 1000)` for ImageNet. That
1000-way head is replaced with:

    Dropout(p) -> Linear(2048 -> 2)

Two output logits with CrossEntropyLoss (rather than one logit with BCE) is
deliberate: it matches the rest of this project's convention, and it gives the
demo a proper softmax over BOTH classes, so `predict_stage1` can return a
calibrated probability for each class rather than one number the caller has to
interpret.

Dropout is exposed because the presentation's Stage 1 table treats
DROPOUT_RATE as a tuned hyperparameter (its best run used 0.4).
"""

import torch
from torch import nn
from torchvision import models

INPUT_SIZE = 224
NUM_CLASSES = 2


def build_resnet50(num_classes: int = NUM_CLASSES, dropout: float = 0.4,
                   pretrained: bool = True, device=None):
    """ResNet50 with a dropout + 2-class head."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features            # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    if device is not None:
        model = model.to(device)
    return model


def parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def verify(model, device, img_size: int = INPUT_SIZE, verbose: bool = True):
    """
    Sanity-check shapes and device before training, so a wrong head or a
    silently-CPU run is caught immediately rather than after an epoch.
    """
    model.eval()
    x = torch.zeros(2, 3, img_size, img_size, device=device)
    with torch.no_grad():
        y = model(x)
    trainable, total = parameter_count(model)
    info = {
        "input_shape": tuple(x.shape),
        "output_shape": tuple(y.shape),
        "num_classes": int(y.shape[1]),
        "trainable_params": trainable,
        "total_params": total,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if y.shape[1] != NUM_CLASSES:
        raise RuntimeError(f"head produces {y.shape[1]} outputs, expected {NUM_CLASSES} "
                           "— the ImageNet 1000-class head was not replaced")
    if verbose:
        print(f"[stage1/model] input {info['input_shape']} -> output {info['output_shape']}")
        print(f"[stage1/model] trainable {trainable:,} / {total:,} params on {device}")
        if info["gpu_name"]:
            free, tot = torch.cuda.mem_get_info()
            print(f"[stage1/model] GPU {info['gpu_name']} "
                  f"({free/1e9:.1f} GB free / {tot/1e9:.1f} GB total)")
    return info
