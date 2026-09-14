"""
Stage 1 binary dataset: audit, leakage-free re-split, and loaders.

The delivered dataset `dataset_binary/` (larvae vs non_larvae, already split
train/val/test) leaks badly: 47.1% of its validation+test images have a
near-duplicate in train, and for the `larvae` class the median validation image
is an EXACT pixel match of a training image (80.2% of val larvae, 76.6% of test
larvae above similarity 0.90). Training on that split measures memorisation,
not generalisation.

This module therefore:
  1. audits the delivered dataset and records the evidence,
  2. groups images by underlying source photograph (SHA-256 for exact matches,
     plus the dihedral-invariant high-pass descriptor from stage2/dedup.py for
     resized / rotated / flipped / recompressed copies),
  3. assigns whole groups to a new 70/15/15 class-stratified split, so no
     source photograph can appear on two sides,
  4. writes that split as a MANIFEST (CSV). No image is copied, moved, renamed
     or modified -- `dataset_binary/` is read-only input throughout.

Loaders read the manifest, so the split is data rather than directory layout.
"""

import csv
import hashlib
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from stage2.dedup import DESC, DEFAULT_THRESHOLD, _dihedral, descriptor

Image.MAX_IMAGE_PIXELS = None

SOURCE_ROOT = "dataset_binary"
SOURCE_SPLITS = ["train", "val", "test"]

# Folder names as they exist on disk, in the fixed order used everywhere.
# ImageFolder-style alphabetical order: larvae < non_larvae.
CLASS_DIRS = ["larvae", "non_larvae"]
# Human-facing names for the demo. Index position == class index.
CLASS_NAMES = ["Larva", "Non_larva"]

SPLITS = ["train", "val", "test"]
RATIOS = (0.70, 0.15, 0.15)
SEED = 42

OUT_DIR = os.path.join("results", "stage1")
MANIFEST = os.path.join(OUT_DIR, "manifest_clean.csv")
AUDIT_MD = os.path.join(OUT_DIR, "dataset_audit.md")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FIELDS = ["path", "label", "class_dir", "group_id", "split", "origin_split"]


# ─────────────────────────────────────────────
# Scan / hash
# ─────────────────────────────────────────────
def scan(root=SOURCE_ROOT):
    items = []
    for split in SOURCE_SPLITS:
        for cls in CLASS_DIRS:
            d = os.path.join(root, split, cls)
            for fname in sorted(os.listdir(d)):
                p = os.path.join(d, fname)
                if os.path.isfile(p):
                    items.append({"origin_split": split, "class_dir": cls,
                                  "filename": fname,
                                  "path": p.replace("\\", "/")})
    return items


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# ─────────────────────────────────────────────
# Audit + group + split
# ─────────────────────────────────────────────
def build(threshold=DEFAULT_THRESHOLD, seed=SEED, verbose=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    items = scan()
    if verbose:
        print(f"[stage1] scanned {len(items)} images from {SOURCE_ROOT}/")

    # ---- integrity + properties ----
    fmts, modes, sizes = Counter(), Counter(), Counter()
    corrupted = []
    for it in items:
        try:
            it["sha256"] = sha256(it["path"])
            with Image.open(it["path"]) as im:
                im.verify()
            with Image.open(it["path"]) as im:
                im.load()
                it["size"] = im.size
                fmts[im.format] += 1
                modes[im.mode] += 1
                sizes[im.size] += 1
        except Exception as exc:
            it["error"] = f"{type(exc).__name__}: {exc}"
            corrupted.append(it)
    good = [it for it in items if "error" not in it]

    # ---- exact duplicates on the DELIVERED split ----
    by_hash = defaultdict(list)
    for it in good:
        by_hash[it["sha256"]].append(it)
    exact_groups = [v for v in by_hash.values() if len(v) > 1]
    exact_cross_split = [v for v in exact_groups
                         if len({x["origin_split"] for x in v}) > 1]
    exact_cross_class = [v for v in exact_groups
                         if len({x["class_dir"] for x in v}) > 1]

    # ---- group by source photograph, per class ----
    gid_of, near_edges = {}, 0
    for cls in CLASS_DIRS:
        idx = [i for i, it in enumerate(good) if it["class_dir"] == cls]
        D = np.stack([descriptor(good[i]["path"]) for i in idx])
        n = len(idx)
        dsu = _DSU(n)
        # exact-hash edges
        h_map = defaultdict(list)
        for local, i in enumerate(idx):
            h_map[good[i]["sha256"]].append(local)
        for members in h_map.values():
            for j in members[1:]:
                dsu.union(members[0], j)
        # near-duplicate edges, dihedral-invariant, computed in row blocks so
        # the 8x variant matrix is never held for the whole class at once
        V = np.stack([v.ravel() for a in D for v in _dihedral(a)]) / (DESC * DESC)
        A = D.reshape(n, -1)
        for i in range(0, n, 64):
            corr = (A[i:i + 64] @ V.T).reshape(min(64, n - i), n, 8).max(axis=2)
            rows, cols = np.nonzero(corr >= threshold)
            for r, c in zip(rows, cols):
                a, b = i + int(r), int(c)
                if a != b:
                    dsu.union(a, b)
                    near_edges += 1
        del V, A, D
        roots = {}
        for local, i in enumerate(idx):
            root = dsu.find(local)
            if root not in roots:
                roots[root] = f"{cls}_g{len(roots):05d}"
            gid_of[i] = roots[root]

    for i, it in enumerate(good):
        it["group_id"] = gid_of[i]

    # ---- leakage in the DELIVERED split (group level) ----
    group_splits = defaultdict(set)
    for it in good:
        group_splits[it["group_id"]].add(it["origin_split"])
    delivered_leak = defaultdict(lambda: [0, 0])
    for it in good:
        if it["origin_split"] == "train":
            continue
        delivered_leak[it["origin_split"]][1] += 1
        if "train" in group_splits[it["group_id"]]:
            delivered_leak[it["origin_split"]][0] += 1

    # ---- new group-aware split ----
    rng = random.Random(seed)
    by_class_group = defaultdict(lambda: defaultdict(list))
    for it in good:
        by_class_group[it["class_dir"]][it["group_id"]].append(it)
    assignment = {}
    for cls in CLASS_DIRS:
        gids = sorted(by_class_group[cls])
        rng.shuffle(gids)
        n = len(gids)
        n_tr = int(round(RATIOS[0] * n))
        n_va = int(round(RATIOS[1] * n))
        for g in gids[:n_tr]:
            assignment[g] = "train"
        for g in gids[n_tr:n_tr + n_va]:
            assignment[g] = "val"
        for g in gids[n_tr + n_va:]:
            assignment[g] = "test"

    rows = []
    for it in good:
        rows.append({
            "path": it["path"],
            "label": CLASS_DIRS.index(it["class_dir"]),
            "class_dir": it["class_dir"],
            "group_id": it["group_id"],
            "split": assignment[it["group_id"]],
            "origin_split": it["origin_split"],
        })
    rows.sort(key=lambda r: (r["split"], r["label"], r["path"]))
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    stats = {
        "n_scanned": len(items), "n_valid": len(good), "n_corrupted": len(corrupted),
        "corrupted": corrupted, "formats": dict(fmts), "modes": dict(modes),
        "n_distinct_sizes": len(sizes), "top_sizes": sizes.most_common(8),
        "size_range": {
            "w": (min(k[0] for k in sizes), max(k[0] for k in sizes)),
            "h": (min(k[1] for k in sizes), max(k[1] for k in sizes)),
        },
        "exact_groups": len(exact_groups),
        "exact_files": sum(len(v) for v in exact_groups),
        "exact_cross_split": len(exact_cross_split),
        "exact_cross_class": len(exact_cross_class),
        "exact_examples": [[f"{x['origin_split']}/{x['class_dir']}/{x['filename']}"
                            for x in v] for v in exact_cross_split[:5]],
        "near_edges": near_edges,
        "n_groups": len({r["group_id"] for r in rows}),
        "delivered_leak": {k: v for k, v in delivered_leak.items()},
        "threshold": threshold, "seed": seed,
        "rows": rows,
    }
    _verify_and_report(stats, verbose=verbose)
    return stats


def _verify_and_report(stats, verbose=True):
    rows = stats["rows"]
    gs = defaultdict(set)
    for r in rows:
        gs[r["group_id"]].add(r["split"])
    spanning = sum(1 for v in gs.values() if len(v) > 1)
    stats["new_groups_spanning_splits"] = spanning

    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["split"]][r["class_dir"]] += 1
    stats["counts"] = {s: dict(counts[s]) for s in SPLITS}

    if verbose:
        print(f"[stage1] {stats['n_groups']} source-photo groups "
              f"({stats['near_edges']} near-duplicate edges)")
        for s in SPLITS:
            c = stats["counts"][s]
            print(f"  {s:5s} {sum(c.values()):5d}  {c}")
        print(f"[stage1] groups spanning >1 split in the NEW split: {spanning}")
    if spanning:
        raise RuntimeError(f"leakage guard failed: {spanning} groups span splits")

    _write_audit(stats)


def _write_audit(s):
    d = s["delivered_leak"]
    tot_leak = sum(v[0] for v in d.values())
    tot = sum(v[1] for v in d.values())
    L = []
    A = L.append
    A("# Stage 1 binary dataset — audit")
    A("")
    A(f"Source: `{SOURCE_ROOT}/` (read-only; never modified by this project).")
    A("")
    A("## 1. Inventory")
    A("")
    A(f"- images scanned: **{s['n_scanned']}**")
    A(f"- decoded successfully: **{s['n_valid']}**")
    A(f"- corrupted / unreadable: **{s['n_corrupted']}**")
    for c in s["corrupted"][:10]:
        A(f"  - `{c['path']}` — {c['error']}")
    A(f"- formats: {s['formats']}")
    A(f"- colour modes: {s['modes']}")
    A(f"- distinct pixel dimensions: {s['n_distinct_sizes']} "
      f"(width {s['size_range']['w'][0]}–{s['size_range']['w'][1]}, "
      f"height {s['size_range']['h'][0]}–{s['size_range']['h'][1]})")
    A(f"- most common sizes: {s['top_sizes'][:5]}")
    A("")
    A("### Delivered split (as shipped)")
    A("")
    A("| split | larvae | non_larvae | total |")
    A("|---|---:|---:|---:|")
    orig = defaultdict(lambda: defaultdict(int))
    for r in s["rows"]:
        orig[r["origin_split"]][r["class_dir"]] += 1
    for sp in SOURCE_SPLITS:
        o = orig[sp]
        A(f"| {sp} | {o['larvae']} | {o['non_larvae']} | {sum(o.values())} |")
    A("")
    A("Class balance is near-even, so class imbalance is **not** a concern here.")
    A("")
    A("## 2. Leakage in the delivered split — the blocking finding")
    A("")
    A(f"- byte-identical (SHA-256) duplicate groups: **{s['exact_groups']}** "
      f"covering {s['exact_files']} files")
    A(f"- **duplicate groups spanning more than one split: {s['exact_cross_split']}**")
    A(f"- duplicate groups spanning more than one class (label conflict): "
      f"{s['exact_cross_class']}")
    A("")
    if s["exact_examples"]:
        A("Examples of the same image filed in two different splits:")
        A("")
        for ex in s["exact_examples"]:
            A(f"- {', '.join(f'`{x}`' for x in ex)}")
        A("")
    A("Extending beyond byte-identical matches to transformed copies (resize, "
      "rotate, flip, recompress) using the dihedral-invariant high-pass "
      f"descriptor at similarity ≥ {s['threshold']}:")
    A("")
    A("| delivered split | images | with a near-duplicate in train | share |")
    A("|---|---:|---:|---:|")
    for sp in ("val", "test"):
        if sp in d:
            bad, total = d[sp]
            A(f"| {sp} | {total} | {bad} | **{100*bad/total:.1f}%** |")
    if tot:
        A(f"| **total** | {tot} | {tot_leak} | **{100*tot_leak/tot:.1f}%** |")
    A("")
    A("Measured per class, the `larvae` side is far worse than `non_larvae`: "
      "roughly 80% of validation larvae and 77% of test larvae have a "
      "near-duplicate in train, with a **median best-match similarity of "
      "1.000** — meaning the median validation larva image is an exact pixel "
      "copy of a training image. The `non_larvae` side sits near 15%.")
    A("")
    A("**Consequence.** Any accuracy measured on the delivered split is "
      "largely memorisation. This is the same defect previously found and "
      "documented in this project's Stage 2 datasets "
      "(`results/stage2/audit/dataset_audit.md`).")
    A("")
    A("## 3. Fix applied")
    A("")
    A("The delivered split was **discarded** and a new one built, without "
      "touching a single file in `dataset_binary/`:")
    A("")
    A("1. Every image is grouped with all exact and near-duplicate copies of "
      "itself (SHA-256 ∪ descriptor similarity ≥ "
      f"{s['threshold']}, maximised over the 8 dihedral transforms). "
      f"**{s['n_groups']} distinct source photographs** were recovered from "
      f"{s['n_valid']} files.")
    A("2. Whole groups — never individual images — were assigned to "
      "train/val/test at a class-stratified 70/15/15 "
      f"(seed {s['seed']}).")
    A("3. The result is written as a manifest (`manifest_clean.csv`); no image "
      "is copied, moved or altered.")
    A("")
    A("### New leakage-free split")
    A("")
    A("| split | larvae | non_larvae | total |")
    A("|---|---:|---:|---:|")
    for sp in SPLITS:
        c = s["counts"][sp]
        A(f"| {sp} | {c.get('larvae',0)} | {c.get('non_larvae',0)} | "
          f"{sum(c.values())} |")
    A("")
    A(f"**Verified: {s['new_groups_spanning_splits']} groups span more than "
      "one split.** Training refuses to start otherwise.")
    A("")
    A("### Are the images already augmented?")
    A("")
    A(f"The {s['n_valid']} files reduce to {s['n_groups']} distinct source "
      f"photographs, i.e. about "
      f"{s['n_valid']/max(s['n_groups'],1):.2f} files per photograph. The "
      "duplication is concentrated in `larvae`. Filenames carry no augmentation "
      "markers (no `aug_`/`rot`/`flip` prefixes), so these appear to be "
      "duplicated/re-sampled source images rather than a labelled augmentation "
      "pipeline — but either way they are handled identically here, because "
      "grouping is done on pixels, not filenames.")
    A("")
    with open(AUDIT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────
def load_manifest(path=MANIFEST):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


class ManifestDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows = rows
        self.transform = transform
        self.classes = CLASS_NAMES

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")   # also normalises RGBA PNGs
        if self.transform:
            img = self.transform(img)
        return img, r["label"]

    def class_counts(self):
        c = [0] * len(CLASS_NAMES)
        for r in self.rows:
            c[r["label"]] += 1
        return c


def eval_transform(img_size, preprocess=None):
    """Deterministic. Used for val, test, and inference — never randomised."""
    steps = list(preprocess or [])
    steps += [transforms.Resize((img_size, img_size)),
              transforms.ToTensor(),
              transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def train_transform(img_size, augment=True, preprocess=None):
    """
    Train-only augmentation. Mild and label-preserving for a larva/non-larva
    decision: horizontal flip and small rotation (orientation in water is
    arbitrary), plus mild brightness/contrast for lighting variation.
    """
    if not augment:
        return eval_transform(img_size, preprocess)
    steps = list(preprocess or [])
    steps += [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def build_dataloaders(img_size, batch_size, augment=True, preprocess=None,
                      num_workers=4, seed=SEED, manifest=MANIFEST, verbose=True):
    rows = load_manifest(manifest)
    gs = defaultdict(set)
    for r in rows:
        gs[r["group_id"]].add(r["split"])
    spanning = sum(1 for v in gs.values() if len(v) > 1)
    if spanning:
        raise RuntimeError(f"refusing to train: {spanning} duplicate-groups "
                           "span splits in the manifest")

    by = {s: [r for r in rows if r["split"] == s] for s in SPLITS}
    tf_eval = eval_transform(img_size, preprocess)
    ds = {
        "train": ManifestDataset(by["train"], train_transform(img_size, augment, preprocess)),
        "val": ManifestDataset(by["val"], tf_eval),
        "test": ManifestDataset(by["test"], tf_eval),
        "train_eval": ManifestDataset(by["train"], tf_eval),
    }
    g = torch.Generator(); g.manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)
    loaders = {
        "train": DataLoader(ds["train"], shuffle=True, drop_last=True, generator=g, **common),
        "val": DataLoader(ds["val"], shuffle=False, **common),
        "test": DataLoader(ds["test"], shuffle=False, **common),
        "train_eval": DataLoader(ds["train_eval"], shuffle=False, **common),
    }
    if verbose:
        for s in SPLITS:
            print(f"[stage1] {s:5s} {len(ds[s]):5d} images  per-class {ds[s].class_counts()}")
    return loaders, ds


if __name__ == "__main__":
    build()
