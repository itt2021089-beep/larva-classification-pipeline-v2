"""
v2 step 1 - fetch field-realistic mosquito larva photographs from iNaturalist.

    python -m v2.fetch_inat --plan          # show what would be fetched
    python -m v2.fetch_inat --run

Why this exists
---------------
Every image in the v1 dataset is a laboratory or microscope capture. The
deployment target is a PHI officer photographing larvae on a hand, on tissue
paper or in a tray with a phone. v1 has no examples of that, which is the
domain gap no amount of augmentation closed.

iNaturalist observations are overwhelmingly smartphone photographs taken in
exactly those conditions, and 6,983 mosquito observations carry an explicit
life-stage = larva annotation. That is the missing half of this project.

Provenance is recorded for every single image: observation id, taxon, rank,
quality grade, licence, attribution string, place and original dimensions.
CC-BY and CC-BY-SA require attribution, so the attribution string is kept
verbatim and written into ATTRIBUTIONS.md.

Grouping rule
-------------
All photos from ONE observation are one specimen in one session. They share an
`obs_<id>` group and must never be split across train/val/test. That is
enforced downstream by v2.curate, but the group id is assigned here.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

API = "https://api.inaturalist.org/v1/observations"
OUT = os.path.join("v2", "raw_inat")
UA = "SafeZoneAI-research/1.0 (ICT4808 Rajarata University; mosquito larva classification)"

# life stage annotation: term_id 1 = Life Stage, value 6 = Larva
LARVA = "term_id=1&term_value_id=6"
LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc"

# OBSERVER_CAP exists because of a real failure found on the first fetch.
# Requesting research-grade Aedes first pulled in a single Mexican health
# jurisdiction that uploads systematic microscope photographs: 72% of the Aedes
# images came from ONE observer and 69% carried a circular eyepiece vignette,
# against 6-14% for every other class. A model trained on that learns
# "dark corners and a bright circle => Aedes" - perfect validation scores,
# total field failure. Capping per-observer contribution and dropping the
# research-grade preference forces the class to span many cameras and setups.
OBSERVER_CAP = 20   # raised from 12: at 20 a single uploader is still <2% of a 1000-image class

TARGETS = [
    # (folder, taxon_id, label, max observations, prefer research grade)
    ("anopheles", 146949, "anopheles", 400, False),   # bottleneck class: take all
    ("culex",     130033, "culex",     600, False),
    ("aedes",      62989, "aedes",    1600, False),   # broad, NOT research-first;
    # Aedes<->Culex is 51% of all field errors, so both get more data
    # negatives: what a PHI officer will actually photograph by mistake.
    # Chironomid (non-biting midge) larvae are THE classic field confuser -
    # same water, similar size, wriggle the same way.
    ("unknown_chironomid", 53275, "unknown_objects", 250, False),
]

PER_PAGE = 100
SLEEP_API = 1.1          # iNaturalist asks for <= 1 request/second
SLEEP_IMG = 0.12


def api_get(url, tries=4):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception as exc:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    return {}


def fetch_observations(taxon_id, cap, research_first):
    """
    Page through observations. iNaturalist caps deep paging at 10k, which we
    never approach. Research-grade is requested first when the class has enough
    of it, because a research-grade identification means the community agreed;
    `needs_id` is one person's guess and gets visually verified later.
    """
    seen, out = set(), []
    per_observer = Counter()
    passes = ([("quality_grade=research", True), ("", False)] if research_first
              else [("", False)])
    for extra, is_research in passes:
        page = 1
        while len(out) < cap:
            url = ("%s?taxon_id=%d&%s&photo_license=%s&per_page=%d&page=%d&order_by=id"
                   % (API, taxon_id, LARVA, LICENSES, PER_PAGE, page))
            if extra:
                url += "&" + extra
            d = api_get(url)
            res = d.get("results") or []
            if not res:
                break
            for o in res:
                if o["id"] in seen:
                    continue
                who = ((o.get("user") or {}).get("login")
                       or (o.get("user") or {}).get("id") or "?")
                if per_observer[who] >= OBSERVER_CAP:
                    continue          # keep one uploader from defining a class
                seen.add(o["id"])
                per_observer[who] += 1
                out.append(o)
                if len(out) >= cap:
                    break
            # Page until the POOL is exhausted or `cap` observations are KEPT.
            # Observations skipped by the observer cap must not consume the
            # budget, or a class dominated by one uploader ends up tiny.
            if page * PER_PAGE >= d.get("total_results", 0) or page >= 60:
                break
            page += 1
            time.sleep(SLEEP_API)
        time.sleep(SLEEP_API)
    return out


def photo_url(p, size="medium"):
    """
    iNat serves square/small/medium/large/original from one stem.

    "medium" is ~500px on the long side. Everything is resized to 224 before
    the model sees it, so large (1024px) buys nothing measurable and costs
    roughly 4x the bytes - which on a slow link is the difference between a
    40-minute fetch and a 3-hour one.
    """
    u = p.get("url") or ""
    for s in ("square", "small", "medium", "thumb"):
        u = u.replace("/" + s + ".", "/" + size + ".")
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-photos-per-obs", type=int, default=3)
    a = ap.parse_args()
    if not (a.plan or a.run):
        a.plan = True

    os.makedirs(OUT, exist_ok=True)
    manifest, summary = [], {}

    for folder, taxon_id, label, cap, research_first in TARGETS:
        print("\n=== %s (taxon %d) ===" % (folder, taxon_id))
        obs = fetch_observations(taxon_id, cap, research_first)
        qg = Counter(o.get("quality_grade") for o in obs)
        rk = Counter((o.get("taxon") or {}).get("rank") for o in obs)
        ob = Counter(((o.get("user") or {}).get("login") or "?") for o in obs)
        top = ob.most_common(1)[0] if ob else ("-", 0)
        nphotos = sum(min(len(o.get("photos") or []), a.max_photos_per_obs)
                      for o in obs)
        print("  observations: %d   photos to fetch: %d" % (len(obs), nphotos))
        print("  quality grade: %s" % dict(qg))
        print("  id rank      : %s" % dict(rk))
        print("  observers    : %d distinct, top contributes %d (%.0f%%)"
              % (len(ob), top[1], 100.0 * top[1] / max(len(obs), 1)))
        summary[folder] = {"observations": len(obs), "photos": nphotos,
                           "quality_grade": dict(qg), "rank": dict(rk),
                           "label": label, "taxon_id": taxon_id}

        if not a.run:
            continue

        d = os.path.join(OUT, folder)
        os.makedirs(d, exist_ok=True)
        got = 0
        for o in obs:
            photos = (o.get("photos") or [])[:a.max_photos_per_obs]
            for pi, p in enumerate(photos):
                url = photo_url(p)
                if not url:
                    continue
                ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
                fn = "inat_%d_%d%s" % (o["id"], p["id"], ext)
                dst = os.path.join(d, fn)
                if not os.path.exists(dst):
                    try:
                        req = urllib.request.Request(
                            url, headers={"User-Agent": UA})
                        with urllib.request.urlopen(req, timeout=90) as r:
                            data = r.read()
                        if len(data) < 4000:      # truncated / placeholder
                            continue
                        with open(dst, "wb") as fh:
                            fh.write(data)
                    except Exception as exc:
                        print("    [warn] %s: %s" % (fn, type(exc).__name__))
                        continue
                    time.sleep(SLEEP_IMG)
                got += 1
                t = o.get("taxon") or {}
                dim = p.get("original_dimensions") or {}
                manifest.append({
                    "path": dst.replace("\\", "/"),
                    "filename": fn,
                    "label": label,
                    "folder": folder,
                    "group_id": "obs_%d" % o["id"],
                    "observation_id": o["id"],
                    "photo_id": p["id"],
                    "photo_index_in_obs": pi,
                    "taxon_name": t.get("name"),
                    "taxon_rank": t.get("rank"),
                    "quality_grade": o.get("quality_grade"),
                    "license": p.get("license_code"),
                    "attribution": p.get("attribution"),
                    "place_guess": (o.get("place_guess") or "")[:120],
                    "observed_on": o.get("observed_on"),
                    "orig_width": dim.get("width"), "orig_height": dim.get("height"),
                    "source": "inaturalist",
                    "source_url": "https://www.inaturalist.org/observations/%d" % o["id"],
                })
            if got and got % 100 == 0:
                print("    %d photos ..." % got)
        print("  downloaded: %d" % got)

    if a.plan and not a.run:
        tot = sum(v["photos"] for v in summary.values())
        print("\nPLAN TOTAL: %d photos across %d classes" % (tot, len(summary)))
        print("re-run with --run to download")
        return 0

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    # attribution file - CC-BY / CC-BY-SA require it
    lic = Counter(m["license"] for m in manifest)
    with open(os.path.join(OUT, "ATTRIBUTIONS.md"), "w", encoding="utf-8") as fh:
        fh.write("# Image attributions\n\n")
        fh.write("Photographs retrieved from iNaturalist. Each line is one "
                 "image, its observation, and the licence and attribution the "
                 "photographer chose. CC-BY and CC-BY-SA require this "
                 "attribution to be preserved wherever the images are used.\n\n")
        for k, v in lic.most_common():
            fh.write("- %s: %d images\n" % (k, v))
        fh.write("\n---\n\n")
        for m in manifest:
            fh.write("- `%s` — %s — %s — %s\n"
                     % (m["filename"], m["license"], m["attribution"],
                        m["source_url"]))

    json.dump({"summary": summary, "downloaded": len(manifest),
               "licenses": dict(lic),
               "note": "Raw download. NOT verified, NOT deduped, NOT split. "
                       "v2.curate does that."},
              open(os.path.join(OUT, "fetch_summary.json"), "w",
                   encoding="utf-8"), indent=2)
    print("\ntotal downloaded: %d" % len(manifest))
    print("licences: %s" % dict(lic))
    print("[out] " + OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
