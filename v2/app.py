"""
Safe Zone AI v2 — web app.

    venv/Scripts/python.exe -m streamlit run v2/app.py

Set SAFEZONE_MODE=lite for the single-backbone build (196 MB instead of
352 MB) — recommended for free hosting.

The app deliberately shows what the model does NOT know. A PHI officer acting
on a confident wrong answer produces a bad surveillance record; being told to
retake the photo costs ten seconds. So an uncertain result is rendered as an
instruction, not as a species.

Why the layout branches on photo count
--------------------------------------
One photo and many photos need different shapes, and forcing them into one
shape is what made the first version hard to read. One photo is a detail view:
large verdict, per-class breakdown, the image beside it. Many photos is a
register: a wide table the officer scans and exports. The batch table was
previously rendered inside a half-width column, which squeezed nine columns
into roughly 600 px; it now gets the full page width, and the four raw
probability columns live in an expander instead of competing with the verdict.
"""

import csv as _csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image

from v2 import pipeline

# "auto" rather than "expanded": on a phone Streamlit renders an expanded
# sidebar as a full-screen overlay, so forcing it open hid the whole app behind
# the accuracy panel until the visitor found the close arrow.
st.set_page_config(page_title="Safe Zone AI — Larva Classifier V2",
                   page_icon="🦟", layout="wide",
                   initial_sidebar_state="auto")

# The verdict colour is carried by a pill and a left border, never by the
# large text itself, which inherits the theme's own foreground colour. That
# keeps the result legible whether the viewer is on the light or the dark
# theme — a dark green heading on a dark background was the alternative.
st.markdown("""
<style>
  :root {
    --ok:#1b7f3b; --warn:#c25e00; --grey:#5f6368;
    --line:rgba(128,128,128,.24); --soft:rgba(128,128,128,.09);
  }
  @media (prefers-color-scheme: dark) {
    :root { --ok:#5cc76a; --warn:#ffa94d; --grey:#9aa0a6; }
  }
  .block-container { padding-top: 2.9rem; max-width: 1240px; }

  .band { background:linear-gradient(100deg,#1b5e20 0%,#2e7d32 45%,#43a047 100%);
          border-radius:14px; padding:1.15rem 1.5rem 1.25rem;
          margin:0 0 1.5rem 0; box-shadow:0 2px 14px rgba(0,0,0,.16); }
  .band h1 { color:#fff; font-size:2rem; font-weight:800; margin:0 0 .25rem 0;
             letter-spacing:-.02em; text-shadow:0 1px 2px rgba(0,0,0,.22); }
  .band p  { color:rgba(255,255,255,.93); margin:0; font-size:.98rem; }

  .step { display:flex; align-items:center; gap:.6rem; font-size:1rem;
          font-weight:700; margin:.1rem 0 .85rem; }
  .step .num { display:inline-flex; align-items:center; justify-content:center;
               width:1.5rem; height:1.5rem; border-radius:50%;
               background:var(--ok); color:#fff; font-size:.82rem;
               font-weight:800; flex:none; }

  .card { border:1px solid var(--line); border-left:5px solid var(--grey);
          border-radius:12px; padding:1rem 1.2rem 1.15rem;
          background:var(--soft); }
  .card.ok   { border-left-color:var(--ok); }
  .card.warn { border-left-color:var(--warn); }
  .verdict { font-size:2.05rem; font-weight:800; letter-spacing:-.025em;
             line-height:1.15; margin:.4rem 0 .15rem; }
  .conf { font-size:.88rem; opacity:.75; }

  .pill { display:inline-block; font-size:.67rem; font-weight:800;
          letter-spacing:.09em; padding:.2rem .55rem; border-radius:20px;
          text-transform:uppercase; color:#fff; }
  .p-ok{background:var(--ok);} .p-warn{background:var(--warn);}
  .p-grey{background:var(--grey);}

  .bar { margin:.55rem 0 .7rem; }
  .bar-head { display:flex; justify-content:space-between; font-size:.85rem;
              margin-bottom:.28rem; }
  .track { height:9px; border-radius:6px; background:var(--soft);
           border:1px solid var(--line); overflow:hidden; }
  .fill { height:100%; background:var(--grey); }
  .fill.top { background:var(--ok); }

  .lbl { font-size:.7rem; font-weight:700; letter-spacing:.12em;
         text-transform:uppercase; opacity:.55; margin-bottom:.3rem; }
  .meta { font-size:.78rem; opacity:.68; line-height:1.55; }

  /* st.columns stacks below ~640 px, which turned four metrics into half a
     screen of scrolling on a phone before any result was visible. This strip
     wraps to a 2x2 grid instead of stacking. */
  .stats { display:flex; flex-wrap:wrap; gap:.6rem; margin:.1rem 0 1rem; }
  .stat { flex:1 1 110px; border:1px solid var(--line); border-radius:10px;
          padding:.6rem .75rem; background:var(--soft); }
  .stat .n { font-size:1.7rem; font-weight:800; line-height:1.1; }
  .stat .k { font-size:.72rem; font-weight:700; letter-spacing:.07em;
             text-transform:uppercase; opacity:.6; margin-top:.15rem; }
  .stat.ok .n   { color:var(--ok); }
  .stat.warn .n { color:var(--warn); }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="band"><h1>🦟 Safe Zone AI — Larva Classifier V2</h1>'
    '<p>Photograph a mosquito larva with your phone — on a hand, on tissue '
    'paper, or in a tray of water — and get the genus.</p></div>',
    unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (first run takes ~20 s) …")
def _load():
    st_ = pipeline.load()
    return {"mode": st_["mode"], "threshold": st_["threshold"],
            "models": [m["name"] for m in st_["stage2"]],
            "device": str(st_["device"]),
            "calibration": st_["calibration"]}


def _step(n, text):
    return '<div class="step"><span class="num">%d</span>%s</div>' % (n, text)


info = _load()
cal = info["calibration"]
tf = (cal.get("test") or {}).get("test_field") or {}
tl = (cal.get("test") or {}).get("test_lab") or {}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### How accurate is this?")
    if tf:
        st.metric("Field photos — all images",
                  "%.1f%%" % (100 * tf["full"]["accuracy"]))
        st.metric("Field photos — when confident",
                  "%.1f%%" % (100 * tf["gated"]["accuracy"]),
                  help="on the %.0f%% of photos it answers"
                       % (100 * tf["coverage"]))
        st.metric("Laboratory / microscope images",
                  "%.1f%%" % (100 * tl["full"]["accuracy"]))
    st.markdown(
        '<div class="meta">Measured on %d held-out real smartphone photographs '
        'and %d laboratory images. Neither set was used for training or for '
        'tuning the confidence threshold.</div>'
        % ((tf.get("full") or {}).get("n", 0), (tl.get("full") or {}).get("n", 0)),
        unsafe_allow_html=True)

    # Settings live here rather than in the main column so the working area is
    # only ever "photos in, answers out".
    with st.expander("⚙️ Advanced settings"):
        thr = st.slider("Confidence threshold", 0.0, 0.999,
                        float(info["threshold"]), 0.01,
                        help="Higher = answers fewer photos, more accurately.")
        force = st.checkbox(
            "Always show a best guess", value=False,
            help="Ignores the threshold. The threshold exists because a "
                 "confident wrong genus is worse than asking for a retake — "
                 "turn this on only to inspect what the model was leaning "
                 "toward.")
        if abs(thr - float(info["threshold"])) > 1e-9:
            st.caption("Calibrated value is %.2f." % info["threshold"])

    st.markdown('<div class="lbl" style="margin-top:1.3rem">Best results</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="meta">Put the larva in a plain white tray or on tissue '
        'paper, hold the phone 10–15 cm away, use macro mode if you have it, '
        'and avoid glare off the water. Take 2–3 shots of each specimen — if '
        'one comes back uncertain, another often will not.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="lbl" style="margin-top:1.3rem">Model</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="meta">Stage 1 ResNet-50 → Stage 2 %s<br>'
        'mode <b>%s</b> · device %s · abstains below %.2f confidence</div>'
        % (", ".join(info["models"]), info["mode"], info["device"],
           info["threshold"]),
        unsafe_allow_html=True)

# ----------------------------------------------------------- step 1: input
with st.container(border=True):
    st.markdown(_step(1, "Add photos"), unsafe_allow_html=True)
    up = st.file_uploader(
        "Upload photos", type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True, label_visibility="collapsed",
        help="Select many at once — a whole field visit can be processed "
             "in one go.")
    # Gated behind a toggle rather than an expander: Streamlit executes an
    # expander's body even while it is collapsed, so st.camera_input would
    # fire a browser camera-permission prompt on page load for every visitor
    # who only ever wanted to upload a file.
    use_cam = st.toggle("📷 Use this device's camera instead",
                        help="Opens the camera. Most officers photograph in "
                             "the field and upload from a laptop afterwards.")
    cam = st.camera_input("Camera", label_visibility="collapsed") if use_cam else None

    files = ([cam] if cam is not None else list(up or []))

    if len(files) > 1:
        st.markdown('<div class="meta" style="margin-top:.5rem">'
                    '<b>%d photos queued.</b></div>' % len(files),
                    unsafe_allow_html=True)
        with st.expander("Preview queued photos"):
            imgs = [Image.open(io.BytesIO(f.getvalue())).convert("RGB")
                    for f in files]
            for start in range(0, len(imgs), 8):
                for col, (im, f) in zip(st.columns(8),
                                        list(zip(imgs, files))[start:start + 8]):
                    with col:
                        st.image(im, use_container_width=True)
                        st.caption(f.name)


def _render_single(r, thr, force):
    """Detail view for one photo: verdict card, then the class breakdown."""
    s1 = r["stage1"]
    if not s1["is_mosquito"]:
        st.markdown(
            '<div class="card"><span class="pill p-grey">Stage 1 · rejected</span>'
            '<div class="verdict">Not a mosquito larva</div>'
            '<div class="conf">%.0f%% confident — the species classifier was '
            'not run.</div></div>' % (100 * s1["confidence"]),
            unsafe_allow_html=True)
        st.info("If you believe this IS a mosquito larva, move closer and "
                "retake so the larva fills more of the frame.", icon="ℹ️")
    elif r["abstained"] and not force:
        st.markdown(
            '<div class="card warn"><span class="pill p-warn">Retake</span>'
            '<div class="verdict">Not confident enough</div>'
            '<div class="conf">Leaning <b>%s</b> at %.0f%%, below the %.0f%% '
            'threshold.</div></div>'
            % (pipeline.DISPLAY.get(r.get("best_guess", ""), "—"),
               100 * r["confidence"], 100 * thr),
            unsafe_allow_html=True)
        st.warning("Move closer, steady the phone, and make sure the whole "
                   "larva is in focus.", icon="📷")
    else:
        cls = "card warn" if r["abstained"] else "card ok"
        pill = ("p-warn", "Below threshold") if r["abstained"] else ("p-ok", "Classified")
        st.markdown(
            '<div class="%s"><span class="pill %s">%s</span>'
            '<div class="verdict">%s</div>'
            '<div class="conf">%.0f%% confidence</div></div>'
            % (cls, pill[0], pill[1], r["final_label"], 100 * r["confidence"]),
            unsafe_allow_html=True)
        if r["abstained"]:
            st.caption("Shown because “always show a best guess” is on.")

    if r["stage2_executed"]:
        st.markdown('<div class="lbl" style="margin-top:1.3rem">All classes</div>',
                    unsafe_allow_html=True)
        ranked = sorted(r["stage2"]["probabilities"].items(), key=lambda kv: -kv[1])
        bars = []
        for i, (k, v) in enumerate(ranked):
            pct = 100 * float(v)
            bars.append(
                '<div class="bar"><div class="bar-head"><span>%s</span>'
                '<span><b>%.1f%%</b></span></div>'
                '<div class="track"><div class="fill%s" style="width:%.1f%%">'
                '</div></div></div>'
                % (pipeline.DISPLAY.get(k, k), pct,
                   " top" if i == 0 else "", pct))
        st.markdown("".join(bars), unsafe_allow_html=True)

    st.markdown('<div class="meta" style="margin-top:.9rem">Stage 1 %s (%.0f%%) '
                '· %.0f ms</div>'
                % (s1["class"], 100 * s1["confidence"], r["total_ms"]),
                unsafe_allow_html=True)


def _row(name, r):
    """One record of the batch table. Keys here are also the CSV columns."""
    if not r["stage1"]["is_mosquito"]:
        verdict, action = "Not a mosquito larva", "skip"
    elif r["abstained"]:
        verdict, action = "Uncertain", "retake"
    else:
        verdict, action = r["final_label"], "record"
    p = (r["stage2"] or {}).get("probabilities", {})
    return {"photo": name, "result": verdict,
            "confidence": round(r["confidence"], 3), "action": action,
            "stage1": "%s (%.2f)" % (r["stage1"]["class"],
                                     r["stage1"]["confidence"]),
            "aedes": round(p.get("aedes", 0), 3),
            "anopheles": round(p.get("anopheles", 0), 3),
            "culex": round(p.get("culex", 0), 3),
            "unknown_objects": round(p.get("unknown_objects", 0), 3)}


@st.cache_data(show_spinner=False, max_entries=512)
def _classify_bytes(data, name, thr, force):
    """
    Classify one uploaded photo, memoised on (bytes, threshold, force).

    Streamlit re-runs this whole script on every widget interaction, so
    without the cache, changing the table filter would re-run inference on
    every photo in the batch — minutes of waiting on a CPU machine just to
    look at a subset. Changing the threshold or the force flag is part of the
    key, so those correctly recompute.
    """
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return _row(name, pipeline.classify(im, threshold=thr, force_answer=force))


NEXT_STEP = {"record": "✅ Record", "retake": "📷 Retake", "skip": "⚪ Skip"}

# --------------------------------------------------------- step 2: results
st.markdown(_step(2, "Results"), unsafe_allow_html=True)

if not files:
    st.info("Upload photos above — or use the camera — to classify them.",
            icon="⬆️")

elif len(files) == 1:
    img = Image.open(io.BytesIO(files[0].getvalue())).convert("RGB")
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.image(img, caption="%d × %d" % img.size, use_container_width=True)
    with right:
        with st.spinner("Classifying …"):
            r = pipeline.classify(img, threshold=thr, force_answer=force)
        _render_single(r, thr, force)

else:
    # Batch: a field visit produces many photos and the officer is at a desk
    # when they process them. A full-width table plus a CSV is what a
    # surveillance record actually needs.
    bar = st.progress(0.0, text="Classifying %d photos …" % len(files))
    rows = []
    for i, f in enumerate(files):
        rows.append(_classify_bytes(f.getvalue(), f.name, thr, force))
        bar.progress((i + 1) / len(files),
                     text="Classifying %d/%d …" % (i + 1, len(files)))
    bar.empty()

    n_ok = sum(1 for r in rows if r["action"] == "record")
    n_retake = sum(1 for r in rows if r["action"] == "retake")
    n_skip = len(rows) - n_ok - n_retake

    st.markdown(
        '<div class="stats">'
        '<div class="stat"><div class="n">%d</div><div class="k">Photos</div></div>'
        '<div class="stat ok"><div class="n">%d</div><div class="k">Classified</div></div>'
        '<div class="stat warn"><div class="n">%d</div><div class="k">Need retake</div></div>'
        '<div class="stat"><div class="n">%d</div><div class="k">Not a larva</div></div>'
        '</div>' % (len(rows), n_ok, n_retake, n_skip),
        unsafe_allow_html=True)

    if n_retake:
        st.warning("%d photo(s) were not confident enough. If the specimens are "
                   "still available, retake those; otherwise record them as "
                   "unidentified rather than guessing." % n_retake, icon="📷")

    choice = st.segmented_control(
        "Show", ["All", "Classified", "Need retake", "Not a larva"],
        default="All", label_visibility="collapsed")
    keep = {"Classified": "record", "Need retake": "retake",
            "Not a larva": "skip"}.get(choice)
    shown = [r for r in rows if keep is None or r["action"] == keep]

    if not shown:
        st.info("No photos in this category.", icon="🔍")
    else:
        st.dataframe(
            [{"photo": r["photo"], "result": r["result"],
              # ProgressColumn formats the raw value, so confidence is scaled
              # to 0-100 for display while the CSV keeps the 0-1 probability.
              "confidence": 100 * r["confidence"],
              "next": NEXT_STEP[r["action"]]} for r in shown],
            use_container_width=True, hide_index=True,
            column_config={
                # "medium", not "large": on a phone a large filename column
                # consumed the whole grid and pushed Result — the one column
                # that matters — off the right edge.
                "photo": st.column_config.TextColumn("Photo", width="medium"),
                "result": st.column_config.TextColumn("Result", width="medium"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%.0f%%",
                    width="medium"),
                "next": st.column_config.TextColumn("Next step",
                                                    width="small")})

    with st.expander("Per-class probabilities"):
        st.dataframe(
            [{"photo": r["photo"], "Aedes": r["aedes"],
              "Anopheles": r["anopheles"], "Culex": r["culex"],
              "Unknown object": r["unknown_objects"], "Stage 1": r["stage1"]}
             for r in shown],
            use_container_width=True, hide_index=True)
        st.caption("These are the model's probabilities, not certainties.")

    buf = io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    st.download_button("⬇️  Download all %d results as CSV" % len(rows),
                       buf.getvalue(), file_name="larva_results.csv",
                       mime="text/csv", use_container_width=True)
    st.caption("The CSV always contains every photo, including the ones "
               "filtered out of the view above. Keep the original photos — "
               "they are the record, and they are also the training data for "
               "a future Sri Lankan model.")

st.markdown("---")
st.caption("Safe Zone AI v2 · ICT 4808 Group 08, Rajarata University of Sri "
           "Lanka · research prototype, not a diagnostic device. Field "
           "photographs sourced from iNaturalist under CC licences — see "
           "ATTRIBUTIONS.md.")
