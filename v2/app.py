"""
Safe Zone AI v2 — web app.

    venv/Scripts/python.exe -m streamlit run v2/app.py

Set SAFEZONE_MODE=lite for the single-backbone build (196 MB instead of
352 MB) — recommended for free hosting.

The app deliberately shows what the model does NOT know. A PHI officer acting
on a confident wrong answer produces a bad surveillance record; being told to
retake the photo costs ten seconds. So an uncertain result is rendered as an
instruction, not as a species.
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image

from v2 import pipeline

st.set_page_config(page_title="Safe Zone AI — Larva Classifier v2.0",
                   page_icon="🦟", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container { padding-top: 4rem; max-width: 1180px; }
  .band { background: linear-gradient(100deg,#1b5e20 0%,#2e7d32 45%,#43a047 100%);
          border-radius: 14px; padding: 1.1rem 1.5rem 1.2rem;
          margin: 0 0 1.3rem 0; box-shadow: 0 2px 12px rgba(0,0,0,.16); }
  .band h1 { color:#fff; font-size:2.1rem; font-weight:800; margin:0 0 .2rem 0;
             letter-spacing:-.02em; text-shadow:0 1px 2px rgba(0,0,0,.22); }
  .band p  { color:rgba(255,255,255,.92); margin:0; font-size:1rem; }
  .verdict { font-size:2.4rem; font-weight:800; letter-spacing:-.03em;
             line-height:1.1; margin:.2rem 0; }
  .v-ok   { color:#2e7d32; } .v-warn { color:#ef6c00; } .v-grey { color:#616161; }
  .sub    { font-size:.85rem; opacity:.7; }
  .lbl    { font-size:.72rem; font-weight:700; letter-spacing:.12em;
            text-transform:uppercase; opacity:.55; margin-bottom:.2rem; }
  .pill   { display:inline-block; font-size:.68rem; font-weight:700;
            letter-spacing:.05em; padding:.16rem .5rem; border-radius:20px;
            text-transform:uppercase; }
  .p-ok{background:rgba(46,125,50,.16);color:#2e7d32;}
  .p-warn{background:rgba(239,108,0,.16);color:#ef6c00;}
  .p-info{background:rgba(21,101,192,.16);color:#1565c0;}
  .meta   { font-size:.78rem; opacity:.68; line-height:1.5; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="band"><h1>🦟 Safe Zone AI — Larva Classifier v2.0</h1>'
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


info = _load()
cal = info["calibration"]
tf = (cal.get("test") or {}).get("test_field") or {}
tl = (cal.get("test") or {}).get("test_lab") or {}

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

    st.markdown('<div class="lbl" style="margin-top:1.4rem">What it cannot do</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="meta">'
        '• <b>No Sri Lankan specimens</b> in training or evaluation. The field '
        'photographs are mostly North American and European.<br>'
        '• <b>Aedes and Culex are confusable</b> — both have a siphon and '
        'differ mainly in its length. Anopheles is the reliable one.<br>'
        '• Labels come from citizen-science identifications, not from an '
        'entomologist.<br>'
        '• This is a research prototype, not a diagnostic device.'
        '</div>', unsafe_allow_html=True)

    st.markdown('<div class="lbl" style="margin-top:1.4rem">Model</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="meta">Stage 1 ResNet-50 → Stage 2 %s<br>'
        'mode <b>%s</b> · device %s · abstains below %.2f confidence</div>'
        % (", ".join(info["models"]), info["mode"], info["device"],
           info["threshold"]),
        unsafe_allow_html=True)

    st.markdown('<div class="lbl" style="margin-top:1.4rem">Best results</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="meta">Put the larva in a plain white tray or on tissue '
        'paper, hold the phone 10–15 cm away, use macro mode if you have it, '
        'and avoid glare off the water.</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 1.15], gap="large")

with c1:
    st.markdown('<div class="lbl">1 · Photo</div>', unsafe_allow_html=True)
    up = st.file_uploader(
        "Upload photos", type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True, label_visibility="collapsed",
        help="Select many at once — a whole field visit can be processed "
             "in one go.")
    cam = st.camera_input("or take one now", label_visibility="visible")
    files = ([cam] if cam is not None else list(up or []))
    src = files[0] if files else None
    if len(files) == 1:
        img = Image.open(io.BytesIO(files[0].getvalue())).convert("RGB")
        st.image(img, caption="%d × %d" % img.size, use_container_width=True)
    elif len(files) > 1:
        st.success("%d photos queued." % len(files), icon="🗂️")
        st.image([Image.open(io.BytesIO(f.getvalue())).convert("RGB")
                  for f in files[:6]],
                 width=96,
                 caption=[f.name[:14] for f in files[:6]])

    with st.expander("Advanced"):
        force = st.checkbox(
            "Always show a best guess (ignore the confidence threshold)",
            value=False,
            help="The threshold exists because a confident wrong genus is "
                 "worse than asking for a retake. Turn this on only to inspect "
                 "what the model was leaning toward.")
        thr = st.slider("Confidence threshold", 0.0, 0.999,
                        float(info["threshold"]), 0.01,
                        help="Higher = answers fewer photos, more accurately.")


def _render_single(r, thr, force):
    """Full detail view for one photo."""
    s1 = r["stage1"]
    if not s1["is_mosquito"]:
        st.markdown('<div class="verdict v-grey">Not a mosquito larva</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sub">Stage 1 confidence %.0f%% — the species '
                    'classifier was not run.</div>' % (100 * s1["confidence"]),
                    unsafe_allow_html=True)
        st.info("If you believe this IS a mosquito larva, move closer and "
                "retake so the larva fills more of the frame.", icon="ℹ️")
    elif r["abstained"] and not force:
        st.markdown('<div class="verdict v-warn">Retake the photo</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sub">Best guess was <b>%s</b> at %.0f%% '
                    'confidence, below the %.0f%% threshold.</div>'
                    % (pipeline.DISPLAY.get(r.get("best_guess", ""), "—"),
                       100 * r["confidence"], 100 * thr),
                    unsafe_allow_html=True)
        st.warning("Move closer, steady the phone, and make sure the whole "
                   "larva is in focus.", icon="📷")
    else:
        st.markdown('<div class="verdict v-ok">%s</div>' % r["final_label"],
                    unsafe_allow_html=True)
        tag = ("p-ok" if not r["abstained"] else "p-warn")
        st.markdown('<span class="pill %s">%.0f%% confidence</span>'
                    % (tag, 100 * r["confidence"]), unsafe_allow_html=True)
        if r["abstained"]:
            st.caption("Below the threshold — shown because “always show a "
                       "best guess” is on.")

    if r["stage2_executed"]:
        st.markdown('<div class="lbl" style="margin-top:1.2rem">All classes</div>',
                    unsafe_allow_html=True)
        for k, v in sorted(r["stage2"]["probabilities"].items(),
                           key=lambda kv: -kv[1]):
            st.markdown("**%s** — %.1f%%" % (pipeline.DISPLAY.get(k, k), 100 * v))
            st.progress(min(max(float(v), 0.0), 1.0))

    st.markdown('<div class="meta" style="margin-top:1rem">Stage 1 %s (%.0f%%) '
                '· %.0f ms</div>'
                % (s1["class"], 100 * s1["confidence"], r["total_ms"]),
                unsafe_allow_html=True)


def _row(name, r):
    """One line of the batch table."""
    if not r["stage1"]["is_mosquito"]:
        verdict, action = "Not a mosquito larva", "—"
    elif r["abstained"]:
        verdict, action = "Uncertain", "RETAKE"
    else:
        verdict, action = r["final_label"], "record"
    p = (r["stage2"] or {}).get("probabilities", {})
    return {"photo": name, "result": verdict, "confidence": round(r["confidence"], 3),
            "action": action,
            "stage1": "%s (%.2f)" % (r["stage1"]["class"], r["stage1"]["confidence"]),
            "aedes": round(p.get("aedes", 0), 3),
            "anopheles": round(p.get("anopheles", 0), 3),
            "culex": round(p.get("culex", 0), 3),
            "unknown_objects": round(p.get("unknown_objects", 0), 3)}


with c2:
    st.markdown('<div class="lbl">2 · Result</div>', unsafe_allow_html=True)
    if not files:
        st.info("Upload or capture a photo to classify it.", icon="⬆️")

    elif len(files) == 1:
        with st.spinner("Classifying …"):
            r = pipeline.classify(img, threshold=thr, force_answer=force)
        _render_single(r, thr, force)

    else:
        # Batch: a field visit produces many photos, and the officer is at a
        # desk when they process them. A table plus a CSV is what a
        # surveillance record actually needs.
        rows, bar = [], st.progress(0.0, text="Classifying %d photos …" % len(files))
        for i, f in enumerate(files):
            im = Image.open(io.BytesIO(f.getvalue())).convert("RGB")
            rows.append(_row(f.name,
                             pipeline.classify(im, threshold=thr,
                                               force_answer=force)))
            bar.progress((i + 1) / len(files),
                         text="Classifying %d/%d …" % (i + 1, len(files)))
        bar.empty()

        n_retake = sum(1 for r in rows if r["action"] == "RETAKE")
        n_ok = sum(1 for r in rows if r["action"] == "record")
        a, b, c = st.columns(3)
        a.metric("Classified", n_ok)
        b.metric("Need retake", n_retake)
        c.metric("Not a larva", len(rows) - n_ok - n_retake)
        if n_retake:
            st.warning("%d photo(s) were not confident enough. If the specimens "
                       "are still available, retake those; otherwise treat them "
                       "as unidentified rather than guessing." % n_retake,
                       icon="📷")

        st.dataframe(rows, use_container_width=True, hide_index=True)

        import csv as _csv
        buf = io.StringIO()
        w = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        st.download_button("Download results as CSV", buf.getvalue(),
                           file_name="larva_results.csv", mime="text/csv",
                           use_container_width=True)
        st.caption("Confidence columns are the model's probabilities, not "
                   "certainties. Keep the original photos — they are the "
                   "record, and they are also the training data for a future "
                   "Sri Lankan model.")

st.markdown("---")
st.caption("Safe Zone AI v2 · ICT 4808 Group 08, Rajarata University of Sri "
           "Lanka · research prototype, not a diagnostic device. Field "
           "photographs sourced from iNaturalist under CC licences — see "
           "ATTRIBUTIONS.md.")
