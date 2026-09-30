# Safe Zone AI v2 — running and sharing the app

Two entry points, both driven by the same `v2.pipeline`:

| | what | command |
|---|---|---|
| **Web app** | Streamlit UI, upload or phone camera | `venv/Scripts/python.exe -m streamlit run v2/app.py` |
| **HTTP API** | `POST /classify`, for apps and scripts | `venv/Scripts/python.exe -m uvicorn v2.api:app --host 0.0.0.0 --port 8000` |

The API also serves interactive docs at `http://localhost:8000/docs`.

```bash
curl -F "file=@larva.jpg" http://localhost:8000/classify
```

---

## Two model modes

```bash
# full — 3-model ensemble (default)
venv/Scripts/python.exe -m streamlit run v2/app.py

# lite — single backbone, for constrained hosting
SAFEZONE_MODE=lite venv/Scripts/python.exe -m streamlit run v2/app.py
```

| mode | checkpoints | size | field val macro-F1 | abstention threshold |
|---|---|---|---|---|
| `full` | ResNet-50 + ConvNeXt-T + EffNetV2-S + EffNetV2-S@320 | **352 MB** | 0.7781 | 0.91 |
| `lite` | ResNet-50 + ConvNeXt-T | **196 MB** | 0.7749 | 0.99 |

**Lite runs ~3× faster, and costs more than validation suggested.** On field
validation it gave up only 0.003 macro-F1, but on the field **test** set the
whole pipeline is 68.2% accurate against 72.2% for full — about 4 points. Use
full wherever the hardware allows. Its threshold is calibrated
separately (`v2/calibrate_lite.py`) because a single model's confidence
distribution is not the ensemble's — reusing the ensemble's 0.91 would silently
move the operating point.

---

## Sharing it with someone else

### Option A — local + tunnel (fastest, use this to let a friend test today)

Run the app locally and expose it with a tunnel. Your GPU does the work, so
inference stays fast and nothing needs uploading.

```bash
venv/Scripts/python.exe -m streamlit run v2/app.py --server.port 8503
```

Then in a second terminal, with [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/):

```bash
cloudflared tunnel --url http://localhost:8503
```

It prints a public `https://<random>.trycloudflare.com` URL. Send that. It
works on a phone browser, including the camera input. The link dies when you
stop the tunnel, and your machine has to stay on.

### Option B — Hugging Face Spaces (best real hosting for this)

Spaces is the right home for a model-heavy app: free CPU tier, ~16 GB RAM, and
Git LFS with limits that comfortably fit 196–352 MB of checkpoints.

1. Create a Space → SDK **Streamlit**.
2. Push `v2/`, `stage1/`, `results/v2/`, `requirements-app.txt` (rename to
   `requirements.txt`), and set `app_file: v2/app.py` in the Space README
   front-matter.
3. Track checkpoints with LFS before the first push:
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   ```
4. Set `SAFEZONE_MODE=lite` in the Space's variables. Free Spaces are CPU-only;
   lite keeps first-load and per-image latency reasonable.

### Option C — Streamlit Community Cloud

Workable, but with a real obstacle: **`convnext_tiny/best_model.pth` is 106 MB
and GitHub rejects files over 100 MB.** So you must either use Git LFS (free
tier gives 1 GB storage and 1 GB bandwidth per month — a handful of app cold
starts can exhaust the bandwidth) or download the checkpoints at startup from a
GitHub Release or Drive link.

If you go this route, use `SAFEZONE_MODE=lite`; the full ensemble's 352 MB plus
CPU PyTorch is likely to exceed the free tier's memory.

---

## What the app deliberately does

It shows an **uncertain** result as an instruction ("retake the photo"), not as
a species. Measured end to end — the larva check and the species classifier
together — on the 381-photo held-out field test set, the full build answers
51% of photos at 88.7% accuracy; forced to answer everything it is 72.2%
accurate. The lite build is lower: 68.2%, and 84.7% on the 48% it answers.
A PHI officer acting on a confident wrong genus produces a bad surveillance
record, while a retake costs ten seconds.

Both figures come from `results/v2/final/end_to_end.json`
(`python -m v2.eval_end_to_end`), which the app sidebar, the API description
and the package README all read — do not type accuracy figures in by hand.

Stage 1 no longer rejects every image it leans against. When it is unsure it
defers to Stage 2, which can still say "retake"; the rejection threshold is
calibrated on validation by `python -m v2.calibrate_gate`.

The sidebar states the measured accuracy and the known limitations — no Sri
Lankan specimens, Aedes/Culex confusability, citizen-science labels. Please
leave that in when you share it.

---

## Notes

* First request loads ~350 MB of weights and takes 15–25 s. Every request after
  is fast (~200 ms on GPU, 1–3 s on CPU).
* The API opens CORS to `*` so a phone browser or a friend's page can call it
  during testing. Tighten `allow_origins` in `v2/api.py` before anything
  resembling production.
* `MAX_BYTES` in `v2/api.py` caps uploads at 20 MB.
* Both entry points honour `SAFEZONE_MODE`.
