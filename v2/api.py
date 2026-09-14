"""
Safe Zone AI v2 - HTTP API.

    venv/Scripts/python.exe -m uvicorn v2.api:app --host 0.0.0.0 --port 8000

    POST /classify      multipart file upload -> JSON prediction
    GET  /health        model status
    GET  /info          what is loaded, and the measured accuracy behind it
    GET  /docs          interactive Swagger UI (FastAPI built-in)

Set SAFEZONE_MODE=lite for the single-backbone build (196 MB instead of
352 MB, ~3x faster, 0.003 field macro-F1 lower, with its own calibrated
threshold).

Example:

    curl -F "file=@larva.jpg" http://localhost:8000/classify
"""

import io
import json
import os
import sys
import time

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from v2 import pipeline

MAX_BYTES = 20 * 1024 * 1024          # 20 MB - a phone photo is ~2-5 MB
ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = FastAPI(
    title="Safe Zone AI — mosquito larva classifier",
    version="2.0",
    description=(
        "Two-stage classifier for mosquito larvae in smartphone photographs.\n\n"
        "**Measured accuracy.** On 381 held-out real smartphone photographs the "
        "pipeline is 72.2% accurate when forced to answer every image, and "
        "89.9% accurate on the 49% it is confident enough to answer. On "
        "laboratory/microscope images it is 93.6% accurate. It has NOT been "
        "validated on Sri Lankan specimens.\n\n"
        "A result with `abstained: true` means *retake the photo* — it is not "
        "a prediction."),
)

# Open CORS so a phone browser or a friend's page can call this during
# testing. Tighten `allow_origins` before anything resembling production.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.on_event("startup")
def _warm():
    try:
        st = pipeline.load()
        print("[api] mode=%s  stage2=%s  threshold=%.2f  device=%s"
              % (st["mode"],
                 ",".join(m["backbone"] for m in st["stage2"]),
                 st["threshold"], st["device"]))
    except Exception as exc:                      # keep /health informative
        print("[api] model load FAILED: %r" % (exc,))


@app.get("/health")
def health():
    try:
        st = pipeline.load()
        return {"status": "ok", "mode": st["mode"],
                "device": str(st["device"]),
                "stage2_models": [m["name"] for m in st["stage2"]],
                "abstention_threshold": st["threshold"]}
    except Exception as exc:
        return JSONResponse(status_code=503,
                            content={"status": "model_unavailable",
                                     "error": repr(exc)})


@app.get("/info")
def info():
    st = pipeline.load()
    cal = st["calibration"]
    tf = (cal.get("test") or {}).get("test_field") or {}
    tl = (cal.get("test") or {}).get("test_lab") or {}
    return {
        "mode": st["mode"],
        "stage1": "ResNet-50, trained on laboratory + field images",
        "stage2": [{"name": m["name"], "backbone": m["backbone"],
                    "input_size": m["size"]} for m in st["stage2"]],
        "classes": pipeline.S2_CLASSES,
        "abstention_threshold": st["threshold"],
        "measured": {
            "field_test_n": (tf.get("full") or {}).get("n"),
            "field_accuracy_all_images": (tf.get("full") or {}).get("accuracy"),
            "field_accuracy_when_confident": (tf.get("gated") or {}).get("accuracy"),
            "field_coverage": tf.get("coverage"),
            "lab_accuracy": (tl.get("full") or {}).get("accuracy"),
        },
        "caveats": [
            "Field accuracy is 72% across all photos; ~90% on the subset the "
            "model is confident about.",
            "No Sri Lankan specimens in training or evaluation.",
            "Aedes and Culex are the main confusion; Anopheles is reliable.",
            "Labels are citizen-science identifications, not entomologist-verified.",
        ],
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...),
                   threshold: float = Query(
                       None, ge=0.0, le=1.0,
                       description="override the calibrated abstention threshold"),
                   force_answer: bool = Query(
                       False,
                       description="return the best guess even when not confident")):
    if file.content_type not in ALLOWED:
        raise HTTPException(415, "unsupported content type %r; send a JPEG, PNG, "
                                 "WEBP or BMP image" % file.content_type)
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than %d MB" % (MAX_BYTES // 1048576))
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "could not decode the image")

    t0 = time.perf_counter()
    try:
        res = pipeline.classify(img, threshold=threshold,
                                force_answer=force_answer)
    except Exception as exc:
        raise HTTPException(500, "inference failed: %r" % (exc,))
    res["filename"] = file.filename
    res["server_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("v2.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    sys.exit(main())
