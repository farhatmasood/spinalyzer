"""
Web application: HTTP routes for studies, commands, finding localisation and
reports.

One study holds up to eight images for a patient: two sagittal (T1, T2) and six
axial (three per weighting). Fewer images are handled the same way; the study
reports on whatever it was given and states what could not be derived.

Runs entirely on the local machine. No image or result leaves the process.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import commands
from . import engine
from . import findings as findings_mod
from . import render
from . import report as report_mod
from . import study as study_mod

STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLES_DIR = engine.REPO_ROOT / "samples"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("spinalyzer")

app = FastAPI(title="Spinalyzer", version=__version__,
              docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR), check_dir=False),
          name="samples")


@app.on_event("startup")
def _startup() -> None:
    log.info("Spinalyzer %s", __version__)
    engine.load_all()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"version": __version__, **engine.status(),
            "limits": {"max_images": study_mod.MAX_IMAGES,
                       "max_sagittal": study_mod.MAX_SAGITTAL,
                       "max_axial": study_mod.MAX_AXIAL}}


@app.get("/api/samples")
def samples() -> Dict[str, List[str]]:
    return {"files": sorted(p.name for p in SAMPLES_DIR.glob("*.png"))}


# ---------------------------------------------------------------------------
# Study payload
# ---------------------------------------------------------------------------

def _series_payload(sr, st) -> Dict[str, Any]:
    dm = {d["label"]: study_mod.disc_metrics(d, sr) for d in sr.discs}
    vm = {v["label"]: study_mod.vertebra_metrics(v, sr) for v in sr.vertebrae}
    regions = study_mod.abnormal_regions(st)
    abnormal = [r["level"] for r in regions] if sr is st.primary_sagittal() else []
    ann = render.Annotations(title=sr.label())
    return {
        "id": sr.id,
        "filename": sr.filename,
        "format": sr.fmt,
        "plane": sr.plane,
        "plane_source": sr.plane_source,
        "weighting": sr.weighting,
        "weighting_source": sr.weighting_source,
        "level": sr.level,
        "level_source": sr.level_source,
        "label": sr.label(),
        "spacing": sr.spacing,
        "mm_available": sr.mm_factors() is not None,
        "shape": list(sr.rgb.shape),
        "original": render.preview(sr.rgb),
        "overlay": render.render_series(sr, ann, dm, vm, abnormal),
        "discs": [study_mod.disc_metrics(d, sr) for d in sr.discs],
        "vertebrae": [study_mod.vertebra_metrics(v, sr) for v in sr.vertebrae],
        "has_sacrum": sr.has_sacrum,
        "grading": [{**g, "level": l} for l, g in sr.grading.items() if "error" not in g],
        "findings": sr.findings,
    }


def _study_payload(st) -> Dict[str, Any]:
    ref = st.primary_sagittal()
    regions = study_mod.abnormal_regions(st)
    return {
        "study_id": st.id,
        "series": [_series_payload(s, st) for s in st.series],
        "reference_series_id": ref.id if ref else None,
        "disc_inventory": st.disc_inventory(),
        "abnormal_regions": regions,
        "alignment": [{"series_id": s.id, "filename": s.filename,
                       "weighting": s.weighting, "level": s.level,
                       "level_source": s.level_source} for s in st.axial()],
        "counts": {"total": len(st.series), "sagittal": len(st.sagittal()),
                   "axial": len(st.axial())},
        "assignable_levels": ["L3-L4", "L4-L5", "L5-S1", "L2-L3", "L1-L2"],
    }


def _get_study(study_id: str):
    st = study_mod.STUDIES.get(study_id)
    if st is None:
        raise HTTPException(404, "Study not found or expired. Upload the images again.")
    return st


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/study")
async def create_study(
    files: List[UploadFile] = File(...),
    planes: Optional[str] = Form(None),
    weightings: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Create a study from one to eight images.

    `planes` and `weightings` are optional comma-separated overrides, one entry
    per file, used when the operator states what a file is.
    """
    if not files:
        raise HTTPException(400, "No file received.")
    if len(files) > study_mod.MAX_IMAGES:
        raise HTTPException(400, f"A study takes at most {study_mod.MAX_IMAGES} images; "
                                 f"{len(files)} were sent.")
    if engine.REG.seg is None:
        raise HTTPException(503, "Segmentation model is not loaded. "
                                 "Check /api/health for the reason.")

    plane_list = [p.strip() or None for p in (planes or "").split(",")] if planes else []
    weight_list = [w.strip() or None for w in (weightings or "").split(",")] if weightings else []

    st = study_mod.new_study()
    errors: List[Dict[str, str]] = []

    for i, up in enumerate(files):
        data = await up.read()
        name = up.filename or f"image_{i + 1}"
        if not data:
            errors.append({"filename": name, "error": "empty file"})
            continue
        try:
            sr = study_mod.build_series(
                data, name,
                plane_override=plane_list[i] if i < len(plane_list) else None,
                weighting_override=weight_list[i] if i < len(weight_list) else None,
            )
        except Exception as e:
            log.exception("failed to build series for %s", name)
            errors.append({"filename": name, "error": str(e)})
            continue
        st.series.append(sr)

    if not st.series:
        raise HTTPException(400, "No image could be read. " +
                            "; ".join(f"{e['filename']}: {e['error']}" for e in errors))

    n_sag = len(st.sagittal())
    n_ax = len(st.axial())
    if n_sag > study_mod.MAX_SAGITTAL:
        errors.append({"filename": "-", "error":
                       f"{n_sag} sagittal images were sent; the study is designed "
                       f"for {study_mod.MAX_SAGITTAL}. All were analysed."})
    if n_ax > study_mod.MAX_AXIAL:
        errors.append({"filename": "-", "error":
                       f"{n_ax} axial images were sent; the study is designed for "
                       f"{study_mod.MAX_AXIAL}. All were analysed."})

    for sr in st.sagittal():
        try:
            study_mod.analyse_sagittal(sr)
        except Exception as e:
            log.exception("analysis failed for %s", sr.filename)
            errors.append({"filename": sr.filename, "error": f"analysis failed: {e}"})

    study_mod.align_axials(st)

    payload = _study_payload(st)
    payload["warnings"] = errors
    return payload


@app.get("/api/study/{study_id}")
def get_study(study_id: str) -> Dict[str, Any]:
    return _study_payload(_get_study(study_id))


@app.post("/api/study/{study_id}/level")
async def set_level(study_id: str, series_id: str = Form(...),
                    level: str = Form("")) -> Dict[str, Any]:
    """Assign a disc level to an axial series by hand."""
    st = _get_study(study_id)
    if not study_mod.set_axial_level(st, series_id, level.strip() or None):
        raise HTTPException(400, "That series is not an axial slice in this study.")
    return _study_payload(st)


@app.post("/api/study/{study_id}/calibrate")
async def calibrate(study_id: str, mm_per_pixel: float = Form(...),
                    series_id: str = Form("")) -> Dict[str, Any]:
    """Set the physical pixel size for one series, or for all when none is named.

    The value applies to the original uploaded image; the resize onto the
    segmentation canvas is corrected for automatically.
    """
    st = _get_study(study_id)
    if mm_per_pixel <= 0:
        raise HTTPException(400, "mm per pixel must be greater than zero.")
    targets = ([st.by_id(series_id)] if series_id else list(st.series))
    if not targets or targets[0] is None:
        raise HTTPException(404, "Series not found in this study.")
    for sr in targets:
        sr.spacing = {"row": float(mm_per_pixel), "col": float(mm_per_pixel),
                      "source": "manual calibration"}
    return _study_payload(st)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

@app.post("/api/study/{study_id}/ask")
async def ask(study_id: str, text: str = Form(...)) -> Dict[str, Any]:
    st = _get_study(study_id)
    try:
        return commands.execute(st, text)
    except Exception as e:
        log.exception("command failed: %s", text)
        raise HTTPException(500, f"Command failed: {e}")


@app.post("/api/study/{study_id}/locate")
async def locate(study_id: str, finding: str = Form(...),
                 threshold: float = Form(0.5)) -> Dict[str, Any]:
    """Mark the structures a named finding describes.

    Backs the Show control beside each row of the findings table.
    """
    st = _get_study(study_id)
    try:
        return commands.locate_finding(st, finding, threshold)
    except Exception as e:
        log.exception("locate failed: %s", finding)
        raise HTTPException(500, f"Locate failed: {e}")


@app.get("/api/findings")
def findings_catalogue() -> Dict[str, Any]:
    """Every finding the models report, and whether it can be marked."""
    return {"findings": findings_mod.catalogue()}


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

@app.get("/api/lexicon")
def lexicon_get() -> Dict[str, Any]:
    return {"aliases": commands.list_aliases()}


@app.post("/api/lexicon")
async def lexicon_post(text: str = Form(...)) -> Dict[str, Any]:
    return commands.lexicon_command(text)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@app.post("/api/study/{study_id}/report/preview")
async def report_preview(study_id: str, options: str = Form("{}")) -> Dict[str, Any]:
    """Render the report images with the chosen layers, without building the PDF."""
    import json
    st = _get_study(study_id)
    opts = report_mod.ReportOptions.from_dict(json.loads(options or "{}"))
    ref = st.primary_sagittal() or (st.series[0] if st.series else None)
    if ref is None:
        raise HTTPException(400, "Study has no image.")

    regions = study_mod.abnormal_regions(st, opts.findings_threshold)
    dm = {d["label"]: study_mod.disc_metrics(d, ref) for d in ref.discs}
    vm = {v["label"]: study_mod.vertebra_metrics(v, ref) for v in ref.vertebrae}
    images = [{
        "series_id": ref.id, "label": ref.label(),
        "image": render.render_series(ref, opts.annotations(ref.label()), dm, vm,
                                      [r["level"] for r in regions]),
    }]
    for s in st.axial():
        adm = {d["label"]: study_mod.disc_metrics(d, s) for d in s.discs}
        images.append({"series_id": s.id, "label": s.label(),
                       "image": render.render_series(s, opts.annotations(s.label()),
                                                     adm, None, [])})
    if opts.abnormal_regions and regions:
        images.append({"series_id": ref.id, "label": "Positive levels",
                       "image": render.render_abnormal_panel(ref, regions)})
    return {"images": images,
            "abnormal_levels": [r["level"] for r in regions],
            "region_count": len(regions)}


@app.post("/api/study/{study_id}/report.pdf")
async def report_pdf(study_id: str, options: str = Form("{}")) -> Response:
    import json
    st = _get_study(study_id)
    opts = report_mod.ReportOptions.from_dict(json.loads(options or "{}"))
    try:
        data = report_mod.build(st, opts)
    except Exception as e:
        log.exception("PDF generation failed")
        raise HTTPException(500, f"PDF generation failed: {e}")
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition":
                             f'attachment; filename="spinalyzer_{st.id}.pdf"'})
