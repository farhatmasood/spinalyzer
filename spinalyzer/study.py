"""
Study model: one patient, one to eight images.

A study holds up to two sagittal series (T1, T2) and up to six axial series
(three T1, three T2). Each axial slice covers a single disc level and is tied
to a level from the sagittal inventory.

Level assignment is explicit about its evidence. Three sources exist, checked
in order:

  filename   a D3/D4/D5 or L4-L5 style token in the filename
  manual     the operator set it in the interface
  order      no evidence in the file, so slices were mapped to the inferior-most
             levels in upload order

The source travels with the assignment and is shown in the interface and the
report, so an order-based mapping is never mistaken for a derived one.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import engine

SEG_IMG_SIZE = engine.SEG_IMG_SIZE

# Labelled from the bottom of the image upward: the inferior-most disc in a
# lumbar sagittal slice is L5-S1.
DISC_ORDER = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2", "T12-L1", "T11-T12"]
VERTEBRA_ORDER = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11"]

# Disc level codes used by the LSMA-PQR training data.
DCODE_TO_LEVEL = {"D3": "L3-L4", "D4": "L4-L5", "D5": "L5-S1"}

MAX_SAGITTAL = 2
MAX_AXIAL = 6
MAX_IMAGES = MAX_SAGITTAL + MAX_AXIAL

STUDY_TTL_S = 4 * 60 * 60
STUDIES: Dict[str, "Study"] = {}


# ---------------------------------------------------------------------------
# Metadata read from the filename
# ---------------------------------------------------------------------------

def detect_weighting(filename: str, spacing: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """Return (weighting, source). 'unspecified' when nothing states it."""
    if spacing and isinstance(spacing.get("dicom"), dict):
        blob = " ".join(str(v) for v in spacing["dicom"].values()).upper()
        if re.search(r"\bT2\b|T2W|T2_", blob):
            return "T2", "DICOM"
        if re.search(r"\bT1\b|T1W|T1_", blob):
            return "T1", "DICOM"
    n = filename.upper()
    if re.search(r"(^|[^A-Z0-9])T2([^A-Z0-9]|$)", n):
        return "T2", "filename"
    if re.search(r"(^|[^A-Z0-9])T1([^A-Z0-9]|$)", n):
        return "T1", "filename"
    return "unspecified", "none"


def detect_plane(filename: str, shape: Tuple[int, ...]) -> Tuple[str, str]:
    """Return (plane, source)."""
    n = filename.lower()
    if "axial" in n or re.search(r"(^|[_-])ax([_-]|\d)", n):
        return "axial", "filename"
    if "sag" in n:
        return "sagittal", "filename"
    h, w = shape[:2]
    return ("sagittal" if h >= w else "axial"), "shape"


def detect_level_token(filename: str) -> Optional[str]:
    """Find an explicit disc level in the filename, or None."""
    n = filename.upper()
    m = re.search(r"([LT])(\d{1,2})\s*[-_/]\s*([LTS])(\d{1,2})", n)
    if m:
        return f"{m.group(1)}{int(m.group(2))}-{m.group(3)}{int(m.group(4))}"
    m = re.search(r"(^|[^A-Z0-9])(D[345])([^A-Z0-9]|$)", n)
    if m:
        return DCODE_TO_LEVEL.get(m.group(2))
    return None


def label_components(components: List[Dict[str, Any]], order: List[str]) -> None:
    """Label bottom-up: the lowest component in the image gets order[0]."""
    for i, comp in enumerate(reversed(components)):
        comp["label"] = order[i] if i < len(order) else f"unlabelled-{i}"


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

@dataclass
class Series:
    id: str
    filename: str
    fmt: str
    plane: str
    plane_source: str
    weighting: str
    weighting_source: str
    rgb: np.ndarray                  # original resolution
    seg_rgb: np.ndarray              # SEG_IMG_SIZE square
    mask: np.ndarray
    class_names: Dict[int, str]
    class_colors: Dict[int, Tuple[int, int, int]]
    discs: List[Dict[str, Any]] = field(default_factory=list)
    vertebrae: List[Dict[str, Any]] = field(default_factory=list)
    has_sacrum: bool = False
    spacing: Optional[Dict[str, Any]] = None
    grading: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    findings: Optional[Dict[str, Any]] = None
    level: Optional[str] = None       # axial only
    level_source: str = "none"        # filename | manual | order | none

    def mm_factors(self) -> Optional[Tuple[float, float]]:
        """mm per row/column on the segmentation canvas, or None if unknown.

        The mask lives on a SEG_IMG_SIZE square, so the original spacing must be
        scaled by the resize ratio before it means anything here.
        """
        if not self.spacing:
            return None
        H, W = self.rgb.shape[:2]
        row = float(self.spacing.get("row", 0)) * (H / SEG_IMG_SIZE)
        col = float(self.spacing.get("col", 0)) * (W / SEG_IMG_SIZE)
        return (row, col) if row > 0 and col > 0 else None

    def label(self) -> str:
        w = self.weighting if self.weighting != "unspecified" else "—"
        if self.plane == "axial":
            return f"Axial {w} · {self.level or 'unassigned'}"
        return f"Sagittal {w}"


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

@dataclass
class Study:
    id: str
    created_at: float
    series: List[Series] = field(default_factory=list)

    def sagittal(self) -> List[Series]:
        return [s for s in self.series if s.plane == "sagittal"]

    def axial(self) -> List[Series]:
        return [s for s in self.series if s.plane == "axial"]

    def by_id(self, sid: str) -> Optional[Series]:
        return next((s for s in self.series if s.id == sid), None)

    def primary_sagittal(self) -> Optional[Series]:
        """Reference series for the level inventory. T2 shows disc hydration
        best, so it is preferred when present; otherwise the first sagittal."""
        sags = self.sagittal()
        if not sags:
            return None
        return next((s for s in sags if s.weighting == "T2"), sags[0])

    def disc_inventory(self) -> List[str]:
        ref = self.primary_sagittal()
        if ref is None:
            return []
        return [d["label"] for d in ref.discs if d.get("label")]

    def axial_for_level(self, level: str) -> List[Series]:
        return [s for s in self.axial() if s.level == level]


def new_study() -> Study:
    _gc()
    st = Study(id=uuid.uuid4().hex[:12], created_at=time.time())
    STUDIES[st.id] = st
    return st


def _gc() -> None:
    now = time.time()
    for k in [k for k, v in STUDIES.items() if now - v.created_at > STUDY_TTL_S]:
        STUDIES.pop(k, None)


# ---------------------------------------------------------------------------
# Building a series
# ---------------------------------------------------------------------------

def build_series(data: bytes, filename: str,
                 plane_override: Optional[str] = None,
                 weighting_override: Optional[str] = None) -> Series:
    rgb, fmt, spacing = engine.load_image(data, filename)

    plane, plane_src = detect_plane(filename, rgb.shape)
    if plane_override in ("sagittal", "axial"):
        plane, plane_src = plane_override, "manual"

    weighting, w_src = detect_weighting(filename, spacing)
    if weighting_override in ("T1", "T2"):
        weighting, w_src = weighting_override, "manual"

    seg_rgb = cv2.resize(rgb, (SEG_IMG_SIZE, SEG_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    mask = engine.run_segmentation(seg_rgb, plane)

    class_names = engine.AXIAL_CLASSES if plane == "axial" else engine.SAGITTAL_CLASSES
    class_colors = engine.AXIAL_COLORS if plane == "axial" else engine.SAGITTAL_COLORS
    name_to_id = {n: cid for cid, n in class_names.items()}

    discs = (engine.connected_components(mask, name_to_id["IVD"])
             if "IVD" in name_to_id else [])
    vertebrae = (engine.connected_components(mask, name_to_id["VB"])
                 if "VB" in name_to_id else [])
    has_sacrum = bool("Sacrum" in name_to_id and (mask == name_to_id["Sacrum"]).any())

    sr = Series(
        id=uuid.uuid4().hex[:8],
        filename=filename, fmt=fmt,
        plane=plane, plane_source=plane_src,
        weighting=weighting, weighting_source=w_src,
        rgb=rgb, seg_rgb=seg_rgb, mask=mask,
        class_names=class_names, class_colors=class_colors,
        discs=discs, vertebrae=vertebrae, has_sacrum=has_sacrum,
        spacing=spacing,
    )

    if plane == "sagittal":
        # A sagittal slice shows the column, so components carry anatomical names.
        label_components(sr.discs, DISC_ORDER)
        label_components(sr.vertebrae, VERTEBRA_ORDER)
    else:
        # An axial slice cuts one level. Its components are not separate
        # anatomical levels, so they are numbered, not named.
        for i, d in enumerate(sr.discs):
            d["label"] = f"disc region {i + 1}"
        tok = detect_level_token(filename)
        if tok:
            sr.level, sr.level_source = tok, "filename"

    return sr


def analyse_sagittal(sr: Series) -> None:
    """Run grading and structured findings on a sagittal series."""
    if sr.plane != "sagittal":
        return
    by_label = {d["label"]: d for d in sr.discs}
    for level, idx in engine.IN_DIST_DISCS.items():
        d = by_label.get(level)
        if d is None:
            continue
        sr.grading[level] = engine.grade_disc(sr.seg_rgb, d, idx)

    if len(sr.discs) >= 3:
        grades = [sr.grading.get(l, {}).get("pfirrmann_grade", 0)
                  for l in ("L3-L4", "L4-L5", "L5-S1")]
        sr.findings = engine.run_findings(sr.seg_rgb, sr.discs, grades)


# ---------------------------------------------------------------------------
# Axial to sagittal level alignment
# ---------------------------------------------------------------------------

def align_axials(st: Study) -> List[Dict[str, Any]]:
    """Give every axial series a disc level. Returns a per-series audit trail.

    Filename evidence wins, then a manual assignment, then upload order. Order
    mapping fills the inferior-most levels of the sagittal inventory, treating
    the upload sequence as superior to inferior -- the usual acquisition
    direction. That mapping is recorded as source 'order' so it reads as a
    default rather than a measurement.
    """
    inventory = st.disc_inventory()          # bottom-up: L5-S1, L4-L5, ...
    audit: List[Dict[str, Any]] = []

    # Only levels the grading head was trained on can carry a grade, so those
    # are the ones worth aligning to. Ordered superior to inferior.
    if inventory:
        candidates = [l for l in ["L3-L4", "L4-L5", "L5-S1"] if l in inventory]
        if not candidates:
            candidates = list(reversed(inventory))[-3:]
    else:
        candidates = ["L3-L4", "L4-L5", "L5-S1"]

    for weighting in ("T1", "T2", "unspecified"):
        group = [s for s in st.axial() if s.weighting == weighting]
        if not group:
            continue

        unresolved = [s for s in group if s.level_source not in ("filename", "manual")]
        taken = {s.level for s in group if s.level_source in ("filename", "manual") and s.level}
        free = [l for l in candidates if l not in taken]

        # Map the tail of the candidate list so a partial stack lands on the
        # lowest levels, which is where a three-slice lumbar study is acquired.
        pool = free[-len(unresolved):] if len(unresolved) <= len(free) else free
        for i, sr in enumerate(unresolved):
            if i < len(pool):
                sr.level, sr.level_source = pool[i], "order"
            else:
                sr.level, sr.level_source = None, "none"

    for sr in st.axial():
        audit.append({
            "series_id": sr.id,
            "filename": sr.filename,
            "weighting": sr.weighting,
            "level": sr.level,
            "level_source": sr.level_source,
        })
    return audit


def set_axial_level(st: Study, series_id: str, level: Optional[str]) -> bool:
    sr = st.by_id(series_id)
    if sr is None or sr.plane != "axial":
        return False
    if level:
        sr.level, sr.level_source = level, "manual"
    else:
        sr.level, sr.level_source = None, "none"
    return True


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def _mm(px: int, factor: Optional[float]) -> Optional[float]:
    return None if factor is None else round(px * factor, 1)


def disc_metrics(d: Dict[str, Any], sr: Series) -> Dict[str, Any]:
    f = sr.mm_factors()
    row = f[0] if f else None
    col = f[1] if f else None
    _, _, w, h = d["bbox"]
    return {
        "label": d.get("label"),
        "height_px": int(h), "height_mm": _mm(h, row),
        "ap_diameter_px": int(w), "ap_diameter_mm": _mm(w, col),
        "area_px": int(d["area"]),
        "area_mm2": (None if not f else round(d["area"] * row * col, 1)),
        "centroid": [round(d["centroid_x"], 1), round(d["centroid_y"], 1)],
        "bbox": list(d["bbox"]),
    }


def vertebra_metrics(v: Dict[str, Any], sr: Series) -> Dict[str, Any]:
    f = sr.mm_factors()
    row = f[0] if f else None
    col = f[1] if f else None
    _, _, w, h = v["bbox"]
    return {
        "label": v.get("label"),
        "height_px": int(h), "height_mm": _mm(h, row),
        "width_px": int(w), "width_mm": _mm(w, col),
        "area_px": int(v["area"]),
        "area_mm2": (None if not f else round(v["area"] * row * col, 1)),
        "centroid": [round(v["centroid_x"], 1), round(v["centroid_y"], 1)],
        "bbox": list(v["bbox"]),
    }


# ---------------------------------------------------------------------------
# Abnormal regions: findings tied to an actual measured component
# ---------------------------------------------------------------------------

def abnormal_regions(st: Study, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Levels carrying a positive pathological finding, with their measurements.

    A region qualifies only when the grading or findings head reports it above
    threshold on a level that exists in the segmentation. Nothing is added for
    a level the models did not score.
    """
    ref = st.primary_sagittal()
    if ref is None:
        return []

    by_label = {d["label"]: d for d in ref.discs}
    regions: List[Dict[str, Any]] = []

    for level, g in ref.grading.items():
        if "error" in g:
            continue
        d = by_label.get(level)
        if d is None:
            continue

        reasons: List[Dict[str, Any]] = []

        # Pfirrmann IV and V are the degenerative grades in the original scale.
        if g.get("pfirrmann_grade", 0) >= 3:
            reasons.append({
                "kind": "Pfirrmann",
                "detail": f"Grade {g['pfirrmann_roman']} — {g['pfirrmann_desc']}",
                "score": None,
                "source": "grading head",
            })
        if g.get("stenosis", 0) >= 1:
            probs = g.get("stenosis_probs") or []
            sc = probs[g["stenosis"]] if g["stenosis"] < len(probs) else None
            reasons.append({
                "kind": "Stenosis",
                "detail": g.get("stenosis_label", ""),
                "score": sc,
                "source": "grading head",
            })
        for name, p in (g.get("pathology_probs") or {}).items():
            if p >= threshold:
                reasons.append({
                    "kind": name.replace("_", " "),
                    "detail": "",
                    "score": round(p, 3),
                    "source": "grading head",
                })

        if reasons:
            regions.append({
                "level": level,
                "series_id": ref.id,
                "bbox": list(d["bbox"]),
                "metrics": disc_metrics(d, ref),
                "reasons": reasons,
                "axial_series": [s.id for s in st.axial_for_level(level)],
            })

    # Level-tagged findings from the structured head, matched to real components.
    if ref.findings and "binary_findings" in ref.findings:
        tagged = {"L3-L4 Involvement": "L3-L4",
                  "L4-L5 Involvement": "L4-L5",
                  "L5-S1 Involvement": "L5-S1",
                  "L4-L5 Compression": "L4-L5",
                  "L5-S1 Compression": "L5-S1"}
        for f in ref.findings["binary_findings"]:
            lvl = tagged.get(f["name"])
            if not lvl or not f["positive"]:
                continue
            d = by_label.get(lvl)
            if d is None:
                continue
            entry = next((r for r in regions if r["level"] == lvl), None)
            if entry is None:
                entry = {
                    "level": lvl, "series_id": ref.id,
                    "bbox": list(d["bbox"]), "metrics": disc_metrics(d, ref),
                    "reasons": [],
                    "axial_series": [s.id for s in st.axial_for_level(lvl)],
                }
                regions.append(entry)
            entry["reasons"].append({
                "kind": f["name"], "detail": "",
                "score": round(f["probability"], 3),
                "source": "findings head",
            })

    order = {l: i for i, l in enumerate(["L3-L4", "L4-L5", "L5-S1"])}
    regions.sort(key=lambda r: order.get(r["level"], 99))
    return regions
