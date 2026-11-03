"""
Locating a named finding.

Neither model outputs a bounding box. The grading head scores eight pathologies
plus herniation *per disc*, and the findings head scores 36 labels for the study
as a whole. Localisation therefore works by joining two things that are both
real:

  the level    a per-disc score above threshold, or a level named in the
               finding itself ("L4-L5 Involvement")
  the region   the segmented structure that the finding describes -- the disc
               for a bulge, the cord for canal stenosis, the posterior elements
               for facet hypertrophy

The outline drawn is the segmentation of that structure. The number quoted is
the model's score. A finding whose structure is not among the segmented
classes is reported without an outline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import engine
from . import study as study_mod

# Per-disc heads on the grading model -> display name.
GRADING_PATHOLOGY = {
    "disc_bulge": "Disc Bulge",
    "disc_protrusion": "Disc Protrusion",
    "disc_extrusion": "Disc Extrusion",
    "herniation": "Disc Herniation",
    "nerve_compression": "Nerve Root Compression",
    "thecal_compression": "Thecal Sac Compression",
    "foraminal_narrowing": "Neural Foraminal Narrowing",
    "facet_hypertrophy": "Facet Joint Hypertrophy",
    "ligamentum_flavum": "Ligamentum Flavum Hypertrophy",
}
DISPLAY_TO_GRADING = {v: k for k, v in GRADING_PATHOLOGY.items()}

# Which segmented structure each finding describes. "disc" and "vertebra" are
# resolved to the component at the level in question; a class name is drawn as
# the whole class within that level's band.
#   sag  -> structures to mark on a sagittal series
#   ax   -> structures to mark on an axial series
#   band -> restrict a class mark to the vertical band of the level (sagittal)
ANATOMY: Dict[str, Dict[str, Any]] = {
    "Disc Bulge":                    {"sag": ["disc"], "ax": ["disc"]},
    "Disc Protrusion":               {"sag": ["disc"], "ax": ["disc"]},
    "Disc Extrusion":                {"sag": ["disc"], "ax": ["disc"]},
    "Disc Herniation":               {"sag": ["disc"], "ax": ["disc"]},
    "Desiccated Disc":               {"sag": ["disc"], "ax": ["disc"]},
    "Annular Tear":                  {"sag": ["disc"], "ax": ["disc"]},
    "Thecal Sac Compression":        {"sag": ["SC"], "ax": ["SC"], "band": True},
    "Spinal Canal Stenosis":         {"sag": ["SC"], "ax": ["SC"], "band": True},
    "Adequate Spinal Canal":         {"sag": ["SC"], "ax": ["SC"], "band": True},
    "No Significant Compression":    {"sag": ["SC"], "ax": ["SC"], "band": True},
    "Nerve Root Compression":        {"sag": ["disc", "SC"], "ax": ["SC", "PE"], "band": True},
    "Neural Foraminal Narrowing":    {"sag": ["disc"], "ax": ["PE"]},
    "Ligamentum Flavum Hypertrophy": {"sag": [], "ax": ["PE"]},
    "Facet Joint Hypertrophy":       {"sag": [], "ax": ["PE"]},
    "Endplate Degeneration":         {"sag": ["adjacent_vertebrae"], "ax": []},
    "Modic Changes":                 {"sag": ["adjacent_vertebrae"], "ax": []},
    "Schmorl Nodules":               {"sag": ["adjacent_vertebrae"], "ax": []},
    "Compression Fracture":          {"sag": ["adjacent_vertebrae"], "ax": []},
    "Hemangioma":                    {"sag": ["adjacent_vertebrae"], "ax": []},
    "Spondylolisthesis":             {"sag": ["adjacent_vertebrae"], "ax": []},
    "L4-L5 Compression":             {"sag": ["disc", "SC"], "ax": ["SC"], "band": True},
    "L5-S1 Compression":             {"sag": ["disc", "SC"], "ax": ["SC"], "band": True},
}

# Findings the models score but whose structure is not in any class list.
NO_STRUCTURE = {
    "Muscle Spasm": "paraspinal musculature is outside the segmented classes",
    "Previous Surgery": "surgical change has no dedicated class",
    "Tarlov Cyst": "sacral nerve-root sleeves are outside the segmented classes",
    "Lipoma": "fat deposits have no dedicated class",
    "Left-Sided Pathology": "a laterality statement, not a structure",
    "Right-Sided Pathology": "a laterality statement, not a structure",
    "Central Pathology": "a position statement, not a structure",
}

LEVEL_TAG_RE = re.compile(r"^((?:[LTD]\d{1,2})-(?:[LTS]\d{1,2}))\s+(Involvement|Compression)$", re.I)

# Every name a user might type -> canonical finding name.
_ALIASES: Dict[str, str] = {}


def _register(canonical: str, *spellings: str) -> None:
    _ALIASES[canonical.lower()] = canonical
    for s in spellings:
        _ALIASES[s.lower()] = canonical


for _n in engine.BINARY_FINDING_NAMES:
    _register(_n)
for _n in GRADING_PATHOLOGY.values():
    _register(_n)

_register("Spinal Canal Stenosis", "stenosis", "canal stenosis", "spinal stenosis",
          "central canal stenosis")
_register("Disc Herniation", "herniation", "herniated disc", "hernia")
_register("Disc Bulge", "bulge", "bulging disc", "disc bulging")
_register("Disc Protrusion", "protrusion", "protruded disc")
_register("Disc Extrusion", "extrusion", "extruded disc")
_register("Thecal Sac Compression", "thecal compression", "thecal sac indentation",
          "thecal indentation")
_register("Nerve Root Compression", "nerve compression", "nerve root impingement",
          "radiculopathy", "root compression")
_register("Neural Foraminal Narrowing", "foraminal narrowing", "foraminal stenosis",
          "neural foraminal stenosis")
_register("Ligamentum Flavum Hypertrophy", "ligamentum flavum", "flavum hypertrophy")
_register("Facet Joint Hypertrophy", "facet hypertrophy", "facet arthropathy",
          "facet joint arthropathy", "facet degeneration")
_register("Desiccated Disc", "desiccation", "disc desiccation", "dehydrated disc")
_register("Annular Tear", "annular fissure", "hiz")
_register("Endplate Degeneration", "endplate change", "endplate changes")
_register("Modic Changes", "modic", "modic change")
_register("Schmorl Nodules", "schmorl", "schmorl node", "schmorl nodes")
_register("Compression Fracture", "vertebral fracture", "fracture")
_register("Spondylolisthesis", "listhesis", "slip", "anterolisthesis")
_register("Muscle Spasm", "spasm")
_register("Hemangioma", "haemangioma")
_register("Tarlov Cyst", "tarlov", "perineural cyst")


def resolve_name(text: str) -> Optional[str]:
    """Map free text onto a canonical finding name, longest phrase first."""
    t = (text or "").strip().lower()
    if t in _ALIASES:
        return _ALIASES[t]
    for phrase in sorted(_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            return _ALIASES[phrase]
    return None


def catalogue() -> List[Dict[str, str]]:
    """Every finding that can be located, for the interface."""
    out = []
    for n in engine.BINARY_FINDING_NAMES:
        if LEVEL_TAG_RE.match(n):
            kind = "level-tagged"
        elif n in NO_STRUCTURE:
            kind = "not segmented"
        elif n in ANATOMY:
            kind = "localisable"
        else:
            kind = "study-level"
        out.append({"name": n, "kind": kind})
    return out


# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    level: Optional[str]
    score: Optional[float]
    source: str
    detail: str = ""


@dataclass
class LocateResult:
    finding: str
    found: bool
    levels: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    study_score: Optional[float] = None
    study_positive: Optional[bool] = None
    marks: List[Dict[str, Any]] = field(default_factory=list)   # per series
    note: str = ""
    reason: str = ""


def _level_band(sr, level: str) -> Optional[Tuple[int, int]]:
    """Vertical extent of a level on a sagittal series, used to clip a class
    mark so the cord is highlighted at that level rather than end to end."""
    d = next((x for x in sr.discs if x.get("label") == level), None)
    if d is None:
        return None
    _, y, _, h = d["bbox"]
    pad = max(8, h)
    return max(0, y - pad), min(sr.mask.shape[0], y + h + pad)


def _adjacent_vertebrae(sr, level: str) -> List[Dict[str, Any]]:
    """The two bodies forming a disc level, e.g. L4 and L5 for L4-L5."""
    parts = level.split("-")
    if len(parts) != 2:
        return []
    return [v for v in sr.vertebrae if v.get("label") in parts]


def _marks_for_series(sr, level: Optional[str], structures: List[str],
                      band: bool) -> List[Dict[str, Any]]:
    """Turn a structure list into concrete regions on one series."""
    marks: List[Dict[str, Any]] = []
    name_to_id = {n: cid for cid, n in sr.class_names.items()}

    for s in structures:
        if s == "disc":
            if sr.plane == "sagittal":
                d = next((x for x in sr.discs if x.get("label") == level), None)
                cands = [d] if d else []
            else:
                # An axial slice cuts one level; its disc region is that level.
                cands = list(sr.discs)
            for d in cands:
                if d is None:
                    continue
                marks.append({"kind": "component", "structure": "IVD",
                              "class_id": name_to_id.get("IVD"),
                              "component_id": d.get("component_id"),
                              "bbox": list(d["bbox"]),
                              "label": d.get("label") or level or "disc"})

        elif s == "adjacent_vertebrae":
            if sr.plane != "sagittal" or not level:
                continue
            for v in _adjacent_vertebrae(sr, level):
                marks.append({"kind": "component", "structure": "VB",
                              "class_id": name_to_id.get("VB"),
                              "component_id": v.get("component_id"),
                              "bbox": list(v["bbox"]),
                              "label": v.get("label", "")})

        else:
            cid = name_to_id.get(s)
            if cid is None or not (sr.mask == cid).any():
                continue
            band_y = _level_band(sr, level) if (band and level and sr.plane == "sagittal") else None
            marks.append({"kind": "class", "structure": s, "class_id": cid,
                          "band": list(band_y) if band_y else None,
                          "label": engine.CLASS_FULL_NAMES.get(s, s)})
    return marks


def locate(st, finding_text: str, threshold: float = 0.5) -> LocateResult:
    """Find where a named finding sits, using only scores the models produced."""
    name = resolve_name(finding_text)
    if name is None:
        return LocateResult(finding=finding_text, found=False,
                            reason="not a finding this model reports")

    res = LocateResult(finding=name, found=False)
    ref = st.primary_sagittal() or (st.series[0] if st.series else None)
    if ref is None:
        res.reason = "no image in this study"
        return res

    # --- study-level score from the findings head -------------------------
    if ref.findings and "binary_findings" in ref.findings:
        hit = next((f for f in ref.findings["binary_findings"]
                    if f["name"].lower() == name.lower()), None)
        if hit:
            res.study_score = round(hit["probability"], 3)
            res.study_positive = hit["probability"] >= threshold

    # --- per-level evidence ----------------------------------------------
    levels: List[str] = []

    tag = LEVEL_TAG_RE.match(name)
    if tag:
        # The finding names its own level.
        lvl = tag.group(1).upper()
        lvl = study_mod.DCODE_TO_LEVEL.get(lvl.split("-")[0], lvl) if lvl.startswith("D") else lvl
        if res.study_score is not None and res.study_positive:
            levels = [lvl]
            res.evidence.append(Evidence(lvl, res.study_score, "findings head",
                                         "level named by the finding"))
    else:
        key = DISPLAY_TO_GRADING.get(name)
        if key:
            for lvl, g in ref.grading.items():
                if "error" in g:
                    continue
                p = (g.get("pathology_probs") or {}).get(key)
                if p is not None and p >= threshold:
                    levels.append(lvl)
                    res.evidence.append(Evidence(lvl, round(p, 3), "grading head",
                                                 "per-disc score"))
        if not levels and name == "Spinal Canal Stenosis":
            # The stenosis head is separate from the pathology heads.
            for lvl, g in ref.grading.items():
                if "error" not in g and g.get("stenosis", 0) >= 1:
                    sc = g["stenosis_probs"][g["stenosis"]]
                    levels.append(lvl)
                    res.evidence.append(Evidence(lvl, round(sc, 3), "grading head",
                                                 f"stenosis {g['stenosis_label']}"))
        if not levels and res.study_positive:
            # Only a study-level score exists. Mark the graded levels and say so.
            levels = [l for l in ("L3-L4", "L4-L5", "L5-S1")
                      if l in ref.grading and "error" not in ref.grading[l]]
            for l in levels:
                res.evidence.append(Evidence(l, res.study_score, "findings head",
                                             "study-level score, not per level"))

    order = {l: i for i, l in enumerate(["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"])}
    levels = sorted(set(levels), key=lambda l: order.get(l, 99))
    res.levels = levels

    # --- no segmented structure -------------------------------------------
    if name in NO_STRUCTURE:
        res.found = bool(levels) or bool(res.study_positive)
        res.note = (f"This model scores {name} but does not segment the structure "
                    f"it describes: {NO_STRUCTURE[name]}. No region is outlined, "
                    f"because placing one would not come from the model.")
        return res

    anat = ANATOMY.get(name)
    if anat is None and tag:
        # "L4-L5 Involvement" and its siblings name their own level, so the disc
        # at that level is the region they describe. A Compression variant also
        # implicates the canal.
        if tag.group(2).lower() == "compression":
            anat = {"sag": ["disc", "SC"], "ax": ["SC"], "band": True}
        else:
            anat = {"sag": ["disc"], "ax": ["disc"]}
    if anat is None:
        res.found = bool(res.study_positive)
        res.note = (f"{name} carries a study-level score only. This model does not "
                    f"tie it to a level or a structure, so nothing is outlined.")
        return res

    if not levels:
        res.reason = (f"{name} did not reach {threshold:.2f} at any level "
                      f"the grading head covers.")
        return res

    # --- build the marks ---------------------------------------------------
    band = bool(anat.get("band"))
    for lvl in levels:
        sag_marks = _marks_for_series(ref, lvl, anat.get("sag", []), band)
        if sag_marks:
            res.marks.append({"series_id": ref.id, "series_label": ref.label(),
                              "plane": ref.plane, "level": lvl, "regions": sag_marks})
        for ax in st.axial_for_level(lvl):
            ax_marks = _marks_for_series(ax, lvl, anat.get("ax", []), False)
            if ax_marks:
                res.marks.append({"series_id": ax.id, "series_label": ax.label(),
                                  "plane": ax.plane, "level": lvl, "regions": ax_marks})

    res.found = bool(res.marks)
    if not res.marks:
        which = "sagittal" if ref.plane == "sagittal" else "axial"
        res.reason = (f"{name} sits in a structure this study does not show on the "
                      f"{which} series. Adding an axial slice at "
                      f"{', '.join(levels)} would show it.")
    else:
        res.note = ("The outline is the segmented structure the finding describes; "
                    "the number is the model's score for it. Neither model outputs "
                    "a bounding box, so the outline locates the anatomy, not a "
                    "model-predicted lesion boundary.")
    return res


def to_dict(r: LocateResult) -> Dict[str, Any]:
    return {
        "finding": r.finding, "found": r.found, "levels": r.levels,
        "study_score": r.study_score, "study_positive": r.study_positive,
        "evidence": [{"level": e.level, "score": e.score, "source": e.source,
                      "detail": e.detail} for e in r.evidence],
        "note": r.note, "reason": r.reason,
        "marks": [{"series_id": m["series_id"], "series_label": m["series_label"],
                   "level": m["level"], "regions": len(m["regions"])} for m in r.marks],
    }
