"""
Deterministic command parser.

Intent and target come from regular expressions over a closed vocabulary.
Every number in an answer is read back from a model output or from image
metadata. A question outside spine imaging returns usage guidance.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import engine
from . import findings as fx
from . import render
from . import study as study_mod

LEXICON_PATH = Path(os.environ.get("SPINALYZER_LEXICON",
                                   str(engine.REPO_ROOT / "lexicon.json")))

# Intent patterns, most specific first.
INTENTS: List[Tuple[str, str]] = [
    (r"\b(report|impression|summary\s+report)\b", "REPORT"),
    (r"\b(compare|versus|vs\.?)\b", "COMPARE"),
    # LOCATE sits above FIND and SEGMENT: "locate", "where is" and "point out"
    # ask for a position, which FIND alone would answer with a yes/no.
    (r"\b(locate|where(?:'s|\s+is|\s+are)?|point\s+out|pinpoint|mark)\b", "LOCATE"),
    (r"\b(pfirrmann|grade|grading|degenerat\w*)\b", "GRADE"),
    (r"\b(measure|measurement|height|width|diameter|dimension|size|area)\b", "MEASURE"),
    (r"\b(find|detect|identify|any\s+sign)\b", "FIND"),
    (r"\b(segment|isolate|highlight|outline|mask|show|display)\b", "SEGMENT"),
    (r"\b(describe|list|inventory|overview|what\s+do\s+you\s+see)\b", "DESCRIBE"),
    (r"\b(help|commands?|examples?)\b", "HELP"),
]

# L4-L5, L4/L5, L4 L5, T12-L1, L5-S1
DISC_RE = re.compile(r"\b([LTSD])\s*(\d{1,2})\s*[-/\s]\s*([LTSD])\s*(\d{1,2})\b", re.I)
# L4, T12, S1 -- a single body
VERTEBRA_RE = re.compile(r"\b([LTS])\s*(\d{1,2})\b", re.I)
DCODE_RE = re.compile(r"\b(D[345])\b", re.I)

STRUCTURES: Dict[str, str] = {
    "spinal cord": "SC", "cord": "SC", "sc": "SC",
    "sacrum": "Sacrum",
    "posterior elements": "PE", "posterior element": "PE", "facet": "PE",
    "facets": "PE", "pe": "PE",
    "intervertebral disc": "IVD", "intervertebral discs": "IVD",
    "discs": "IVD", "ivd": "IVD",
    "vertebral body": "VB", "vertebral bodies": "VB",
    "vertebrae": "VB", "vb": "VB",
}

PATHOLOGY: Dict[str, str] = {
    "spinal canal stenosis": "Spinal Canal Stenosis",
    "canal stenosis": "Spinal Canal Stenosis",
    "spinal stenosis": "Spinal Canal Stenosis",
    "stenosis": "Spinal Canal Stenosis",
    "disc herniation": "Disc Herniation", "herniation": "Disc Herniation",
    "disc bulge": "Disc Bulge", "bulge": "Disc Bulge",
    "disc protrusion": "Disc Protrusion", "protrusion": "Disc Protrusion",
    "disc extrusion": "Disc Extrusion", "extrusion": "Disc Extrusion",
    "thecal sac compression": "Thecal Sac Compression",
    "nerve root compression": "Nerve Root Compression",
    "neural foraminal narrowing": "Neural Foraminal Narrowing",
    "foraminal narrowing": "Neural Foraminal Narrowing",
    "annular tear": "Annular Tear",
    "desiccated disc": "Desiccated Disc", "desiccation": "Desiccated Disc",
    "ligamentum flavum hypertrophy": "Ligamentum Flavum Hypertrophy",
    "ligamentum flavum": "Ligamentum Flavum Hypertrophy",
    "facet joint hypertrophy": "Facet Joint Hypertrophy",
    "facet hypertrophy": "Facet Joint Hypertrophy",
    "schmorl nodules": "Schmorl Nodules", "schmorl": "Schmorl Nodules",
    "endplate degeneration": "Endplate Degeneration",
    "modic changes": "Modic Changes", "modic": "Modic Changes",
    "compression fracture": "Compression Fracture",
    "spondylolisthesis": "Spondylolisthesis",
    "hemangioma": "Hemangioma",
    "tarlov cyst": "Tarlov Cyst", "tarlov": "Tarlov Cyst",
    "lipoma": "Lipoma",
    "muscle spasm": "Muscle Spasm",
}

# Intents that are a complete request on their own. The rest need a target
# before the question counts as in scope.
STANDALONE_INTENTS = {"DESCRIBE", "HELP", "REPORT"}

SPINE_WORDS = (
    "spine", "spinal", "lumbar", "disc", "disk", "vertebra", "vertebral",
    "sacrum", "sacral", "canal", "cord", "thecal", "pfirrmann", "stenosis",
    "herniat", "bulge", "protrusion", "extrusion", "foramen", "foraminal",
    "endplate", "modic", "facet", "annular", "level", "sagittal", "axial",
    "segment", "measure", "grade", "report", "mri", "scan", "image", "slice",
)


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

_LEXICON: List[Dict[str, str]] = []


def load_lexicon() -> None:
    global _LEXICON
    if LEXICON_PATH.exists():
        try:
            _LEXICON = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
        except Exception:
            _LEXICON = []
    else:
        _LEXICON = []


def save_lexicon() -> None:
    LEXICON_PATH.write_text(json.dumps(_LEXICON, indent=2), encoding="utf-8")


def list_aliases() -> List[Dict[str, str]]:
    return list(_LEXICON)


def add_alias(pattern: str, expansion: str) -> Dict[str, str]:
    pattern = pattern.strip()
    expansion = expansion.strip()
    for a in _LEXICON:
        if a["pattern"].lower() == pattern.lower():
            a["expansion"] = expansion
            save_lexicon()
            return a
    entry = {"pattern": pattern, "expansion": expansion}
    _LEXICON.append(entry)
    save_lexicon()
    return entry


def remove_alias(pattern: str) -> bool:
    global _LEXICON
    before = len(_LEXICON)
    _LEXICON = [a for a in _LEXICON if a["pattern"].lower() != pattern.strip().lower()]
    if len(_LEXICON) != before:
        save_lexicon()
        return True
    return False


def clear_aliases() -> int:
    global _LEXICON
    n = len(_LEXICON)
    _LEXICON = []
    save_lexicon()
    return n


def apply_lexicon(text: str) -> str:
    out = text
    for a in sorted(_LEXICON, key=lambda x: len(x["pattern"]), reverse=True):
        out = re.sub(rf"\b{re.escape(a['pattern'])}\b", a["expansion"], out, flags=re.I)
    return out


def lexicon_command(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    low = t.lower()
    if low in ("list", "show", "aliases"):
        return {"ok": True, "message": f"{len(_LEXICON)} term(s) defined.",
                "aliases": list_aliases()}
    if low == "clear":
        return {"ok": True, "message": f"Removed {clear_aliases()} term(s).",
                "aliases": []}
    m = re.match(r"^(?:remove|delete|forget)\s+(.+)$", t, re.I)
    if m:
        ok = remove_alias(m.group(1))
        return {"ok": ok,
                "message": f"Removed '{m.group(1)}'." if ok else f"'{m.group(1)}' is not defined.",
                "aliases": list_aliases()}
    m = re.match(r"^(.+?)\s*(?:=|->|=>|:|→)\s*(.+)$", t)
    if m:
        entry = add_alias(m.group(1), m.group(2))
        return {"ok": True,
                "message": f"'{entry['pattern']}' now expands to '{entry['expansion']}'.",
                "aliases": list_aliases()}
    return {"ok": False,
            "message": "Use  term = expansion  to define,  remove term  to delete, "
                       "or  list  to see everything.",
            "aliases": list_aliases()}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass
class Command:
    intent: str = "DESCRIBE"
    target: Optional[str] = None
    target_kind: Optional[str] = None      # disc | vertebra | structure | pathology
    second_target: Optional[str] = None
    pathology: Optional[str] = None
    raw: str = ""
    expanded: str = ""
    in_scope: bool = True


def _disc_label(a: str, na: str, b: str, nb: str) -> Optional[str]:
    a, b = a.upper(), b.upper()
    if a == "D" or b == "D":
        return None
    if a not in "LTS" or b not in "LTS":
        return None
    return f"{a}{int(na)}-{b}{int(nb)}"


def parse(text: str) -> Command:
    raw = text or ""
    expanded = apply_lexicon(raw)
    t = expanded.strip().lower()
    cmd = Command(raw=raw, expanded=expanded)

    matched = False
    for pattern, intent in INTENTS:
        if re.search(pattern, t, re.I):
            cmd.intent = intent
            matched = True
            break

    # A disc is tried before a vertebra, because "L4-L5" also contains "L4".
    m = DISC_RE.search(t)
    if m:
        lbl = _disc_label(*m.groups())
        if lbl:
            cmd.target, cmd.target_kind = lbl, "disc"

    if cmd.target is None:
        m = DCODE_RE.search(t)
        if m:
            lbl = study_mod.DCODE_TO_LEVEL.get(m.group(1).upper())
            if lbl:
                cmd.target, cmd.target_kind = lbl, "disc"

    if cmd.target is None:
        m = VERTEBRA_RE.search(t)
        if m:
            letter, num = m.group(1).upper(), int(m.group(2))
            ok = ((letter == "L" and 1 <= num <= 5)
                  or (letter == "S" and num == 1)
                  or (letter == "T" and 1 <= num <= 12))
            if ok:
                cmd.target, cmd.target_kind = f"{letter}{num}", "vertebra"

    if cmd.target is None:
        for phrase in sorted(STRUCTURES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(phrase)}\b", t):
                cmd.target, cmd.target_kind = STRUCTURES[phrase], "structure"
                break

    for phrase in sorted(PATHOLOGY, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            cmd.pathology = PATHOLOGY[phrase]
            if cmd.target_kind is None:
                cmd.target, cmd.target_kind = cmd.pathology, "pathology"
            break

    # "show me the spinal cord" reads as FIND but means SEGMENT.
    if cmd.intent == "FIND" and cmd.pathology is None and \
            cmd.target_kind in ("disc", "vertebra", "structure"):
        cmd.intent = "SEGMENT"

    # "show spondylolisthesis" matches SEGMENT on the verb, but the target is a
    # pathology, so the question is where it sits, not which class to tint.
    if cmd.intent == "SEGMENT" and cmd.pathology is not None and \
            cmd.target_kind == "pathology":
        cmd.intent = "LOCATE"

    # "locate L4-L5" names anatomy, not a finding, so tint the component.
    if cmd.intent == "LOCATE" and cmd.pathology is None and \
            cmd.target_kind in ("disc", "vertebra", "structure"):
        cmd.intent = "SEGMENT"

    if cmd.intent == "COMPARE":
        discs = [d for d in (_disc_label(*mm.groups()) for mm in DISC_RE.finditer(t)) if d]
        if len(discs) >= 2:
            cmd.target, cmd.second_target, cmd.target_kind = discs[0], discs[1], "disc"
        else:
            verts = [f"{mm.group(1).upper()}{int(mm.group(2))}"
                     for mm in VERTEBRA_RE.finditer(t)]
            used = {p for d in discs for p in d.split("-")}
            verts = [v for v in verts if v not in used]
            if len(verts) >= 2:
                cmd.target, cmd.second_target, cmd.target_kind = verts[0], verts[1], "vertebra"

    # A verb that needs an object cannot put a question in scope by itself:
    # "where is my car" matches LOCATE but names nothing this tool reads. Only
    # the verbs that stand alone, a resolved target, or a spine term qualify.
    cmd.in_scope = bool((matched and cmd.intent in STANDALONE_INTENTS)
                        or cmd.target or cmd.pathology
                        or any(w in t for w in SPINE_WORDS))
    if not cmd.in_scope:
        cmd.intent = "UNKNOWN"
    elif not matched:
        if cmd.target_kind == "pathology":
            cmd.intent = "FIND"
        elif cmd.target:
            cmd.intent = "SEGMENT"
        else:
            cmd.intent = "DESCRIBE"
    return cmd


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def _find_disc(sr, label: str) -> Optional[Dict[str, Any]]:
    return next((d for d in sr.discs if d.get("label") == label), None)


def _find_vertebra(sr, label: str) -> Optional[Dict[str, Any]]:
    return next((v for v in sr.vertebrae if v.get("label") == label), None)


def _ref(st) -> Optional[Any]:
    return st.primary_sagittal() or (st.series[0] if st.series else None)


def _px_mm(px: int, mm: Optional[float], squared: bool = False) -> str:
    u = "px²" if squared else "px"
    if mm is None:
        return f"{px} {u}"
    return f"{px} {u} ({mm} mm{'²' if squared else ''})"


def _scale_note(sr) -> str:
    if sr.spacing:
        return f"Scale from {sr.spacing.get('source', 'calibration')}."
    return ("No pixel-size metadata in this file, so sizes stay in pixels. "
            "Set a scale under Calibration to convert to millimetres.")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def act_segment(st, cmd: Command) -> Dict[str, Any]:
    sr = _ref(st)
    if sr is None:
        return {"text": "No image loaded.", "image": None}

    if cmd.target_kind == "disc":
        d = _find_disc(sr, cmd.target)
        if not d:
            have = ", ".join(x["label"] for x in sr.discs) or "none"
            return {"text": f"{cmd.target} was not segmented in this study. "
                            f"Levels present: {have}.", "image": None}
        m = study_mod.disc_metrics(d, sr)
        return {
            "text": f"**{cmd.target} intervertebral disc**\n\n"
                    f"- Height: {_px_mm(m['height_px'], m['height_mm'])}\n"
                    f"- AP diameter: {_px_mm(m['ap_diameter_px'], m['ap_diameter_mm'])}\n"
                    f"- Area: {_px_mm(m['area_px'], m['area_mm2'], True)}\n\n"
                    f"{_scale_note(sr)}",
            "image": render.render_target(sr, d, "disc", m, f"{cmd.target} disc"),
            "data": m,
        }

    if cmd.target_kind == "vertebra":
        v = _find_vertebra(sr, cmd.target)
        if not v:
            have = ", ".join(x["label"] for x in sr.vertebrae) or "none"
            return {"text": f"{cmd.target} vertebral body was not segmented. "
                            f"Bodies present: {have}.", "image": None}
        m = study_mod.vertebra_metrics(v, sr)
        return {
            "text": f"**{cmd.target} vertebral body**\n\n"
                    f"- Height: {_px_mm(m['height_px'], m['height_mm'])}\n"
                    f"- Width: {_px_mm(m['width_px'], m['width_mm'])}\n"
                    f"- Area: {_px_mm(m['area_px'], m['area_mm2'], True)}\n\n"
                    f"{_scale_note(sr)}",
            "image": render.render_target(sr, v, "vertebra", m, f"{cmd.target} vertebral body"),
            "data": m,
        }

    if cmd.target_kind == "structure":
        cid = next((c for c, n in sr.class_names.items() if n == cmd.target), None)
        if cid is None:
            avail = ", ".join(n for c, n in sr.class_names.items() if c != 0)
            return {"text": f"'{cmd.target}' is not a class in this segmentation "
                            f"({avail}).", "image": None}
        n_px = int((sr.mask == cid).sum())
        pct = 100.0 * n_px / sr.mask.size
        full = engine.CLASS_FULL_NAMES.get(cmd.target, cmd.target)
        return {
            "text": f"**{full}** — {n_px:,} px, {pct:.2f}% of the segmented field.",
            "image": render.render_class(sr, cid, full),
        }

    return {"text": "Name a level or structure, for example "
                    "*segment L4*, *segment L4-L5*, or *show the spinal cord*.",
            "image": None}


def act_measure(st, cmd: Command) -> Dict[str, Any]:
    sr = _ref(st)
    if sr is None:
        return {"text": "No image loaded.", "image": None}

    if cmd.target_kind == "disc":
        d = _find_disc(sr, cmd.target)
        if not d:
            return {"text": f"{cmd.target} was not segmented in this study.", "image": None}
        m = study_mod.disc_metrics(d, sr)
        return {
            "text": f"**{cmd.target}**\n\n"
                    f"| Measurement | Value |\n|---|---|\n"
                    f"| Height | {_px_mm(m['height_px'], m['height_mm'])} |\n"
                    f"| AP diameter | {_px_mm(m['ap_diameter_px'], m['ap_diameter_mm'])} |\n"
                    f"| Area | {_px_mm(m['area_px'], m['area_mm2'], True)} |\n\n{_scale_note(sr)}",
            "image": render.render_target(sr, d, "disc", m, f"{cmd.target} measurements"),
            "data": m,
        }

    if cmd.target_kind == "vertebra":
        v = _find_vertebra(sr, cmd.target)
        if not v:
            return {"text": f"{cmd.target} was not segmented in this study.", "image": None}
        m = study_mod.vertebra_metrics(v, sr)
        return {
            "text": f"**{cmd.target}**\n\n"
                    f"| Measurement | Value |\n|---|---|\n"
                    f"| Height | {_px_mm(m['height_px'], m['height_mm'])} |\n"
                    f"| Width | {_px_mm(m['width_px'], m['width_mm'])} |\n"
                    f"| Area | {_px_mm(m['area_px'], m['area_mm2'], True)} |\n\n{_scale_note(sr)}",
            "image": render.render_target(sr, v, "vertebra", m, f"{cmd.target} measurements"),
            "data": m,
        }

    rows = ["| Level | Height | AP diameter | Area |", "|---|---|---|---|"]
    for d in sr.discs:
        m = study_mod.disc_metrics(d, sr)
        rows.append(f"| {m['label']} | {_px_mm(m['height_px'], m['height_mm'])} | "
                    f"{_px_mm(m['ap_diameter_px'], m['ap_diameter_mm'])} | "
                    f"{_px_mm(m['area_px'], m['area_mm2'], True)} |")
    return {"text": "**All segmented disc levels**\n\n" + "\n".join(rows) +
                    f"\n\n{_scale_note(sr)}",
            "image": None}


def act_grade(st, cmd: Command) -> Dict[str, Any]:
    sr = _ref(st)
    if sr is None:
        return {"text": "No image loaded.", "image": None}
    if not sr.grading:
        return {"text": "No level in this study was graded. Grading needs a sagittal "
                        "slice containing L3-L4, L4-L5 or L5-S1.", "image": None}

    if cmd.target_kind == "disc":
        g = sr.grading.get(cmd.target)
        if g is None:
            trained = ", ".join(engine.IN_DIST_DISCS)
            return {"text": f"{cmd.target} carries no grade. The grading model was "
                            f"trained on {trained} only, so no value is reported for "
                            f"other levels.", "image": None}
        if "error" in g:
            return {"text": f"{cmd.target}: {g['error']}", "image": None}
        d = _find_disc(sr, cmd.target)
        cum = ", ".join(f"{v:.2f}" for v in g["pfirrmann_cumulative"])
        return {
            "text": f"**{cmd.target}**\n\n"
                    f"- Pfirrmann grade **{g['pfirrmann_roman']}** — {g['pfirrmann_desc']}\n"
                    f"- Stenosis: **{g['stenosis_label']}** "
                    f"(p = {g['stenosis_probs'][g['stenosis']]:.2f})\n"
                    f"- Ordinal boundary probabilities: {cum}",
            "image": (render.render_target(sr, d, "disc", study_mod.disc_metrics(d, sr),
                                           f"{cmd.target} — Pfirrmann {g['pfirrmann_roman']}")
                      if d else None),
            "data": g,
        }

    rows = ["| Level | Pfirrmann | Description | Stenosis |", "|---|---|---|---|"]
    for lvl in ("L3-L4", "L4-L5", "L5-S1"):
        g = sr.grading.get(lvl)
        if not g or "error" in g:
            continue
        rows.append(f"| {lvl} | {g['pfirrmann_roman']} | {g['pfirrmann_desc']} | "
                    f"{g['stenosis_label']} |")
    return {"text": "**Pfirrmann grading**\n\n" + "\n".join(rows), "image": None}


def act_find(st, cmd: Command) -> Dict[str, Any]:
    sr = _ref(st)
    if sr is None:
        return {"text": "No image loaded.", "image": None}
    if not sr.findings or "binary_findings" not in sr.findings:
        reason = (sr.findings or {}).get("error", "structured findings were not produced")
        return {"text": f"No findings available: {reason}.", "image": None}

    if cmd.pathology:
        hit = next((f for f in sr.findings["binary_findings"]
                    if f["name"].lower() == cmd.pathology.lower()), None)
        if hit is None:
            return {"text": f"'{cmd.pathology}' is not in this model's finding list.",
                    "image": None}
        verdict = "present" if hit["positive"] else "not present"
        lines = [f"**{hit['name']}** — {verdict} "
                 f"(probability {hit['probability']:.2f}, threshold 0.50)."]
        # Screening answers the yes/no; the locator supplies where it sits.
        loc = locate_finding(st, hit["name"])
        images = loc.get("images") or []
        if images:
            lines.append(f"\nMarked at {', '.join(loc['locate']['levels'])}.")
        elif loc["locate"].get("note"):
            lines.append(f"\n{loc['locate']['note']}")
        return {"text": "\n".join(lines),
                "image": loc.get("image"), "images": images,
                "locate": loc.get("locate"), "data": hit}

    positives = [f for f in sr.findings["binary_findings"]
                 if f["positive"] and f["pathological"]]
    if not positives:
        return {"text": "No pathological finding reached the 0.50 threshold.",
                "image": None}
    rows = ["| Finding | Probability |", "|---|---|"]
    for f in sorted(positives, key=lambda x: -x["probability"]):
        rows.append(f"| {f['name']} | {f['probability']:.2f} |")
    regions = study_mod.abnormal_regions(st)
    return {"text": "**Positive findings**\n\n" + "\n".join(rows),
            "image": render.render_abnormal_panel(sr, regions) if regions else None}


def locate_finding(st, name: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Shared by the LOCATE intent and the interface's per-row Show button."""
    r = fx.locate(st, name, threshold)

    if not r.found and not r.levels:
        why = (r.reason or "no evidence above the threshold").rstrip(".")
        return {"text": f"**{r.finding}** — {why}.", "image": None, "images": [],
                "locate": fx.to_dict(r)}

    lines = [f"**{r.finding}**"]
    if r.study_score is not None:
        verdict = "positive" if r.study_positive else "negative"
        lines.append(f"\nStudy-level score {r.study_score:.3f} ({verdict}, "
                     f"threshold {threshold:.2f}) from the findings head.")
    if r.evidence:
        lines.append("\n| Level | Score | Source | Basis |\n|---|---|---|---|")
        for e in r.evidence:
            sc = f"{e.score:.3f}" if e.score is not None else "-"
            lines.append(f"| {e.level or '-'} | {sc} | {e.source} | {e.detail or '-'} |")

    images: List[Dict[str, Any]] = []
    for m in r.marks:
        sr = st.by_id(m["series_id"])
        if sr is None:
            continue
        sc = next((e.score for e in r.evidence if e.level == m["level"]), None)
        images.append({
            "series_id": sr.id,
            "label": f"{r.finding} · {m['level']} · {sr.label()}",
            "level": m["level"],
            "image": render.render_finding(sr, m, r.finding, sc),
        })

    if images:
        lines.append(f"\nMarked on {len(images)} series at "
                     f"{', '.join(r.levels)}.")
    if r.note:
        lines.append(f"\n_{r.note}_")
    if r.reason:
        lines.append(f"\n{r.reason}")

    return {"text": "\n".join(lines),
            "image": images[0]["image"] if images else None,
            "images": images,
            "locate": fx.to_dict(r)}


def act_locate(st, cmd: Command) -> Dict[str, Any]:
    target = cmd.pathology or cmd.target
    if not target:
        return {"text": "Name a finding to locate, for example "
                        "*locate stenosis*, *where is the herniation*, or "
                        "*show spondylolisthesis*.", "image": None}
    return locate_finding(st, target)


def act_compare(st, cmd: Command) -> Dict[str, Any]:
    sr = _ref(st)
    if sr is None or not cmd.target or not cmd.second_target:
        return {"text": "Name two levels, for example *compare L4-L5 and L5-S1*.",
                "image": None}

    rows = ["| Measurement | " + cmd.target + " | " + cmd.second_target + " |", "|---|---|---|"]
    if cmd.target_kind == "disc":
        a, b = _find_disc(sr, cmd.target), _find_disc(sr, cmd.second_target)
        if not a or not b:
            missing = [l for l, x in ((cmd.target, a), (cmd.second_target, b)) if not x]
            return {"text": f"Not segmented: {', '.join(missing)}.", "image": None}
        ma, mb = study_mod.disc_metrics(a, sr), study_mod.disc_metrics(b, sr)
        rows.append(f"| Height | {_px_mm(ma['height_px'], ma['height_mm'])} | "
                    f"{_px_mm(mb['height_px'], mb['height_mm'])} |")
        rows.append(f"| AP diameter | {_px_mm(ma['ap_diameter_px'], ma['ap_diameter_mm'])} | "
                    f"{_px_mm(mb['ap_diameter_px'], mb['ap_diameter_mm'])} |")
        rows.append(f"| Area | {_px_mm(ma['area_px'], ma['area_mm2'], True)} | "
                    f"{_px_mm(mb['area_px'], mb['area_mm2'], True)} |")
        ga, gb = sr.grading.get(cmd.target), sr.grading.get(cmd.second_target)
        if ga and gb and "error" not in ga and "error" not in gb:
            rows.append(f"| Pfirrmann | {ga['pfirrmann_roman']} | {gb['pfirrmann_roman']} |")
            rows.append(f"| Stenosis | {ga['stenosis_label']} | {gb['stenosis_label']} |")
    else:
        a, b = _find_vertebra(sr, cmd.target), _find_vertebra(sr, cmd.second_target)
        if not a or not b:
            missing = [l for l, x in ((cmd.target, a), (cmd.second_target, b)) if not x]
            return {"text": f"Not segmented: {', '.join(missing)}.", "image": None}
        ma, mb = study_mod.vertebra_metrics(a, sr), study_mod.vertebra_metrics(b, sr)
        rows.append(f"| Height | {_px_mm(ma['height_px'], ma['height_mm'])} | "
                    f"{_px_mm(mb['height_px'], mb['height_mm'])} |")
        rows.append(f"| Width | {_px_mm(ma['width_px'], ma['width_mm'])} | "
                    f"{_px_mm(mb['width_px'], mb['width_mm'])} |")

    ann = render.Annotations(segmentation=False, boxes=True, labels=True,
                             measurements=True, legend=False,
                             title=f"{cmd.target} vs {cmd.second_target}")
    return {"text": "\n".join(rows) + f"\n\n{_scale_note(sr)}",
            "image": render.render_series(sr, ann)}


def act_describe(st, cmd: Command) -> Dict[str, Any]:
    if not st.series:
        return {"text": "No image loaded.", "image": None}

    lines = [f"**Study contents** — {len(st.series)} image(s)\n"]
    for s in st.series:
        bits = [s.label(), s.fmt]
        if s.plane == "axial" and s.level:
            bits.append(f"level source: {s.level_source}")
        lines.append(f"- {' · '.join(bits)}  ({s.filename})")

    ref = _ref(st)
    if ref and ref.plane == "sagittal":
        discs = ", ".join(d["label"] for d in ref.discs) or "none"
        verts = ", ".join(v["label"] for v in ref.vertebrae) or "none"
        lines += [f"\n**Segmented on {ref.label()}**",
                  f"- Discs: {discs}", f"- Vertebral bodies: {verts}",
                  f"- Sacrum visible: {'yes' if ref.has_sacrum else 'no'}"]
        if ref.grading:
            graded = ", ".join(f"{l} ({g['pfirrmann_roman']})"
                               for l, g in ref.grading.items() if "error" not in g)
            lines.append(f"- Graded: {graded or 'none'}")

    regions = study_mod.abnormal_regions(st)
    if regions:
        lines.append(f"\n**Levels with a positive finding:** "
                     f"{', '.join(r['level'] for r in regions)}")
    else:
        lines.append("\nNo level reached the 0.50 reporting threshold.")

    lines.append(f"\n{_scale_note(ref)}" if ref else "")

    ann = render.Annotations(title="Study overview")
    dm = {d["label"]: study_mod.disc_metrics(d, ref) for d in ref.discs} if ref else {}
    vm = {v["label"]: study_mod.vertebra_metrics(v, ref) for v in ref.vertebrae} if ref else {}
    return {"text": "\n".join(lines),
            "image": (render.render_series(ref, ann, dm, vm,
                                           [r["level"] for r in regions]) if ref else None)}


def act_help(st, cmd: Command) -> Dict[str, Any]:
    return {"text":
            "**What you can ask**\n\n"
            "| Command | Result |\n|---|---|\n"
            "| `segment L4` | tints the L4 vertebral body only |\n"
            "| `segment L4-L5` | tints the L4-L5 disc only |\n"
            "| `show the spinal cord` | tints one anatomical class |\n"
            "| `measure L5-S1` | height, AP diameter, area |\n"
            "| `grade L4-L5` | Pfirrmann grade and stenosis |\n"
            "| `find stenosis` | screens for one named finding |\n"
            "| `locate stenosis` | marks the structure it sits in, level by level |\n"
            "| `where is the herniation` | same, phrased as a question |\n"
            "| `show spondylolisthesis` | same, on the bodies forming the level |\n"
            "| `compare L4-L5 and L5-S1` | side-by-side table |\n"
            "| `describe` | study inventory |\n"
            "| `report` | builds the printable report |\n\n"
            "Millimetres appear when the file carries pixel spacing (DICOM or "
            "NIfTI) or after you set a scale under Calibration. Otherwise sizes "
            "stay in pixels.",
            "image": None}


def act_unknown(st, cmd: Command) -> Dict[str, Any]:
    return {"text":
            "That question sits outside this tool. It reads lumbar spine images "
            "and answers on segmentation, measurement, Pfirrmann grading and the "
            "structured findings the models were trained on.\n\n"
            "Try `describe`, `segment L4-L5`, `measure L5-S1`, `grade L4-L5`, "
            "`find stenosis`, or `help`.",
            "image": None}


ACTIONS = {
    "SEGMENT": act_segment,
    "MEASURE": act_measure,
    "GRADE": act_grade,
    "FIND": act_find,
    "LOCATE": act_locate,
    "COMPARE": act_compare,
    "DESCRIBE": act_describe,
    "HELP": act_help,
    "UNKNOWN": act_unknown,
}


def execute(st, text: str) -> Dict[str, Any]:
    cmd = parse(text)
    if cmd.intent == "REPORT":
        return {"intent": "REPORT", "target": None,
                "text": "Open the Report panel to choose annotations and download "
                        "the PDF.", "image": None}
    fn = ACTIONS.get(cmd.intent, act_unknown)
    out = fn(st, cmd)
    out.update({"intent": cmd.intent, "target": cmd.target,
                "target_kind": cmd.target_kind, "expanded": cmd.expanded})
    return out


load_lexicon()
