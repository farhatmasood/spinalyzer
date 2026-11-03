"""
Image rendering.

Two rules govern every function here:

  A request for one level draws that one level. Asking for L4 tints the L4
  vertebral body and nothing else; the rest of the slice stays as greyscale
  anatomy.

  Annotation is opt-in. The caller states which layers it wants and the image
  carries only those, so a report can be reduced to the levels that matter.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
HIGHLIGHT = (0, 255, 255)      # cyan, target outline
ABNORMAL = (255, 80, 80)       # red, abnormal region
NORMAL_BOX = (150, 150, 150)   # grey, unremarkable region
TEXT = (240, 240, 240)
BAR_BG = (26, 26, 26)
FINDING = (255, 196, 0)        # amber, located finding

# The Hershey fonts OpenCV ships carry ASCII only; anything else is drawn as
# "??". Text bound for cv2.putText goes through here first.
_SUBS = {"·": "-", "–": "-", "—": "-", "→": "->",
         "²": "2", "’": "'", "“": '"', "”": '"',
         "×": "x", "≤": "<=", "≥": ">="}


def _ascii(text: str) -> str:
    t = str(text)
    for a, b in _SUBS.items():
        t = t.replace(a, b)
    return t.encode("ascii", "ignore").decode("ascii")


@dataclass
class Annotations:
    """Which layers an image should carry."""
    segmentation: bool = True      # translucent class colours
    boxes: bool = True             # component outlines
    labels: bool = True            # anatomical names
    measurements: bool = False     # size text beside each box
    abnormal_only: bool = False    # draw only levels with a positive finding
    legend: bool = True
    title: str = ""

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Annotations":
        d = d or {}
        return cls(
            segmentation=bool(d.get("segmentation", True)),
            boxes=bool(d.get("boxes", True)),
            labels=bool(d.get("labels", True)),
            measurements=bool(d.get("measurements", False)),
            abnormal_only=bool(d.get("abnormal_only", False)),
            legend=bool(d.get("legend", True)),
            title=str(d.get("title", "")),
        )


def encode_png(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def preview(rgb: np.ndarray, max_dim: int = 560) -> str:
    h, w = rgb.shape[:2]
    s = max_dim / max(h, w)
    if s < 1.0:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
    return encode_png(rgb)


# ---------------------------------------------------------------------------
# Component isolation
# ---------------------------------------------------------------------------

def component_mask(sr, class_id: int, component_id: int) -> np.ndarray:
    """Boolean mask of exactly one connected component.

    Recomputed with the same connectivity used when the components were first
    detected, so component_id refers to the same blob.
    """
    binary = (sr.mask == class_id).astype(np.uint8)
    n, labelmap = cv2.connectedComponents(binary)
    if component_id < 1 or component_id >= n:
        return np.zeros_like(sr.mask, dtype=bool)
    return labelmap == component_id


def _class_id_for(sr, kind: str) -> Optional[int]:
    name = {"disc": "IVD", "vertebra": "VB"}.get(kind, kind)
    for cid, n in sr.class_names.items():
        if n == name:
            return cid
    return None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _grey_base(sr) -> np.ndarray:
    """Anatomy with no class colour, so a tint reads as the answer."""
    g = cv2.cvtColor(sr.seg_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def _tint(base: np.ndarray, mask_bool: np.ndarray,
          colour: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    if not mask_bool.any():
        return base
    layer = base.copy()
    layer[mask_bool] = colour
    return cv2.addWeighted(base, 1.0 - alpha, layer, alpha, 0)


def _outline(img: np.ndarray, mask_bool: np.ndarray,
             colour: Tuple[int, int, int], thickness: int = 2) -> None:
    if not mask_bool.any():
        return
    contours, _ = cv2.findContours((mask_bool * 255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, colour, thickness, cv2.LINE_AA)


def _caption(img: np.ndarray, text: str, anchor: Tuple[int, int],
             colour: Tuple[int, int, int], scale: float = 0.42,
             align: str = "left") -> None:
    """Draw a label with a solid backing so it stays legible over anatomy.

    `align="right"` places the text so it ENDS at the anchor, which lets
    vertebral labels sit left of their box while disc labels sit right, instead
    of every label stacking into the same column.
    """
    if not text:
        return
    text = _ascii(text)
    x, y = anchor
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    if align == "right":
        x -= tw
    x = max(2, min(x, img.shape[1] - tw - 4))
    y = max(th + 4, min(y, img.shape[0] - 4))
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + 3), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)


def _title_bar(img: np.ndarray, title: str) -> np.ndarray:
    if not title:
        return img
    bar = np.full((26, img.shape[1], 3), BAR_BG, dtype=np.uint8)
    cv2.putText(bar, _ascii(title)[:80], (8, 18), FONT, 0.45, TEXT, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _legend_bar(sr, width: int) -> np.ndarray:
    rows = [(cid, n) for cid, n in sr.class_names.items() if cid != 0]
    bar = np.full((16 + 18 * len(rows), width, 3), BAR_BG, dtype=np.uint8)
    y = 16
    for cid, name in rows:
        col = sr.class_colors.get(cid, (255, 255, 255))
        cv2.rectangle(bar, (10, y - 9), (26, y + 3), col, -1)
        cv2.putText(bar, name, (32, y + 2), FONT, 0.38, TEXT, 1, cv2.LINE_AA)
        y += 18
    return bar


def _size_text(m: Dict[str, Any], kind: str) -> str:
    if kind == "disc":
        if m.get("height_mm") is not None:
            return f"h {m['height_mm']} mm · AP {m['ap_diameter_mm']} mm"
        return f"h {m['height_px']} px · AP {m['ap_diameter_px']} px"
    if m.get("height_mm") is not None:
        return f"{m['height_mm']} x {m['width_mm']} mm"
    return f"{m['height_px']} x {m['width_px']} px"


# ---------------------------------------------------------------------------
# Single-target view
# ---------------------------------------------------------------------------

def render_target(sr, region: Dict[str, Any], kind: str,
                  metrics: Optional[Dict[str, Any]] = None,
                  title: str = "") -> str:
    """Tint one component. Everything else stays greyscale."""
    cid = _class_id_for(sr, kind)
    base = _grey_base(sr)

    if cid is not None and "component_id" in region:
        blob = component_mask(sr, cid, region["component_id"])
        colour = sr.class_colors.get(cid, HIGHLIGHT)
        base = _tint(base, blob, colour, alpha=0.55)
        _outline(base, blob, HIGHLIGHT, 2)
    else:
        x, y, w, h = region["bbox"]
        cv2.rectangle(base, (x, y), (x + w, y + h), HIGHLIGHT, 2)

    x, y, w, h = region["bbox"]
    label = region.get("label", "")
    if label:
        _caption(base, label, (x + w + 6, y + h // 2), HIGHLIGHT, 0.48)
    if metrics:
        _caption(base, _size_text(metrics, kind), (x + w + 6, y + h // 2 + 16), TEXT, 0.4)

    return encode_png(_title_bar(base, title))


def render_class(sr, class_id: int, title: str = "") -> str:
    """Tint every pixel of one class -- used for 'show the spinal cord'."""
    base = _grey_base(sr)
    blob = sr.mask == class_id
    if not blob.any():
        return encode_png(_title_bar(base, title + " (not present)"))
    colour = sr.class_colors.get(class_id, HIGHLIGHT)
    base = _tint(base, blob, colour, alpha=0.55)
    _outline(base, blob, HIGHLIGHT, 1)
    return encode_png(_title_bar(base, title))


# ---------------------------------------------------------------------------
# Study view with annotation control
# ---------------------------------------------------------------------------

def render_series(sr, ann: Annotations,
                  disc_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
                  vertebra_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
                  abnormal_levels: Optional[Sequence[str]] = None) -> str:
    """Compose a series image from the requested layers only."""
    abnormal = set(abnormal_levels or ())
    base = _grey_base(sr)

    if ann.segmentation:
        overlay = np.zeros_like(base)
        painted = np.zeros(sr.mask.shape, dtype=bool)
        for cid, col in sr.class_colors.items():
            if cid == 0:
                continue
            sel = sr.mask == cid
            overlay[sel] = col
            painted |= sel
        base = _tint(base, painted, (0, 0, 0), 0.0)          # keep shape
        blended = cv2.addWeighted(base, 0.68, overlay, 0.32, 0)
        base = np.where(painted[..., None], blended, base).astype(np.uint8)

    def draw(items: List[Dict[str, Any]], kind: str,
             metrics: Optional[Dict[str, Dict[str, Any]]],
             side: str = "right") -> None:
        """Annotate one family of components.

        Vertebral labels go to the left of their box and disc labels to the
        right, so a sagittal slice carrying fourteen components does not stack
        every caption into one column.
        """
        for it in items:
            label = it.get("label", "")
            is_abnormal = label in abnormal
            if ann.abnormal_only and not is_abnormal:
                continue
            colour = ABNORMAL if is_abnormal else NORMAL_BOX
            x, y, w, h = it["bbox"]
            if ann.boxes:
                cv2.rectangle(base, (x, y), (x + w, y + h), colour, 2 if is_abnormal else 1)
            if side == "right":
                anchor_x, align = x + w + 6, "left"
            else:
                anchor_x, align = x - 6, "right"
            line = y + h // 2
            if ann.labels and label:
                _caption(base, label, (anchor_x, line), colour, 0.44, align)
                line += 14
            if ann.measurements and metrics and label in metrics:
                _caption(base, _size_text(metrics[label], kind),
                         (anchor_x, line), TEXT, 0.36, align)

    if sr.plane == "sagittal":
        draw(sr.vertebrae, "vertebra", vertebra_metrics, side="left")
    draw(sr.discs, "disc", disc_metrics, side="right")

    out = _title_bar(base, ann.title or sr.label())
    if ann.legend and ann.segmentation:
        out = np.vstack([out, _legend_bar(sr, out.shape[1])])
    return encode_png(out)


def render_finding(sr, mark: Dict[str, Any], finding: str,
                   score: Optional[float] = None, title: str = "") -> str:
    """Mark the structures one finding describes, on one series.

    Regions come from `findings.locate`, which pairs a scored level with the
    segmented structure the finding sits in. Everything else stays greyscale so
    the marked anatomy is the only thing tinted.
    """
    base = _grey_base(sr)
    drawn = 0

    for r in mark.get("regions", []):
        cid = r.get("class_id")
        blob = None

        if r["kind"] == "component" and cid is not None and r.get("component_id"):
            blob = component_mask(sr, cid, r["component_id"])
        elif r["kind"] == "class" and cid is not None:
            blob = sr.mask == cid
            band = r.get("band")
            if band:
                clip = np.zeros_like(blob)
                clip[int(band[0]):int(band[1]), :] = True
                blob = blob & clip

        if blob is not None and blob.any():
            base = _tint(base, blob, FINDING, 0.42)
            _outline(base, blob, FINDING, 2)
            ys, xs = np.where(blob)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            drawn += 1
        elif r.get("bbox"):
            x1, y1, w, h = r["bbox"]
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(base, (x1, y1), (x2, y2), FINDING, 2)
            drawn += 1
        else:
            continue

        tag = r.get("label", "") or r.get("structure", "")
        _caption(base, tag, (x2 + 6, (y1 + y2) // 2), FINDING, 0.44)

    head = title or f"{finding} - {mark.get('level', '')} - {sr.label()}"
    if score is not None:
        head += f"  (score {score:.2f})"
    out = _title_bar(base, head)

    foot = np.full((20, out.shape[1], 3), BAR_BG, dtype=np.uint8)
    note = ("outline = segmented structure, not a predicted lesion boundary"
            if drawn else "no segmented structure to mark on this series")
    cv2.putText(foot, _ascii(note), (8, 14), FONT, 0.34, (170, 170, 170), 1, cv2.LINE_AA)
    return encode_png(np.vstack([out, foot]))


def render_abnormal_panel(sr, regions: List[Dict[str, Any]], title: str = "") -> str:
    """Abnormal levels only, each outlined and measured."""
    base = _grey_base(sr)
    cid = _class_id_for(sr, "disc")
    by_label = {d.get("label"): d for d in sr.discs}

    for r in regions:
        d = by_label.get(r["level"])
        if d is None:
            continue
        if cid is not None and "component_id" in d:
            blob = component_mask(sr, cid, d["component_id"])
            base = _tint(base, blob, ABNORMAL, 0.42)
            _outline(base, blob, ABNORMAL, 2)
        x, y, w, h = r["bbox"]
        cv2.rectangle(base, (x, y), (x + w, y + h), ABNORMAL, 2)
        _caption(base, r["level"], (x + w + 6, y + h // 2), ABNORMAL, 0.46)
        _caption(base, _size_text(r["metrics"], "disc"),
                 (x + w + 6, y + h // 2 + 15), TEXT, 0.38)

    if not regions:
        _caption(base, "No level met the reporting threshold", (10, 22), TEXT, 0.45)
    return encode_png(_title_bar(base, title or "Levels with a positive finding"))
