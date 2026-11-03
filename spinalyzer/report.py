"""
PDF report.

The caller chooses which sections appear and which annotation layers are drawn
on the images, so a report can be reduced to the levels that matter instead of
carrying every box at once.

Every value printed here is read back from a stored model output or from image
metadata. Where a quantity was never measured, the cell reads "not measured".
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from fpdf import FPDF

from . import render
from . import study as study_mod

PAGE_W = 210.0
MARGIN = 14.0
CONTENT_W = PAGE_W - 2 * MARGIN

INK = (26, 26, 26)
MUTED = (105, 105, 105)
RULE = (205, 205, 205)
HEAD_BG = (238, 238, 238)
FLAG = (176, 42, 42)


@dataclass
class ReportOptions:
    """Sections and annotation layers requested by the operator."""
    # sections
    technique: bool = True
    images: bool = True
    inventory: bool = True
    vertebral_morphometry: bool = True
    disc_morphometry: bool = True
    grading: bool = True
    abnormal_regions: bool = True
    findings_table: bool = True
    axial_alignment: bool = True
    impression: bool = True
    # annotation layers drawn on the images
    ann_segmentation: bool = True
    ann_boxes: bool = True
    ann_labels: bool = True
    ann_measurements: bool = True
    ann_abnormal_only: bool = False
    ann_legend: bool = True
    # content controls
    findings_threshold: float = 0.5
    findings_positive_only: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ReportOptions":
        d = d or {}
        o = cls()
        for k in vars(o):
            if k in d:
                cur = getattr(o, k)
                setattr(o, k, float(d[k]) if isinstance(cur, float) else bool(d[k]))
        return o

    def annotations(self, title: str = "") -> render.Annotations:
        return render.Annotations(
            segmentation=self.ann_segmentation,
            boxes=self.ann_boxes,
            labels=self.ann_labels,
            measurements=self.ann_measurements,
            abnormal_only=self.ann_abnormal_only,
            legend=self.ann_legend,
            title=title,
        )


def _ascii(s: Any) -> str:
    """Core PDF fonts are latin-1; map the few symbols used to safe text."""
    t = str(s)
    for a, b in (("—", "-"), ("–", "-"), ("’", "'"),
                 ("“", '"'), ("”", '"'), ("²", "2"),
                 ("·", "-"), ("→", "->"), ("≤", "<="),
                 ("≥", ">="), ("×", "x")):
        t = t.replace(a, b)
    return t.encode("latin-1", "replace").decode("latin-1")


class _PDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, "Spinalyzer report", align="L")
        self.cell(0, 6, self.study_ref, align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*RULE)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.multi_cell(
            0, 3.6,
            _ascii("Automated output of local image models. Not a medical diagnosis. "
                   f"Page {self.page_no()} of {{nb}}"),
            align="C")


def _h1(pdf: _PDF, text: str, keep: float = 14.0) -> None:
    """Section heading. `keep` is the space its first content needs, so a
    heading never sits alone at the foot of a page."""
    if pdf.get_y() + keep > 258:
        pdf.add_page()
    pdf.ln(2)
    pdf.set_x(MARGIN)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*INK)
    pdf.cell(0, 6.5, _ascii(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*RULE)
    pdf.line(MARGIN, pdf.get_y() + 0.5, PAGE_W - MARGIN, pdf.get_y() + 0.5)
    pdf.ln(3)


def _para(pdf: _PDF, text: str, size: float = 8.6) -> None:
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(*INK)
    pdf.multi_cell(CONTENT_W, 4.4, _ascii(text))
    pdf.ln(1.2)


def _kv(pdf: _PDF, key: str, value: str) -> None:
    # multi_cell leaves the cursor wherever the last wrapped line ended, so the
    # left margin is reasserted before every pair. Without this a wrapped value
    # pushes the next key off the right edge.
    if pdf.get_y() > 255:
        pdf.add_page()
    pdf.set_x(MARGIN)
    pdf.set_font("Helvetica", "B", 8.6)
    pdf.set_text_color(*INK)
    pdf.cell(46, 4.8, _ascii(key))
    pdf.set_font("Helvetica", "", 8.6)
    pdf.multi_cell(CONTENT_W - 46, 4.8, _ascii(value),
                   new_x="LMARGIN", new_y="NEXT")


def _table(pdf: _PDF, headers: List[str], rows: List[List[str]],
           widths: Optional[List[float]] = None,
           flag_rows: Optional[set] = None) -> None:
    if not rows:
        _para(pdf, "Nothing to report in this section.")
        return
    n = len(headers)
    widths = widths or [CONTENT_W / n] * n
    flag_rows = flag_rows or set()

    # A header alone at the foot of a page is worse than a page break.
    if pdf.get_y() + 5.6 + 5.0 * min(len(rows), 2) > 262:
        pdf.add_page()

    pdf.set_x(MARGIN)
    pdf.set_font("Helvetica", "B", 7.8)
    pdf.set_fill_color(*HEAD_BG)
    pdf.set_text_color(*INK)
    for h, w in zip(headers, widths):
        pdf.cell(w, 5.6, _ascii(h), border=0, fill=True, align="L")
    pdf.ln(5.6)

    pdf.set_font("Helvetica", "", 7.8)
    pdf.set_draw_color(*RULE)
    for i, row in enumerate(rows):
        if pdf.get_y() > 262:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 7.8)
            pdf.set_fill_color(*HEAD_BG)
            for h, w in zip(headers, widths):
                pdf.cell(w, 5.6, _ascii(h), border=0, fill=True, align="L")
            pdf.ln(5.6)
            pdf.set_font("Helvetica", "", 7.8)
        pdf.set_x(MARGIN)
        pdf.set_text_color(*(FLAG if i in flag_rows else INK))
        for cell, w in zip(row, widths):
            # Trim to what the column can actually show, so text never runs
            # under its neighbour.
            txt = _ascii(cell)
            while txt and pdf.get_string_width(txt) > w - 2:
                txt = txt[:-1]
            if txt != _ascii(cell) and len(txt) > 1:
                txt = txt[:-1] + "*"
            pdf.cell(w, 5.0, txt, border=0, align="L")
        pdf.ln(5.0)
        pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
    pdf.ln(2)


def _place_image(pdf: _PDF, data_uri: str, width: float,
                 caption: str = "") -> None:
    if not data_uri or "," not in data_uri:
        return
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    if pdf.get_y() + width * 0.9 > 250:
        pdf.add_page()
    x = (PAGE_W - width) / 2
    pdf.image(io.BytesIO(raw), x=x, w=width)
    if caption:
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(CONTENT_W, 3.8, _ascii(caption), align="C")
    pdf.ln(2)


def _size(px: Optional[int], mm: Optional[float], squared: bool = False) -> str:
    if px is None:
        return "not measured"
    if mm is None:
        return f"{px} px{'2' if squared else ''}"
    return f"{px} px{'2' if squared else ''} / {mm} mm{'2' if squared else ''}"


_GRID_GAP = 3.0
_GRID_CAP = 7.0


def _cell_width(cols: int) -> float:
    return (CONTENT_W - _GRID_GAP * (cols - 1)) / cols


def _grid_height(items: List[Dict[str, str]], cols: int = 4,
                 rows_per_block: int = 2) -> float:
    """Height the first block of a grid will occupy.

    Called before the section heading so the heading and its plate stay on the
    same page; a legend strip under one thumbnail makes that block taller than
    a naive aspect-ratio guess.
    """
    if not items:
        return 0.0
    from PIL import Image as _Im
    cell_w = _cell_width(cols)
    block = items[:cols * rows_per_block]
    heights = []
    for it in block:
        raw = base64.b64decode(it["image"].split(",", 1)[1])
        with _Im.open(io.BytesIO(raw)) as im:
            heights.append(cell_w * im.height / im.width)
    rows = (len(block) + cols - 1) // cols
    return (max(heights) + _GRID_CAP) * rows


def _image_grid(pdf: _PDF, items: List[Dict[str, str]],
                cols: int = 4, rows_per_block: int = 2) -> None:
    """Lay images out `cols` across the content width, `rows_per_block` deep.

    Eight images -- the maximum a study holds -- land as two rows of four on a
    single page instead of one plate per page.
    """
    if not items:
        return
    from PIL import Image as _Im
    gap = _GRID_GAP
    cell_w = _cell_width(cols)
    cap_h = _GRID_CAP

    idx = 0
    while idx < len(items):
        block = items[idx:idx + cols * rows_per_block]
        # Measure the tallest image in this block so rows align.
        heights = []
        for it in block:
            raw = base64.b64decode(it["image"].split(",", 1)[1])
            with _Im.open(io.BytesIO(raw)) as im:
                heights.append(cell_w * im.height / im.width)
        row_h = max(heights) + cap_h

        need = row_h * ((len(block) + cols - 1) // cols)
        if pdf.get_y() + need > 258:
            pdf.add_page()

        top = pdf.get_y()
        for i, it in enumerate(block):
            r, c = divmod(i, cols)
            x = MARGIN + c * (cell_w + gap)
            y = top + r * row_h
            raw = base64.b64decode(it["image"].split(",", 1)[1])
            pdf.image(io.BytesIO(raw), x=x, y=y, w=cell_w)
            pdf.set_xy(x, y + heights[i] + 0.6)
            pdf.set_font("Helvetica", "", 5.6)
            pdf.set_text_color(*MUTED)
            pdf.cell(cell_w, 2.6, _ascii(it.get("caption", ""))[:44], align="C")
        used = ((len(block) + cols - 1) // cols) * row_h
        pdf.set_xy(MARGIN, top + used + 1.5)
        idx += len(block)
    pdf.ln(1)


# ---------------------------------------------------------------------------

def build(st, opts: ReportOptions) -> bytes:
    ref = st.primary_sagittal() or (st.series[0] if st.series else None)
    regions = study_mod.abnormal_regions(st, opts.findings_threshold)
    abnormal_levels = [r["level"] for r in regions]

    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.study_ref = f"study {st.id}"
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ---- masthead ------------------------------------------------------
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*INK)
    pdf.cell(0, 8, "LUMBAR SPINE MRI - STRUCTURED REPORT",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*INK)
    pdf.set_line_width(0.5)
    pdf.line(MARGIN, pdf.get_y() + 0.4, PAGE_W - MARGIN, pdf.get_y() + 0.4)
    pdf.set_line_width(0.2)
    pdf.ln(2.6)

    half = CONTENT_W / 2
    pdf.set_font("Helvetica", "", 7.8)
    pdf.set_text_color(*MUTED)
    left = f"Study {st.id}"
    right = f"Generated {datetime.now().strftime('%d %b %Y  %H:%M')}"
    pdf.cell(half, 4.2, _ascii(left))
    pdf.cell(half, 4.2, _ascii(right), align="R", new_x="LMARGIN", new_y="NEXT")
    seq = f"{len(st.series)} image(s): {len(st.sagittal())} sagittal, {len(st.axial())} axial"
    scale_src = ref.spacing["source"] if (ref and ref.spacing) else "no pixel-size metadata"
    pdf.cell(half, 4.2, _ascii(seq))
    pdf.cell(half, 4.2, _ascii(f"Scale: {scale_src}"), align="R",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # ---- impression, first ---------------------------------------------
    # A reporting radiologist reads the conclusion before the evidence, so the
    # impression opens the document rather than closing it.
    if opts.impression:
        _h1(pdf, "Impression")
        for line in _impression(st, regions, opts):
            _para(pdf, line, 8.8)
        pdf.ln(0.5)

    # ---- key levels at a glance ----------------------------------------
    if opts.abnormal_regions:
        _h1(pdf, "Levels meeting the reporting threshold")
        if not regions:
            _para(pdf, f"No level reached {opts.findings_threshold:.2f}.")
        else:
            rows = []
            for r in regions:
                m = r["metrics"]
                g = ref.grading.get(r["level"], {}) if ref else {}
                kinds = ", ".join(x["kind"] for x in r["reasons"][:5])
                rows.append([r["level"],
                             g.get("pfirrmann_roman", "-"),
                             _size(m["height_px"], m["height_mm"]),
                             _size(m["ap_diameter_px"], m["ap_diameter_mm"]),
                             kinds])
            _table(pdf, ["Level", "Pf", "Disc height", "AP diameter", "Findings"],
                   rows, [20, 12, 38, 38, 74], set(range(len(rows))))

    # ---- image plate ----------------------------------------------------
    if opts.images and ref is not None:
        # Render first, then measure, so the heading knows how much room the
        # plate needs and does not strand itself at the foot of a page.
        plate: List[Dict[str, str]] = []
        for i, s in enumerate(st.series):
            dm = {d["label"]: study_mod.disc_metrics(d, s) for d in s.discs}
            vm = {v["label"]: study_mod.vertebra_metrics(v, s) for v in s.vertebrae}
            marks = abnormal_levels if s is ref else []
            ann = opts.annotations(s.label())
            # One legend per plate is enough; repeating it under eight
            # thumbnails costs height and says nothing new.
            ann.legend = opts.ann_legend and i == 0
            plate.append({
                "image": render.render_series(s, ann, dm, vm, marks),
                "caption": s.label() + (f"  [{s.level_source}]"
                                        if s.plane == "axial" else ""),
            })

        _h1(pdf, "Imaging", keep=_grid_height(plate) + 12)
        layers = [n for n, on in (("segmentation", opts.ann_segmentation),
                                  ("outlines", opts.ann_boxes),
                                  ("labels", opts.ann_labels),
                                  ("measurements", opts.ann_measurements)) if on]
        scope = ("levels above threshold only" if opts.ann_abnormal_only
                 else "all segmented levels")
        _para(pdf, f"Layers: {', '.join(layers) if layers else 'none'}.  Scope: {scope}.",
              7.8)
        _image_grid(pdf, plate, cols=4, rows_per_block=2)

        if regions:
            _para(pdf, "Levels above threshold, outlined and measured:", 7.8)
            _image_grid(pdf, [{"image": render.render_abnormal_panel(ref, regions),
                               "caption": "positive levels"}],
                        cols=4, rows_per_block=1)

    # ---- inventory ------------------------------------------------------
    if opts.inventory and ref is not None:
        _h1(pdf, "Anatomical inventory")
        _kv(pdf, "Reference series", ref.label())
        _kv(pdf, "Disc levels", ", ".join(d["label"] for d in ref.discs) or "none segmented")
        _kv(pdf, "Vertebral bodies", ", ".join(v["label"] for v in ref.vertebrae) or "none segmented")
        _kv(pdf, "Sacrum visible", "yes" if ref.has_sacrum else "no")
        pdf.ln(1)

    # Axial alignment
    if opts.axial_alignment and st.axial():
        _h1(pdf, "Axial to sagittal level alignment")
        _para(pdf, "Level source states the evidence: 'filename' was read from the "
                   "file name, 'manual' was set by the operator, 'order' means the "
                   "file carried no level so slices were mapped to the inferior-most "
                   "levels in upload order.")
        rows = [[s.filename[:34], s.weighting, s.level or "unassigned", s.level_source]
                for s in st.axial()]
        _table(pdf, ["File", "Weighting", "Level", "Level source"],
               rows, [78, 26, 34, 44])

    # Vertebral morphometry
    if opts.vertebral_morphometry and ref is not None and ref.vertebrae:
        _h1(pdf, "Vertebral body morphometry")
        rows = []
        for v in ref.vertebrae:
            m = study_mod.vertebra_metrics(v, ref)
            rows.append([m["label"], _size(m["height_px"], m["height_mm"]),
                         _size(m["width_px"], m["width_mm"]),
                         _size(m["area_px"], m["area_mm2"], True)])
        _table(pdf, ["Body", "Height", "Width", "Area"], rows, [30, 51, 51, 50])

    # Disc morphometry
    if opts.disc_morphometry and ref is not None and ref.discs:
        _h1(pdf, "Intervertebral disc morphometry")
        rows, flags = [], set()
        for i, d in enumerate(ref.discs):
            m = study_mod.disc_metrics(d, ref)
            if m["label"] in abnormal_levels:
                flags.add(i)
            rows.append([m["label"], _size(m["height_px"], m["height_mm"]),
                         _size(m["ap_diameter_px"], m["ap_diameter_mm"]),
                         _size(m["area_px"], m["area_mm2"], True)])
        _table(pdf, ["Level", "Height", "AP diameter", "Area"],
               rows, [30, 51, 51, 50], flags)
        if not ref.spacing:
            _para(pdf, "No pixel-size metadata accompanied this file, so no "
                       "millimetre value is given. Set a scale under Calibration "
                       "to convert.")

    # Grading
    if opts.grading and ref is not None and ref.grading:
        _h1(pdf, "Pfirrmann grading and stenosis")
        rows, flags = [], set()
        for i, lvl in enumerate(("L3-L4", "L4-L5", "L5-S1")):
            g = ref.grading.get(lvl)
            if not g or "error" in g:
                continue
            if g["pfirrmann_grade"] >= 3 or g["stenosis"] >= 1:
                flags.add(len(rows))
            rows.append([lvl, g["pfirrmann_roman"], g["pfirrmann_desc"],
                         f"{g['stenosis_label']} ({g['stenosis_probs'][g['stenosis']]:.2f})"])
        _table(pdf, ["Level", "Grade", "Description", "Stenosis"],
               rows, [24, 18, 96, 44], flags)
        _para(pdf, "Grades come from an ordinal head with four cumulative "
                   "boundaries; the grade is the count of boundaries above 0.50. "
                   "Only the three levels above were in the training distribution.")

    # Per-level evidence behind each flagged level
    if opts.abnormal_regions and regions:
        _h1(pdf, "Evidence by level")
        for r in regions:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*FLAG)
            pdf.cell(0, 5.2, _ascii(r["level"]), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*INK)
            rows = [[x["kind"], x["detail"] or "-",
                     f"{x['score']:.2f}" if x["score"] is not None else "-",
                     x["source"]] for x in r["reasons"]]
            _table(pdf, ["Finding", "Detail", "Score", "Source"],
                   rows, [46, 76, 20, 40])

    # Findings table
    if opts.findings_table and ref is not None and ref.findings \
            and "binary_findings" in ref.findings:
        _h1(pdf, "Structured findings")
        items = ref.findings["binary_findings"]
        if opts.findings_positive_only:
            items = [f for f in items if f["probability"] >= opts.findings_threshold]
        items = sorted(items, key=lambda f: -f["probability"])
        rows, flags = [], set()
        for f in items:
            if f["positive"] and f["pathological"]:
                flags.add(len(rows))
            rows.append([f["name"], f"{f['probability']:.3f}",
                         "positive" if f["positive"] else "negative",
                         "abnormality" if f["pathological"] else "descriptor"])
        _table(pdf, ["Finding", "Probability", "Verdict", "Type"],
               rows, [82, 30, 30, 40], flags)
        _kv(pdf, "L4-L5 disc type", f"{ref.findings['l4_l5_disc_type']} "
            f"(p = {ref.findings['l4_l5_disc_type_prob']:.2f})")
        _kv(pdf, "L5-S1 disc type", f"{ref.findings['l5_s1_disc_type']} "
            f"(p = {ref.findings['l5_s1_disc_type_prob']:.2f})")
        _kv(pdf, "Overall severity", ref.findings["severity"])

    # Technique closes the report: provenance, not the headline.
    if opts.technique:
        _h1(pdf, "Technique and provenance")
        rows = []
        for s in st.series:
            scale = (f"{s.spacing['row']} x {s.spacing['col']} mm/px "
                     f"({s.spacing['source']})" if s.spacing else "pixels only")
            rows.append([s.label(), s.filename[:30], s.fmt, scale])
        _table(pdf, ["Series", "File", "Format", "Scale"], rows, [40, 60, 24, 58])
        _kv(pdf, "Segmentation", "DINOv2-ViT-B/14, LoRA r=16, 384 x 384")
        _kv(pdf, "Grading", "Ordinal Pfirrmann head; trained on L3-L4, L4-L5, L5-S1")
        _kv(pdf, "Findings", "ResNet-18 disc encoder, multi-disc attention aggregator")
        _kv(pdf, "Reporting threshold", f"{opts.findings_threshold:.2f}")

    pdf.ln(3)
    pdf.set_draw_color(*RULE)
    pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 7.6)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(CONTENT_W, 3.8, _ascii(
        "Every value above was read back from a model output or from image "
        "metadata stored with this study. Quantities that were never measured "
        "are marked 'not measured'. This document is automated research output "
        "and is not a medical diagnosis."))

    out = pdf.output()
    return bytes(out)


def _impression(st, regions: List[Dict[str, Any]], opts: ReportOptions) -> List[str]:
    ref = st.primary_sagittal() or (st.series[0] if st.series else None)
    if ref is None:
        return ["No image was analysed."]

    lines: List[str] = []
    n_sag, n_ax = len(st.sagittal()), len(st.axial())
    lines.append(f"Study of {len(st.series)} image(s): {n_sag} sagittal and "
                 f"{n_ax} axial. Segmentation identified "
                 f"{len(ref.discs)} disc level(s) and {len(ref.vertebrae)} "
                 f"vertebral body/bodies on {ref.label()}.")

    graded = {l: g for l, g in ref.grading.items() if "error" not in g}
    if graded:
        parts = [f"{l} grade {g['pfirrmann_roman']}" for l, g in sorted(graded.items())]
        lines.append("Pfirrmann grading: " + "; ".join(parts) + ".")
        degen = [l for l, g in graded.items() if g["pfirrmann_grade"] >= 3]
        if degen:
            lines.append("Grade IV or V, indicating loss of disc height and signal, "
                         f"at {', '.join(sorted(degen))}.")
        sten = [(l, g["stenosis_label"]) for l, g in graded.items() if g["stenosis"] >= 1]
        if sten:
            lines.append("Stenosis above mild at " +
                         "; ".join(f"{l} ({s})" for l, s in sorted(sten)) + ".")
    else:
        lines.append("No level carried a grade. The grading head covers L3-L4, "
                     "L4-L5 and L5-S1 only.")

    if regions:
        lines.append(f"{len(regions)} level(s) met the "
                     f"{opts.findings_threshold:.2f} threshold: "
                     f"{', '.join(r['level'] for r in regions)}.")
        for r in regions:
            m = r["metrics"]
            size = (f"{m['height_mm']} mm height" if m["height_mm"] is not None
                    else f"{m['height_px']} px height")
            kinds = ", ".join(x["kind"] for x in r["reasons"][:4])
            lines.append(f"{r['level']}: {size}; {kinds}.")
    else:
        lines.append(f"No finding reached the {opts.findings_threshold:.2f} threshold.")

    if not ref.spacing:
        lines.append("This study carried no pixel-size metadata, so measurements "
                     "are in pixels and cannot be converted without a calibration "
                     "value.")
    return lines
