"""
Model loading and raw inference.

Every number produced here comes from a trained model output or from image
metadata. Nothing is estimated, defaulted, or inferred when the underlying
evidence is absent -- absent values are returned as None.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .models import SegmentationAdapter, GradingAdapter, StructuredFindingsAdapter

log = logging.getLogger("spinalyzer.engine")

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = Path(os.environ.get("SPINALYZER_CKPT_DIR", str(REPO_ROOT / "checkpoints")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEG_IMG_SIZE = 384
GRADE_IMG_SIZE = 224
FINDINGS_IMG_SIZE = 224

SAGITTAL_CLASSES = {0: "Background", 1: "VB", 2: "IVD", 3: "Sacrum", 4: "SC"}
AXIAL_CLASSES = {0: "Background", 1: "IVD", 2: "PE", 3: "SC"}

SAGITTAL_COLORS = {
    0: (0, 0, 0),
    1: (255, 165, 0),     # VB
    2: (255, 64, 64),     # IVD
    3: (200, 0, 255),     # Sacrum
    4: (64, 200, 255),    # SC
}
AXIAL_COLORS = {
    0: (0, 0, 0),
    1: (255, 64, 64),     # IVD
    2: (0, 220, 120),     # PE
    3: (64, 200, 255),    # SC
}

CLASS_FULL_NAMES = {
    "VB": "Vertebral body",
    "IVD": "Intervertebral disc",
    "Sacrum": "Sacrum",
    "SC": "Spinal cord",
    "PE": "Posterior elements",
}

PFIRRMANN_ROMAN = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V"}
PFIRRMANN_DESC = {
    0: "homogeneous bright signal, normal disc height",
    1: "inhomogeneous bright signal, normal disc height",
    2: "inhomogeneous intermediate signal, slight loss of height",
    3: "inhomogeneous low signal, loss of disc height",
    4: "collapsed disc, no nucleus/annulus distinction",
}

STENOSIS_LABELS = {0: "Mild", 1: "Moderate", 2: "Severe"}
SEVERITY_LABELS = {0: "Mild", 1: "Moderate", 2: "Severe"}

DISC_TYPE_LABELS = {
    0: "None",
    1: "Diffuse bulge",
    2: "Central protrusion",
    3: "Paracentral protrusion",
    4: "Foraminal protrusion",
    5: "Extrusion",
}

BINARY_FINDING_NAMES = [
    "Muscle Spasm", "Disc Herniation", "Disc Protrusion", "Disc Bulge",
    "Disc Extrusion", "Thecal Sac Compression", "Nerve Root Compression",
    "Spinal Canal Stenosis", "Spondylolisthesis", "Annular Tear",
    "Desiccated Disc", "Ligamentum Flavum Hypertrophy", "Facet Joint Hypertrophy",
    "Schmorl Nodules", "Endplate Degeneration", "Modic Changes",
    "Neural Foraminal Narrowing", "L1-L2 Involvement", "L2-L3 Involvement",
    "L3-L4 Involvement", "L4-L5 Involvement", "L5-S1 Involvement",
    "D11-D12 Involvement", "D12-L1 Involvement", "Left-Sided Pathology",
    "Right-Sided Pathology", "Central Pathology", "Compression Fracture",
    "Previous Surgery", "Hemangioma", "Tarlov Cyst", "Lipoma",
    "Adequate Spinal Canal", "No Significant Compression",
    "L4-L5 Compression", "L5-S1 Compression",
]

# Findings that describe an actual abnormality. Used to decide which regions
# get annotated on the report image. Normal/negative statements are excluded.
PATHOLOGICAL_FINDINGS = {
    "Muscle Spasm", "Disc Herniation", "Disc Protrusion", "Disc Bulge",
    "Disc Extrusion", "Thecal Sac Compression", "Nerve Root Compression",
    "Spinal Canal Stenosis", "Spondylolisthesis", "Annular Tear",
    "Desiccated Disc", "Ligamentum Flavum Hypertrophy", "Facet Joint Hypertrophy",
    "Schmorl Nodules", "Endplate Degeneration", "Modic Changes",
    "Neural Foraminal Narrowing", "Compression Fracture", "Hemangioma",
    "Tarlov Cyst", "Lipoma",
}

# The grading head was trained on exactly these three levels. Requests for any
# other level are refused rather than answered with an out-of-distribution guess.
IN_DIST_DISCS = {"L3-L4": 0, "L4-L5": 1, "L5-S1": 2}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class Registry:
    seg: Optional[SegmentationAdapter] = None
    grade: Optional[GradingAdapter] = None
    findings: Optional[StructuredFindingsAdapter] = None
    load_errors: Dict[str, str] = {}


REG = Registry()


def _load_state(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def _load_segmentation() -> Optional[SegmentationAdapter]:
    path = CKPT_ROOT / "segmentation" / "best_model.pth"
    if not path.exists():
        REG.load_errors["segmentation"] = f"checkpoint not found at {path}"
        return None
    try:
        model = SegmentationAdapter(
            img_size=SEG_IMG_SIZE, patch_size=14,
            num_axial_classes=len(AXIAL_CLASSES),
            num_sagittal_classes=len(SAGITTAL_CLASSES),
            hidden_dim=768, lora_r=16, lora_alpha=32,
        )
        missing, unexpected = model.load_state_dict(_load_state(path), strict=False)
        log.info("segmentation: %d missing / %d unexpected keys", len(missing), len(unexpected))
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        log.exception("segmentation load failed")
        REG.load_errors["segmentation"] = str(e)
        return None


def _load_grading() -> Optional[GradingAdapter]:
    path = CKPT_ROOT / "grading" / "best_model.pth"
    if not path.exists():
        REG.load_errors["grading"] = f"checkpoint not found at {path}"
        return None
    try:
        model = GradingAdapter(
            img_size=GRADE_IMG_SIZE, hidden_dim=768,
            num_pfirrmann_classes=5, num_stenosis_classes=3, num_disc_levels=3,
            lora_r=8, lora_alpha=16, use_mc_dropout=False,
        )
        missing, unexpected = model.load_state_dict(_load_state(path), strict=False)
        log.info("grading: %d missing / %d unexpected keys", len(missing), len(unexpected))
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        log.exception("grading load failed")
        REG.load_errors["grading"] = str(e)
        return None


def _load_findings() -> Optional[StructuredFindingsAdapter]:
    path = CKPT_ROOT / "findings" / "best_model.pth"
    if not path.exists():
        REG.load_errors["findings"] = f"checkpoint not found at {path}"
        return None
    try:
        model = StructuredFindingsAdapter(
            backbone_name="resnet18", hidden_dim=256,
            num_binary_labels=len(BINARY_FINDING_NAMES),
            num_disc_types=len(DISC_TYPE_LABELS),
            num_severity_classes=len(SEVERITY_LABELS),
            num_pfirrmann_classes=5, freeze_backbone=False, dropout=0.3,
        )
        missing, unexpected = model.load_state_dict(_load_state(path), strict=False)
        log.info("findings: %d missing / %d unexpected keys", len(missing), len(unexpected))
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        log.exception("findings load failed")
        REG.load_errors["findings"] = str(e)
        return None


def load_all() -> None:
    log.info("device=%s  checkpoints=%s", DEVICE, CKPT_ROOT)
    REG.seg = _load_segmentation()
    REG.grade = _load_grading()
    REG.findings = _load_findings()
    log.info("loaded  seg=%s grade=%s findings=%s",
             REG.seg is not None, REG.grade is not None, REG.findings is not None)


def status() -> Dict[str, Any]:
    return {
        "device": str(DEVICE),
        "models": {
            "segmentation": REG.seg is not None,
            "grading": REG.grade is not None,
            "findings": REG.findings is not None,
        },
        "errors": REG.load_errors,
        "checkpoint_dir": str(CKPT_ROOT),
    }


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _normalise_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return np.stack([gray] * 3, axis=-1)
    if gray.ndim == 3 and gray.shape[2] == 1:
        return np.repeat(gray, 3, axis=2)
    return gray


def _looks_like_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def load_image(data: bytes, filename: str) -> Tuple[np.ndarray, str, Optional[Dict[str, Any]]]:
    """Return (rgb, format, spacing).

    `spacing` is {"row": mm, "col": mm, "source": str} only when the file
    actually carries physical pixel size. Plain raster formats (PNG/JPG/BMP/TIF)
    have no such metadata, so None is returned and no scale is invented.
    """
    low = filename.lower()

    if low.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        suffix = ".nii.gz" if low.endswith(".gz") else ".nii"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / f"volume{suffix}"
            tmp.write_bytes(data)
            nii = nib.load(str(tmp), mmap=False)
            vol = np.asarray(nii.get_fdata())
            zooms = nii.header.get_zooms()
        axis = int(np.argmin(vol.shape))
        sl = _normalise_uint8(np.take(vol, vol.shape[axis] // 2, axis=axis))
        in_plane = [float(z) for i, z in enumerate(zooms[:3]) if i != axis]
        spacing = None
        if len(in_plane) >= 2 and all(v > 0 for v in in_plane[:2]):
            spacing = {"row": in_plane[0], "col": in_plane[1], "source": "NIfTI header"}
        return _to_rgb(sl), "NIfTI (middle slice)", spacing

    if low.endswith(".dcm") or _looks_like_dicom(data):
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(data), force=True)
        arr = ds.pixel_array
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        arr = _normalise_uint8(arr.astype(np.float32))
        spacing = None
        ps = getattr(ds, "PixelSpacing", None)
        if ps is not None and len(ps) >= 2:
            spacing = {"row": float(ps[0]), "col": float(ps[1]), "source": "DICOM PixelSpacing"}
        else:
            ip = getattr(ds, "ImagerPixelSpacing", None)
            if ip is not None and len(ip) >= 2:
                spacing = {"row": float(ip[0]), "col": float(ip[1]),
                           "source": "DICOM ImagerPixelSpacing"}
        meta = {
            "series": str(getattr(ds, "SeriesDescription", "") or ""),
            "sequence": str(getattr(ds, "ScanningSequence", "") or ""),
        }
        if spacing:
            spacing["dicom"] = meta
        return _to_rgb(arr), "DICOM", spacing

    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img), (img.format or "image"), None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_segmentation(rgb_seg: np.ndarray, plane: str) -> np.ndarray:
    """rgb_seg must already be SEG_IMG_SIZE x SEG_IMG_SIZE."""
    if REG.seg is None:
        raise RuntimeError("segmentation model unavailable")
    t = torch.from_numpy(rgb_seg).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        masks, _ = (REG.seg.forward_axial(t) if plane == "axial"
                    else REG.seg.forward_sagittal(t))
    return masks.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def connected_components(mask: np.ndarray, class_id: int,
                         min_area: int = 150) -> List[Dict[str, Any]]:
    binary = (mask == class_id).astype(np.uint8)
    n, labelmap, stats, centroids = cv2.connectedComponentsWithStats(binary)
    out: List[Dict[str, Any]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append({
            "component_id": int(i),
            "centroid_x": float(centroids[i][0]),
            "centroid_y": float(centroids[i][1]),
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(area),
        })
    out.sort(key=lambda d: d["centroid_y"])
    return out


def crop_region(rgb: np.ndarray, region: Dict[str, Any], context: float = 2.5) -> np.ndarray:
    H, W = rgb.shape[:2]
    x, y, w, h = region["bbox"]
    cx, cy = x + w // 2, y + h // 2
    size = max(int(max(w, h) * context), 64)
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(W, x1 + size)
    y2 = min(H, y1 + size)
    if x2 - x1 < size:
        x1 = max(0, x2 - size)
    if y2 - y1 < size:
        y1 = max(0, y2 - size)
    return rgb[y1:y2, x1:x2]


def grade_disc(rgb_seg: np.ndarray, disc: Dict[str, Any], level_idx: int) -> Dict[str, Any]:
    """Pfirrmann + stenosis for one disc. `level_idx` must be 0/1/2 (L3-L4/L4-L5/L5-S1)."""
    if REG.grade is None:
        return {"error": "grading model unavailable"}
    crop = crop_region(rgb_seg, disc)
    if crop.size == 0:
        return {"error": "empty crop"}
    crop = cv2.resize(crop, (GRADE_IMG_SIZE, GRADE_IMG_SIZE))
    x = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    lvl = torch.tensor([level_idx], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = REG.grade(x, disc_level=lvl)

    # The Pfirrmann head is ordinal: four cumulative-sigmoid boundaries.
    logits = out["pfirrmann_logits"][0].cpu().numpy()
    cum = 1.0 / (1.0 + np.exp(-logits))
    grade = int((cum > 0.5).sum())

    sten = torch.softmax(out["stenosis_logits"][0], 0).cpu().numpy()
    pathology = {k: float(torch.sigmoid(v[0]).item())
                 for k, v in out["pathology_logits"].items()}
    # The herniation head sits outside pathology_heads but scores the same disc.
    if "herniation_logit" in out:
        pathology["herniation"] = float(torch.sigmoid(out["herniation_logit"][0]).item())
    return {
        "pfirrmann_grade": grade,
        "pfirrmann_roman": PFIRRMANN_ROMAN[grade],
        "pfirrmann_desc": PFIRRMANN_DESC[grade],
        "pfirrmann_cumulative": [round(float(v), 4) for v in cum],
        "stenosis": int(sten.argmax()),
        "stenosis_label": STENOSIS_LABELS[int(sten.argmax())],
        "stenosis_probs": [round(float(v), 4) for v in sten],
        "pathology_probs": pathology,
    }


def run_findings(rgb_seg: np.ndarray, discs: List[Dict[str, Any]],
                 grades: List[int]) -> Dict[str, Any]:
    """Structured findings. Requires the inferior-most three discs."""
    if REG.findings is None:
        return {"error": "findings model unavailable"}
    if len(discs) < 3:
        return {"error": f"needs 3 disc components, found {len(discs)}"}
    crops = []
    for d in discs[-3:]:
        c = crop_region(rgb_seg, d)
        if c.size == 0:
            return {"error": "empty disc crop"}
        c = cv2.resize(c, (FINDINGS_IMG_SIZE, FINDINGS_IMG_SIZE))
        crops.append(torch.from_numpy(c).permute(2, 0, 1).float() / 255.0)
    x = torch.stack(crops, 0).unsqueeze(0).to(DEVICE)
    g = torch.tensor([grades], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = REG.findings(x, g)

    probs = torch.sigmoid(out["binary_logits"][0]).cpu().numpy()
    l4l5 = torch.softmax(out["l4l5_type_logits"][0], 0).cpu().numpy()
    l5s1 = torch.softmax(out["l5s1_type_logits"][0], 0).cpu().numpy()
    sev = torch.softmax(out["severity_logits"][0], 0).cpu().numpy()

    return {
        "binary_findings": [
            {"name": n, "probability": float(probs[i]),
             "positive": bool(probs[i] >= 0.5),
             "pathological": n in PATHOLOGICAL_FINDINGS}
            for i, n in enumerate(BINARY_FINDING_NAMES)
        ],
        "l4_l5_disc_type": DISC_TYPE_LABELS[int(l4l5.argmax())],
        "l4_l5_disc_type_prob": round(float(l4l5.max()), 4),
        "l5_s1_disc_type": DISC_TYPE_LABELS[int(l5s1.argmax())],
        "l5_s1_disc_type_prob": round(float(l5s1.max()), 4),
        "severity": SEVERITY_LABELS[int(sev.argmax())],
        "severity_probs": [round(float(v), 4) for v in sev],
    }
