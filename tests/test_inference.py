"""
End-to-end inference on the bundled samples with the released checkpoints.

Skipped when the checkpoints are absent or are Git LFS pointer files.

Run:  python -m pytest tests/test_inference.py -v
"""
import json

import pytest

from spinalyzer import engine

CHECKPOINTS = [engine.CKPT_ROOT / name / "best_model.pth"
               for name in ("segmentation", "grading", "findings")]
SAMPLES = engine.REPO_ROOT / "samples"

pytestmark = pytest.mark.skipif(
    not all(p.exists() and p.stat().st_size > 1_000_000 for p in CHECKPOINTS),
    reason="checkpoints not available (run `git lfs pull`)",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from spinalyzer.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def study(client):
    names = ["sag_T1_0001_S8.png", "axial_T1_0001_D4.png"]
    files = [("files", (n, (SAMPLES / n).read_bytes(), "image/png")) for n in names]
    r = client.post("/api/study", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def test_all_models_load(client):
    health = client.get("/api/health").json()
    assert health["errors"] == {}
    assert all(health["models"].values())


def test_sagittal_segmentation_and_grading(study):
    sag = next(s for s in study["series"] if s["plane"] == "sagittal")
    labels = {d["label"] for d in sag["discs"]}
    assert {"L3-L4", "L4-L5", "L5-S1"} <= labels
    graded = {g["level"]: g for g in sag["grading"]}
    assert set(graded) == {"L3-L4", "L4-L5", "L5-S1"}
    for g in graded.values():
        assert 0 <= g["pfirrmann_grade"] <= 4
        assert sum(g["stenosis_probs"]) == pytest.approx(1.0, abs=1e-3)
    assert len(sag["findings"]["binary_findings"]) == len(engine.BINARY_FINDING_NAMES)


def test_axial_level_from_filename(study):
    ax = next(s for s in study["series"] if s["plane"] == "axial")
    assert ax["level"] == "L4-L5"
    assert ax["level_source"] == "filename"


def test_commands(client, study):
    sid = study["study_id"]
    for text in ("grade L4-L5", "measure L5-S1", "locate stenosis", "segment L4"):
        r = client.post(f"/api/study/{sid}/ask", data={"text": text})
        assert r.status_code == 200, (text, r.text)
        assert r.json()["text"]


def test_report_pdf(client, study):
    r = client.post(f"/api/study/{study['study_id']}/report.pdf",
                    data={"options": json.dumps({})})
    assert r.status_code == 200
    assert r.content.startswith(b"%PDF")
