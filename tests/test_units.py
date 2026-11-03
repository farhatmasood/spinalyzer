"""
Unit tests that need no GPU and no checkpoint.

Run:  python -m pytest tests/ -v
"""
import numpy as np
import pytest

from spinalyzer import commands, study


# --------------------------------------------------------------- filename read

class TestFilenameMetadata:
    @pytest.mark.parametrize("name,expect", [
        ("sag_T1_0001_S8.png", "sagittal"),
        ("axial_T1_0001_D4.png", "axial"),
        ("SAG_T2.dcm", "sagittal"),
    ])
    def test_plane_from_filename(self, name, expect):
        plane, src = study.detect_plane(name, (512, 256, 3))
        assert plane == expect
        assert src == "filename"

    def test_plane_falls_back_to_shape(self):
        assert study.detect_plane("scan.png", (512, 256, 3)) == ("sagittal", "shape")
        assert study.detect_plane("scan.png", (256, 512, 3)) == ("axial", "shape")

    @pytest.mark.parametrize("name,expect", [
        ("sag_T1_0001.png", "T1"),
        ("sag_T2_0001.png", "T2"),
        ("scan.png", "unspecified"),
    ])
    def test_weighting(self, name, expect):
        assert study.detect_weighting(name, None)[0] == expect

    @pytest.mark.parametrize("name,expect", [
        ("axial_T1_0001_D4.png", "L4-L5"),
        ("ax_D3.png", "L3-L4"),
        ("ax_D5.png", "L5-S1"),
        ("slice_L4-L5.png", "L4-L5"),
        ("plain.png", None),
    ])
    def test_level_token(self, name, expect):
        assert study.detect_level_token(name) == expect


# --------------------------------------------------------------- labelling

class TestLabelling:
    def test_labels_run_bottom_up(self):
        """The lowest component in the image is the most inferior level."""
        comps = [{"centroid_y": 10.0}, {"centroid_y": 50.0}, {"centroid_y": 90.0}]
        study.label_components(comps, study.DISC_ORDER)
        assert comps[2]["label"] == "L5-S1"     # lowest on the image
        assert comps[1]["label"] == "L4-L5"
        assert comps[0]["label"] == "L3-L4"

    def test_more_components_than_names(self):
        comps = [{"centroid_y": float(i)} for i in range(12)]
        study.label_components(comps, study.DISC_ORDER)
        assert all("label" in c for c in comps)


# --------------------------------------------------------------- parser

class TestParser:
    def test_single_body_is_a_vertebra(self):
        c = commands.parse("segment L4")
        assert (c.intent, c.target, c.target_kind) == ("SEGMENT", "L4", "vertebra")

    def test_level_pair_is_a_disc(self):
        c = commands.parse("segment L4-L5")
        assert (c.intent, c.target, c.target_kind) == ("SEGMENT", "L4-L5", "disc")

    def test_disc_wins_over_vertebra(self):
        """L4-L5 contains 'L4' but must not resolve to the L4 body."""
        assert commands.parse("L4-L5").target_kind == "disc"
        assert commands.parse("L4").target_kind == "vertebra"

    @pytest.mark.parametrize("text,intent", [
        ("measure L5-S1", "MEASURE"),
        ("grade L4-L5", "GRADE"),
        ("find stenosis", "FIND"),
        ("compare L3-L4 and L4-L5", "COMPARE"),
        ("describe", "DESCRIBE"),
        ("report", "REPORT"),
        ("help", "HELP"),
    ])
    def test_intents(self, text, intent):
        assert commands.parse(text).intent == intent

    @pytest.mark.parametrize("text", [
        "what is the weather today",
        "who won the world cup",
        "write me a poem",
    ])
    def test_off_topic_is_refused(self, text):
        c = commands.parse(text)
        assert c.intent == "UNKNOWN"
        assert c.in_scope is False

    def test_dcode_maps_to_level(self):
        assert commands.parse("grade D4").target == "L4-L5"

    def test_compare_captures_both(self):
        c = commands.parse("compare L3-L4 and L4-L5")
        assert (c.target, c.second_target) == ("L3-L4", "L4-L5")

    def test_show_structure_is_segment_not_find(self):
        c = commands.parse("show me the spinal cord")
        assert c.intent == "SEGMENT"
        assert c.target == "SC"


# --------------------------------------------------------------- lexicon

class TestLexicon:
    def test_alias_expands(self):
        commands._LEXICON.append({"pattern": "g4", "expansion": "grade L4-L5"})
        try:
            c = commands.parse("g4")
            assert c.intent == "GRADE"
            assert c.target == "L4-L5"
        finally:
            commands._LEXICON.clear()

    def test_untouched_without_alias(self):
        commands._LEXICON.clear()
        assert commands.apply_lexicon("grade L4-L5") == "grade L4-L5"


# --------------------------------------------------------------- scale

class _FakeSeries:
    """Minimal stand-in carrying only what mm_factors needs."""
    def __init__(self, h, w, spacing):
        self.rgb = np.zeros((h, w, 3), dtype=np.uint8)
        self.spacing = spacing
    mm_factors = study.Series.mm_factors


class TestScale:
    def test_no_spacing_gives_no_factor(self):
        assert _FakeSeries(384, 384, None).mm_factors() is None

    def test_square_image_passes_spacing_through(self):
        f = _FakeSeries(384, 384, {"row": 0.6, "col": 0.6}).mm_factors()
        assert f == pytest.approx((0.6, 0.6))

    def test_resize_ratio_is_applied(self):
        """A 768-tall original halves to the 384 canvas, so each canvas row
        covers two original rows."""
        f = _FakeSeries(768, 768, {"row": 0.5, "col": 0.5}).mm_factors()
        assert f == pytest.approx((1.0, 1.0))

    def test_non_square_keeps_axes_separate(self):
        f = _FakeSeries(768, 384, {"row": 0.5, "col": 0.5}).mm_factors()
        assert f == pytest.approx((1.0, 0.5))

    def test_zero_spacing_rejected(self):
        assert _FakeSeries(384, 384, {"row": 0.0, "col": 0.5}).mm_factors() is None
