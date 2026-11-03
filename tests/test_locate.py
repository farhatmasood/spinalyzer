"""
Tests for finding resolution and locate routing. No GPU, no checkpoint.

Run:  python -m pytest tests/ -v
"""
import pytest

from spinalyzer import commands
from spinalyzer import findings as fx


class TestNameResolution:
    @pytest.mark.parametrize("text,expect", [
        ("stenosis", "Spinal Canal Stenosis"),
        ("canal stenosis", "Spinal Canal Stenosis"),
        ("herniation", "Disc Herniation"),
        ("herniated disc", "Disc Herniation"),
        ("bulge", "Disc Bulge"),
        ("protrusion", "Disc Protrusion"),
        ("modic", "Modic Changes"),
        ("schmorl", "Schmorl Nodules"),
        ("listhesis", "Spondylolisthesis"),
        ("facet arthropathy", "Facet Joint Hypertrophy"),
        ("ligamentum flavum", "Ligamentum Flavum Hypertrophy"),
        ("foraminal narrowing", "Neural Foraminal Narrowing"),
    ])
    def test_alias_maps_to_canonical(self, text, expect):
        assert fx.resolve_name(text) == expect

    def test_phrase_inside_a_sentence(self):
        assert fx.resolve_name("where is the disc herniation") == "Disc Herniation"

    def test_longest_phrase_wins(self):
        """'canal stenosis' must not be shadowed by the shorter 'stenosis'."""
        assert fx.resolve_name("neural foraminal stenosis") == "Neural Foraminal Narrowing"

    def test_unknown_returns_none(self):
        assert fx.resolve_name("pneumonia") is None


class TestCatalogue:
    def test_covers_every_binary_finding(self):
        from spinalyzer import engine
        names = {c["name"] for c in fx.catalogue()}
        assert names == set(engine.BINARY_FINDING_NAMES)

    def test_kinds_are_known(self):
        kinds = {c["kind"] for c in fx.catalogue()}
        assert kinds <= {"localisable", "level-tagged", "not segmented", "study-level"}

    def test_level_tagged_detected(self):
        by_name = {c["name"]: c["kind"] for c in fx.catalogue()}
        assert by_name["L4-L5 Involvement"] == "level-tagged"
        assert by_name["L5-S1 Compression"] == "level-tagged"

    def test_unsegmented_flagged(self):
        by_name = {c["name"]: c["kind"] for c in fx.catalogue()}
        assert by_name["Muscle Spasm"] == "not segmented"


class TestGradingHeadCoverage:
    def test_every_grading_head_has_a_display_name(self):
        """The per-disc heads are the only per-level evidence, so each must map
        onto a finding the user can ask for."""
        expected = {"disc_bulge", "disc_protrusion", "disc_extrusion",
                    "nerve_compression", "thecal_compression",
                    "foraminal_narrowing", "facet_hypertrophy",
                    "ligamentum_flavum", "herniation"}
        assert set(fx.GRADING_PATHOLOGY) == expected

    def test_display_names_have_anatomy(self):
        for display in fx.GRADING_PATHOLOGY.values():
            assert display in fx.ANATOMY, f"{display} has no anatomical anchor"


class TestLocateRouting:
    @pytest.mark.parametrize("text", [
        "locate stenosis",
        "where is the herniation",
        "where's the bulge",
        "show spondylolisthesis",
        "point out disc bulge",
        "pinpoint modic changes",
        "mark schmorl nodules",
    ])
    def test_phrasings_route_to_locate(self, text):
        assert commands.parse(text).intent == "LOCATE"

    def test_anatomy_target_stays_segment(self):
        """'locate L4-L5' names anatomy, not a finding."""
        c = commands.parse("locate L4-L5")
        assert c.intent == "SEGMENT"
        assert c.target == "L4-L5"

    def test_show_structure_stays_segment(self):
        assert commands.parse("show the spinal cord").intent == "SEGMENT"

    def test_find_remains_screening(self):
        assert commands.parse("find stenosis").intent == "FIND"

    def test_locate_carries_the_pathology(self):
        c = commands.parse("locate disc bulge")
        assert c.pathology == "Disc Bulge"

    def test_off_topic_still_refused(self):
        assert commands.parse("where is my car").intent == "UNKNOWN"


class TestAnatomyMap:
    def test_disc_findings_anchor_to_the_disc(self):
        for n in ("Disc Bulge", "Disc Protrusion", "Disc Extrusion",
                  "Disc Herniation", "Desiccated Disc", "Annular Tear"):
            assert "disc" in fx.ANATOMY[n]["sag"]

    def test_canal_findings_anchor_to_the_cord(self):
        for n in ("Spinal Canal Stenosis", "Thecal Sac Compression"):
            assert "SC" in fx.ANATOMY[n]["sag"]
            assert fx.ANATOMY[n].get("band") is True

    def test_posterior_findings_are_axial_only(self):
        for n in ("Ligamentum Flavum Hypertrophy", "Facet Joint Hypertrophy"):
            assert fx.ANATOMY[n]["sag"] == []
            assert "PE" in fx.ANATOMY[n]["ax"]

    def test_vertebral_findings_use_adjacent_bodies(self):
        for n in ("Modic Changes", "Schmorl Nodules", "Compression Fracture",
                  "Hemangioma", "Spondylolisthesis"):
            assert "adjacent_vertebrae" in fx.ANATOMY[n]["sag"]

    def test_unsegmented_carry_a_reason(self):
        for n, why in fx.NO_STRUCTURE.items():
            assert why and isinstance(why, str)
