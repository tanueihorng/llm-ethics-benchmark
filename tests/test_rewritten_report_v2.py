"""Regression checks for the plain-academic v2 rewrite."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "rewritten_fyp_report_v2.docx"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def visible_text() -> str:
    if not REPORT.exists():
        pytest.skip(f"{REPORT} is not present in this checkout")
    with zipfile.ZipFile(REPORT) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in document.findall(".//w:p", NS):
        parts = []
        for node in paragraph.iter():
            if node.tag == f"{{{W_NS}}}t":
                parts.append(node.text or "")
            elif node.tag == f"{{{W_NS}}}tab":
                parts.append("\t")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def test_v2_removes_the_reviewer_and_repository_dialect() -> None:
    text = visible_text().lower()
    forbidden = (
        "harmful-ward",
        "capability anchor",
        "bounded null",
        "validation-informed family",
        "mechanical outcome",
        "report-owned interpretation",
        "judge-primary",
        "survivor set",
        "analysis tree",
    )
    assert not [phrase for phrase in forbidden if phrase in text]


def test_v2_uses_no_em_dash_and_preserves_refreshable_toc() -> None:
    text = visible_text()
    assert "—" not in text
    with zipfile.ZipFile(REPORT) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
        settings = ElementTree.fromstring(archive.read("word/settings.xml"))
    instructions = " ".join(
        node.text or "" for node in document.findall(".//w:instrText", NS)
    )
    assert 'TOC \\o "1-2" \\h \\z \\u' in instructions
    update = settings.find("w:updateFields", NS)
    assert update is not None and update.get(f"{{{W_NS}}}val") == "true"


def test_v2_keeps_core_results_and_plain_conclusion() -> None:
    text = visible_text()
    assert re.search(r"no statistically significant increase in harmful compliance", text, re.I)
    assert "−0.090" in text
    assert "−0.030" in text
    assert "−0.048" in text
    assert "Chapter 7: Conclusion and Future Work" in text


def test_v2_cached_front_matter_matches_the_verified_render() -> None:
    text = visible_text()
    for entry in (
        "Chapter 4: Results\t22",
        "Chapter 5: Scorer Validation and Robustness\t29",
        "Chapter 6: Discussion\t36",
        "Chapter 7: Conclusion and Future Work\t43",
        "References\t45",
        "Appendix C: Human-Validation Boundaries\t50",
        "Figure 1.1. Changes in MMLU accuracy and HarmBench harmful compliance\t2",
        "Table 6.2. Comparison with published quantization results\t37",
    ):
        assert entry in text
