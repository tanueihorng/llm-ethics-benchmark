"""Regression checks for the plain-academic v3 rewrite."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "rewritten_fyp_report_v3.docx"
REPORT_V1 = REPO_ROOT / "docs" / "rewritten_fyp_report_v1.docx"
REPORT_V2 = REPO_ROOT / "docs" / "rewritten_fyp_report_v2.docx"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def visible_text(path: Path = REPORT) -> str:
    if not path.exists():
        pytest.skip(f"{path} is not present in this checkout")
    with zipfile.ZipFile(path) as archive:
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


def test_v3_removes_the_reviewer_and_repository_dialect() -> None:
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


def test_v3_uses_no_em_dash_and_preserves_refreshable_toc() -> None:
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


def test_v3_keeps_core_results_and_plain_conclusion() -> None:
    text = visible_text()
    assert re.search(r"no statistically significant increase in harmful compliance", text, re.I)
    assert "−0.090" in text
    assert "−0.030" in text
    assert "−0.048" in text
    assert "Chapter 7: Conclusion and Future Work" in text


def test_v3_cached_front_matter_matches_the_verified_render() -> None:
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


def test_v3_citations_and_numbers_match_v1_v2() -> None:
    text_v3 = visible_text(REPORT)
    text_v2 = visible_text(REPORT_V2)

    citations_v3 = re.findall(r"\[(?:\d+)(?:\]-\[\d+)?\]", text_v3)
    citations_v2 = re.findall(r"\[(?:\d+)(?:\]-\[\d+)?\]", text_v2)
    assert citations_v3 == citations_v2, "Citation marker sequence diverged between v3 and v2"

    text_no_cit_v3 = re.sub(r"\[(?:\d+)(?:\]-\[\d+)?\]", "", text_v3)
    text_no_cit_v2 = re.sub(r"\[(?:\d+)(?:\]-\[\d+)?\]", "", text_v2)

    numbers_v3 = sorted(re.findall(r"(?<![A-Za-z])[-+−]?(?:\d{2,}(?:\.\d+)?|\d+\.\d+)%?", text_no_cit_v3))
    numbers_v2 = sorted(re.findall(r"(?<![A-Za-z])[-+−]?(?:\d{2,}(?:\.\d+)?|\d+\.\d+)%?", text_no_cit_v2))
    assert numbers_v3 == numbers_v2, "Numerical token inventory diverged between v3 and v2"
