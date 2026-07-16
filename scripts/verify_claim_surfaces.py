#!/usr/bin/env python3
"""Verify claim-registry freshness, surface coverage, and per-file semantics."""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import yaml

from claim_registry import ROOT, build_registry, registry_is_fresh
from sync_claim_surfaces import sync_surfaces


MANIFEST = ROOT / "configs/claim_surfaces.yaml"
PAIR_MARKER = re.compile(
    r"/\* CLAIM_REGISTRY:PAIRS ([0-9a-f]+) \*/\nconst PAIRS = (\[.*?\]);\n"
    r"/\* END_CLAIM_REGISTRY:PAIRS \*/",
    re.DOTALL,
)
DEFENSE_MARKER = re.compile(
    r"<!-- CLAIM_REGISTRY:DEFENSE_ASR ([0-9a-f]+) -->\n(.*?)\n"
    r"<!-- END_CLAIM_REGISTRY:DEFENSE_ASR -->",
    re.DOTALL,
)
VOLATILE_PATTERNS = (
    re.compile(r"\b\d+\s+(?:automated\s+)?tests?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:claim\s+(?:lock|gate)|verify-claims|machine\s+checks?)[^\n]{0,40}\d+/\d+\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+/\d+\s+(?:claim\s+)?checks?\b", re.IGNORECASE),
    re.compile(
        r"data-count=[\"']\d+[\"'][^\n]{0,160}(?:automated\s+tests?|claim\s+checks?|machine\s+checks?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:ahead|behind)\s+(?:of\s+)?(?:origin/)?main\s+by\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:main|origin/main|HEAD)\s+(?:is\s+)?(?:at|on)\s+[0-9a-f]{7,40}\b", re.IGNORECASE),
)


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def discover_paths(manifest: dict[str, Any], root: Path = ROOT) -> set[str]:
    found: set[str] = set()
    for pattern in manifest["discovery"]:
        for path in root.glob(pattern):
            if path.is_file():
                found.add(path.relative_to(root).as_posix())
    return found


def unregistered_surfaces(manifest: dict[str, Any], root: Path = ROOT) -> set[str]:
    registered = {row["path"] for row in manifest["surfaces"]}
    return discover_paths(manifest, root) - registered


def find_volatile_claims(text: str) -> list[str]:
    hits: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        # Dated revision logs and audit transcripts are immutable history, not
        # live status claims. Current-facing prose must omit the date escape.
        if re.match(r'^[\s>|*\[\"()\-]*20\d{2}-\d{2}-\d{2}\b', line):
            continue
        probe = re.sub(r"[*`_]", "", line)
        for pattern in VOLATILE_PATTERNS:
            if pattern.search(probe):
                hits.append(f"line {line_number}: {line.strip()[:140]}")
                break
    return hits


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        paragraphs.append("".join(paragraph.itertext()))
    return "\n".join(paragraphs)


def _docx_units(path: Path) -> list[str]:
    """Rendered claim units in body order: paragraphs, plus each table row
    joined into one unit so a value stays bound to its row's model/metric
    cells (Word stores every cell as a separate paragraph)."""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    body = root.find("w:body", namespace)
    units: list[str] = []
    for child in body:
        tag = child.tag.rsplit("}", 1)[1]
        if tag == "p":
            text = "".join(child.itertext()).strip()
            if text:
                units.append(text)
        elif tag == "tbl":
            for row in child.findall(".//w:tr", namespace):
                cells = ["".join(cell.itertext()).strip() for cell in row.findall("w:tc", namespace)]
                units.append(" | ".join(cells))
    return units


_LATEX_COMMENT = re.compile(r"(?<!\\)%.*")
_LATEX_TABULAR = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)


def _latex_units(text: str) -> list[str]:
    """Comment-stripped LaTeX units: each tabular ROW is its own unit (so a
    model name in one row can never bind to a value in another row), and the
    remaining prose splits at blank lines."""
    lines = []
    for line in text.splitlines():
        line = _LATEX_COMMENT.sub("", line)
        # Sectioning commands are their own units — otherwise a heading
        # merges into the following paragraph and its words (e.g. "what
        # survives") bind to that paragraph's values.
        if re.match(r"\s*\\(chapter|(sub)*section)\*?\b", line):
            lines.extend(["", line, ""])
        else:
            lines.append(line)
    text = "\n".join(lines)
    units: list[str] = []
    cursor = 0
    for match in _LATEX_TABULAR.finditer(text):
        prose = text[cursor:match.start()]
        units.extend(block.strip() for block in re.split(r"\n\s*\n", prose) if block.strip())
        for row in match.group(0).split(r"\\"):
            row = row.strip()
            if row:
                units.append(row)
        cursor = match.end()
    tail = text[cursor:]
    units.extend(block.strip() for block in re.split(r"\n\s*\n", tail) if block.strip())
    return units


def _text_units(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _surface_units(path: Path) -> list[str]:
    if path.suffix == ".docx":
        return _docx_units(path)
    if path.suffix == ".tex":
        return _latex_units(path.read_text(encoding="utf-8"))
    return _text_units(path.read_text(encoding="utf-8"))


def _surface_text(path: Path) -> str:
    if path.suffix == ".docx":
        return _docx_text(path)
    return path.read_text(encoding="utf-8")


def validate_deck_pairs(text: str, registry: dict[str, Any]) -> tuple[bool, str]:
    match = PAIR_MARKER.search(text)
    if not match:
        return False, "missing generated PAIRS marker"
    if match.group(1) != registry["registry_fingerprint"]:
        return False, "PAIRS marker fingerprint is stale"
    actual = json.loads(match.group(2))
    expected = registry["render"]["deck_pairs"]
    if len(actual) != len(expected):
        return False, f"expected {len(expected)} pairs, found {len(actual)}"
    for got, want in zip(actual, expected, strict=True):
        for key, value in want.items():
            if got.get(key) != value:
                return False, f"{want['id']}.{key}: {got.get(key)!r} != {value!r}"
    return True, f"{len(actual)} registry-derived pair records"


def validate_defense(text: str, registry: dict[str, Any]) -> tuple[bool, str]:
    match = DEFENSE_MARKER.search(text)
    if not match:
        return False, "missing generated defense-ASR marker"
    if match.group(1) != registry["registry_fingerprint"]:
        return False, "defense-ASR marker fingerprint is stale"
    block = match.group(2)
    for row in registry["render"]["defense_asr_rows"]:
        if row["display_name"] not in block:
            return False, f"missing defense row for {row['pair_id']}"
        for value in (row["p_value"], row["bh_q_value"]):
            rendered = "1.000" if value == 1.0 else f"{value:.3f}".lstrip("0")
            if rendered not in block:
                return False, f"missing {row['pair_id']} value {rendered}"
    return True, "all defense rows match registry p/q values"


def _validate_markdown(text: str, registry: dict[str, Any]) -> tuple[bool, str]:
    comparable = text.replace("*", "").replace("`", "")
    rows = {
        row[0]: row for row in registry["render"]["report_table_6_1"]
        if row[1] == "HarmBench ASR (judge)"
    }
    for pair_id, row in rows.items():
        lines = [line for line in comparable.splitlines() if pair_id in line]
        if not lines:
            return False, f"missing row for {pair_id}"
        if not any(all(token in line for token in (row[2], row[3], row[4])) for line in lines):
            return False, f"{pair_id} row is not registry-derived ({row[2]}->{row[3]} {row[4]})"
    return True, "five per-pair ASR rows match registry"


# Retired point estimates that may only appear with their era made explicit.
# CI bounds are exempt (bracketed intervals are stripped before matching):
# "[-0.055, +0.055]" is a legitimate 512-era interval, whereas the bare
# 128-era value "+0.055" is the retired Qwen headline (truncation artefact,
# D41) and must carry its 128-token scope wherever it appears live.
RETIRED_SCOPED_TOKENS = {"+0.055": "128"}
_BRACKETED = re.compile(r"\[[^\][]{0,80}\]")
_HISTORY_MARKER = "Document Revision History"

_DECREASE_WORDS = ("decrease", "declin", "falls", "fell", "drop", "safety-improving", "benign direction")
_INCREASE_WORDS = ("increase", "rise", "rose", "regression", "worsen")
# Non-replication only. "scorer-robust(ness)" is the name of the PROPERTY the
# independent judge tested, not a caveat — accepting it would accept the
# opposite meaning ("the survivor is scorer-robust").
_CAVEAT_SIGNALS = ("does not replicate", "not replicate", "does not reproduce", "not reproduce",
                   "fails to replicate", "fails to reproduce", "scorer-depend", "scorer-sensitive")

_METRIC_MARKERS = {
    "mmlu_accuracy": ("mmlu",),
    "arc_accuracy": ("arc",),
    "xstest_over_refusal": ("over-refusal", "over refusal", "overrefusal"),
    "harmbench_asr_judge": ("asr", "attack-success", "attack success", "harmful compliance", "harmful-compliance"),
}
# For survival statements, "HarmBench contrast" is an ASR contrast even when
# the token "ASR" is absent ("no HarmBench contrast survives …").
_ASR_SURVIVAL_MARKERS = _METRIC_MARKERS["harmbench_asr_judge"] + ("harmbench",)
_NEGATIONS = ("no ", "none", "not ", "nor ", "never", "neither", "zero ", "n.s.", "does not", "doesn't", "non-significant")

# Splits after ./!/? followed by a plausible sentence opener; decimals like
# "0.59" and closers like "(q = 0.008). The" survive intact. Table-row units
# are treated as single sentences by construction.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'§0-9])")


def _norm(text: str) -> str:
    return (
        text.replace("−", "-")
        .replace("‑", "-")
        .replace(r"\_", "_")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
        .replace(r"\,", "")
    )


def _delta_token(delta: float) -> str:
    return f"{delta:+.3f}".replace("+0.000", "0.000")


def _name_variants(display_name: str, pair_id: str) -> tuple[str, ...]:
    variants = {display_name, display_name.replace("-", " "), pair_id}
    # "Qwen3-1.7B" is also written "Qwen 1.7B"; drop the family version digit.
    versionless = re.sub(r"^([A-Za-z]+)\d+[- ]", r"\1-", display_name)
    variants.update({versionless, versionless.replace("-", " ")})
    # "Llama-3.2-3B" is also written "Llama-3B" (family + size only).
    parts = display_name.split("-")
    if len(parts) > 2:
        short = f"{parts[0]}-{parts[-1]}"
        variants.update({short, short.replace("-", " ")})
    return tuple(variants)


def _binding_unit(units: list[str], *needle_groups: tuple[str, ...]) -> str | None:
    """First unit that contains at least one needle from every group."""
    for unit in units:
        lowered = unit.lower()
        if all(any(n.lower() in lowered for n in group) for group in needle_groups):
            return unit
    return None


def _sentences(units: list[str]) -> list[str]:
    out: list[str] = []
    for unit in units:
        if " | " in unit:  # table row: one claim unit, never split
            out.append(unit)
        else:
            out.extend(_SENT_SPLIT.split(unit))
    return out


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in needles)


def _negated_before(sentence: str, index: int, window: int = 40) -> bool:
    """Negation preceding a match within the SAME CLAUSE. The window is
    truncated at the last clause delimiter, so an unrelated negation in an
    earlier clause ("No MMLU; HarmBench ASR survives") cannot shadow the
    match, while "…robust — no ASR contrast survives" stays negated (the
    negation follows the delimiter)."""
    segment = sentence[max(0, index - window):index]
    # Coordinating conjunctions end the negation's scope too: in "does not
    # depend on sample size and replicates …" the "not" belongs to the first
    # conjunct and must not license the second.
    clause = re.split(r"[;:,.()—|]|\b(?:and|but|while|whereas|yet)\b", segment)[-1]
    return _has_any(clause, _NEGATIONS)


def _validate_primary_claims(
    units: list[str], registry: dict[str, Any]
) -> list[tuple[str, bool, str]]:
    """Semantic rendered-artifact checks with an explicitly stated guarantee.

    Every claim family has a positive half and a contradiction half:
      * POSITIVE existence checks — some live unit asserts the claim with
        its value bound to the right model/metric/direction/caveat, values
        driven by the registry; and
      * `contradiction:` entries — ENUMERATED patterns that assert the
        opposite: a survivor delta attributed to another pair, an un-negated
        "ASR … survives" (negation checked LOCALLY, just before the match),
        hedged or wrong survivor counts (any number word or digit run), a κ
        value attributed to the wrong scorer in either label-value order, a
        competing primary budget, a non-significance or wrong-direction
        qualifier clause-adjacent to a significant delta, and a replication
        claim on the scorer-dependent survivor.

    What this does NOT provide: open-vocabulary contradiction detection.
    A wrong sentence phrased outside the enumerated patterns can coexist
    with the correct one and pass. Known residuals, explicitly accepted:
    attribution phrasings outside the enumerated preposition set ("0.59,
    attributable to the regex"), same-clause identity attribution beyond
    the 80-char window, waived-claim families split across paragraphs, and
    any paraphrase a clause-window regex cannot anchor. The builders'
    snippet/recomputation locks (verify_report_claims) are the
    complementary guard at the source layer, and these documents are
    builder-generated — the realistic threat is drift, not an adversarial
    author. Contradiction entries are never waivable; a waiver is
    invalidated when its claim is satisfied or its family signature
    appears within a live unit (see validate_surface)."""
    normed = [_norm(u) for u in units]
    live: list[str] = []
    for unit in normed:
        if _HISTORY_MARKER in unit:
            break
        live.append(unit)
    sentences = _sentences(live)
    claims = registry["claims"]
    results: list[tuple[str, bool, str]] = []

    # 1. Each BH-FDR survivor appears with its model AND metric AND delta,
    #    and its delta is never clause-adjacent to ANOTHER pair's name in a
    #    survivor-context sentence (right value on the wrong model is the
    #    canonical drift this gate exists for).
    all_pairs = claims["pairs"]
    for survivor in claims["multiplicity"]["survivors"]:
        token = _delta_token(survivor["delta"])
        names = _name_variants(survivor["display_name"], survivor["pair_id"])
        metrics = _METRIC_MARKERS[survivor["metric"]]
        hit = _binding_unit(live, (token,), names, metrics)
        results.append((
            f"survivor:{survivor['pair_id']}:{survivor['metric']}",
            hit is not None,
            f"{token} bound to {survivor['display_name']} + {survivor['metric_label']}"
            if hit else f"no unit binds {token} to {survivor['display_name']} + {survivor['metric_label']}",
        ))
        wrong_names = sorted(
            {v for pid, p in all_pairs.items() if pid != survivor["pair_id"]
             for v in _name_variants(p["name"], pid)},
            key=len, reverse=True,
        )
        wrong_alt = "|".join(re.escape(v) for v in wrong_names)
        etoken = re.escape(token)
        # Opening parens are allowed inside the window ("Mistral-7B MMLU
        # (-0.090)" is exactly how a wrong attribution is written, and the
        # name-first arm catches it); CLOSING parens, commas, semicolons,
        # and cell separators end the clause — that is what keeps the
        # legitimate survivor enumeration ("(Qwen 1.7B, -0.090) and on ARC
        # for the Llama…") from cross-binding to the next list item.
        identity_re = re.compile(
            rf"({wrong_alt})[^),;|]{{0,80}}{etoken}|{etoken}[^),;|]{{0,80}}({wrong_alt})",
            re.IGNORECASE,
        )
        offender = next((s for s in sentences
                         if _has_any(s, ("surviv", "fdr", "benjamini", "multiplicity"))
                         and identity_re.search(s)), None)
        results.append((
            f"contradiction:survivor-identity:{survivor['pair_id']}",
            offender is None,
            f"{token} never attributed to another pair" if offender is None
            else f"survivor delta {token} attributed to the wrong pair: {offender[:120]}",
        ))

    # 2. The survivor count is asserted as exact. Committing formulations:
    #    "exactly three", "only three contrasts", "the three contrasts that
    #    do survive". A bare "three" would also match 3B model names.
    number_words = {2: "two", 3: "three", 4: "four"}
    count = claims["multiplicity"]["survivor_count"]
    word = number_words[count]
    # The exactness phrase must be anchored to the surviving-contrasts noun:
    # "exactly three [primary] contrasts", "exactly three survive" (verb
    # directly follows), or the definite "the three contrasts that survive".
    # "Exactly 3 models were tested" or "three models survive screening"
    # must not satisfy the count via an unrelated noun.
    exact_re = re.compile(
        rf"\bexactly ({count}|{word})( [a-z-]+)? contrasts?\b"
        rf"|\bexactly ({count}|{word}) (contrasts? )?surviv"
        rf"|\bonly ({count}|{word})\b[^.;]{{0,30}}?contrasts?"
        rf"|\bthe ({count}|{word}) contrasts that (do )?survive\b",
        re.IGNORECASE,
    )
    hit = next((s for s in sentences if exact_re.search(s)
                and _has_any(s, ("surviv",))
                and _has_any(s, ("BH", "Benjamini", "FDR", "false-discovery", "false discovery", "multiplicity"))), None)
    results.append(("survivor-count", hit is not None,
                    "exact survivor count asserted with BH-FDR context" if hit else "no exact BH-FDR survivor-count statement"))
    # Contradiction: hedged counts, or any OTHER count — spelled (zero
    # through the hyphenated tens; longest-first so "seventeen" wins over
    # "seven") or a digit run — presented as the surviving-contrasts count.
    number_alt = (
        r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:-\w+)?"
        r"|nineteen|eighteen|seventeen|sixteen|fifteen|fourteen|thirteen|twelve|eleven"
        r"|zero|one|two|three|four|five|six|seven|eight|nine|ten|hundred|\d+"
    )
    # Hedges and wrong counts must anchor to the contrasts noun or the
    # survival verb directly — "at least one benchmark … survives" is not a
    # survivor-count statement.
    hedge_re = re.compile(
        rf"\b(at least|more than|at minimum|upwards of) ({number_alt})( [a-z-]+)? contrasts?\b[^.;]*?surviv"
        rf"|\b(at least|more than|at minimum|upwards of) ({number_alt}) (contrasts? )?surviv"
        rf"|\b(exactly|only) (?!(?:{count}|{word})\b)({number_alt})( [a-z-]+)? contrasts?\b[^.;]*?surviv"
        rf"|\b(exactly|only) (?!(?:{count}|{word})\b)({number_alt}) (contrasts? )?surviv",
        re.IGNORECASE,
    )
    offender = next((s for s in sentences if hedge_re.search(s)), None)
    results.append(("contradiction:survivor-count", offender is None,
                    "no hedged/wrong survivor count" if offender is None
                    else f"hedged or wrong survivor count: {offender[:120]}"))

    # 3. No-ASR-survivor claim: a negated survival statement must exist, the
    #    registry must agree, and no sentence may assert ASR survival of the
    #    BH correction without negation.
    if claims["multiplicity"]["asr_survivor_count"] != 0:
        results.append(("no-asr-survivor", False, "registry itself reports an ASR BH survivor"))
        results.append(("contradiction:asr-survival", False, "registry itself reports an ASR BH survivor"))
    else:
        hit = next((s for s in sentences
                    if _has_any(s, ("no ", "none", "not one"))
                    and _has_any(s, ("surviv",))
                    and _has_any(s, _ASR_SURVIVAL_MARKERS)), None)
        results.append(("no-asr-survivor", hit is not None,
                        "ASR non-survival stated" if hit else "no statement that no ASR contrast survives BH-FDR"))
        # The ASR marker must be clause-adjacent to the survival verb — mere
        # co-occurrence would flag family-definition enumerations ("five
        # pairs × HarmBench ASR, MMLU, ARC … exactly three survive").
        asr_survives_re = re.compile(
            r"(asr|harmbench|attack.success|harmful.compliance)[^,;.()]{0,40}surviv"
            r"|surviv[a-z]*[^,;.()]{0,40}(asr|harmbench|attack.success)",
            re.IGNORECASE,
        )
        # Negated if a negation immediately precedes the match OR sits inside
        # the matched span ("harmful-compliance change is NOT robust — NO ASR
        # contrast survives" matches from the marker with the negation inside).
        offender = next((s for s in sentences for m in asr_survives_re.finditer(s)
                         if not _negated_before(s, m.start())
                         and not _has_any(m.group(0), _NEGATIONS)), None)
        results.append(("contradiction:asr-survival", offender is None,
                        "no un-negated ASR-survival sentence" if offender is None
                        else f"asserts ASR survival without negation: {offender[:120]}"))

    # 4. Every individually-significant ASR contrast appears with its model,
    #    delta, direction language matching the delta's sign, AND a positive
    #    significance signal ("significant" not negated, or "excludes zero").
    for pair_id, pair in claims["pairs"].items():
        asr = pair["asr"]
        if not asr["significant"]:
            continue
        token = _delta_token(asr["delta"])
        names = _name_variants(pair["name"], pair_id)
        direction = _DECREASE_WORDS if asr["delta"] < 0 else _INCREASE_WORDS
        hit = None
        for unit in live:
            lowered = unit.lower()
            if not (token in unit and _has_any(unit, names) and any(d in lowered for d in direction)):
                continue
            designified = re.sub(r"\b(non-|not |not statistically |in)significant", "", lowered)
            if "significant" in designified or "excludes zero" in lowered or "excludes 0" in lowered:
                hit = unit
                break
        results.append((
            f"asr-direction:{pair_id}",
            hit is not None,
            f"significant ΔASR {token} bound to {pair['name']} with correct direction"
            if hit else f"no unit states {pair['name']} ΔASR {token} as a significant "
                        f"{'decrease' if asr['delta'] < 0 else 'increase'}",
        ))
        # Contradiction: a non-significance or wrong-direction qualifier
        # CLAUSE-ADJACENT to this delta (windows stop at , ; and parens —
        # enumeration sentences legitimately carry other pairs' qualifiers
        # at a distance). Post-BH non-survival of this delta is a TRUE
        # statement, so BH-context sentences are exempt from the
        # significance half.
        wrong_direction = _INCREASE_WORDS if asr["delta"] < 0 else _DECREASE_WORDS
        etoken = re.escape(token)
        nonsig_re = re.compile(
            rf"{etoken}[^(),;]{{0,30}}\b(non-significant|not significant|not statistically significant|insignificant)"
            rf"|\b(non-significant|not significant|insignificant)\b[^(),;]{{0,30}}{etoken}",
            re.IGNORECASE,
        )
        wrongdir_re = re.compile(
            rf"{etoken}[^(),;]{{0,40}}({'|'.join(wrong_direction)})"
            rf"|({'|'.join(wrong_direction)})[a-z]*[^(),;]{{0,40}}{etoken}",
            re.IGNORECASE,
        )
        offender = None
        for s in sentences:
            if token not in s or not _has_any(s, names):
                continue
            bh_context = _has_any(s, ("BH", "Benjamini", "FDR", "multiplicity", "correction"))
            if not bh_context and nonsig_re.search(s):
                offender = s
                break
            match = wrongdir_re.search(s)
            if match and not any(d in match.group(0).lower() for d in direction):
                offender = s
                break
        results.append((f"contradiction:asr-direction:{pair_id}", offender is None,
                        "no contradictory significance/direction sentence" if offender is None
                        else f"contradicts the {pair['name']} ΔASR claim: {offender[:120]}"))

    # 5. The over-refusal survivor must carry its scorer-dependence caveat
    #    (independent-judge non-replication) somewhere alongside its delta,
    #    and no sentence may assert the opposite (replication/robustness).
    for survivor in claims["multiplicity"]["survivors"]:
        if survivor["metric"] != "xstest_over_refusal":
            continue
        token = _delta_token(survivor["delta"])
        hit = _binding_unit(live, (token,), _CAVEAT_SIGNALS)
        results.append((
            "or-survivor-caveat",
            hit is not None,
            f"{token} carries the independent-judge caveat" if hit
            else f"over-refusal survivor {token} never appears with its scorer-dependence caveat",
        ))
        replication_re = re.compile(
            r"scorer-robust|replicates under|reproduces under|replicated by|reproduced by"
            r"|confirmed by the independent",
            re.IGNORECASE,
        )
        offender = next((s for s in sentences if token in s
                         for m in replication_re.finditer(s)
                         if not _negated_before(s, m.start())), None)
        results.append(("contradiction:or-survivor-caveat", offender is None,
                        "no replication claim for the over-refusal survivor" if offender is None
                        else f"asserts the survivor replicates: {offender[:120]}"))

    # 6. The primary generation budget is stated as such, and no OTHER budget
    #    is called primary/reference in the same breath.
    budget = str(claims["study"]["generation_budget"])
    budget_re = re.compile(
        rf"{budget}[-\s]token[^.;]*?(reference|primary)|(primary|reference)[^.;]*?{budget}[-\s]token",
        re.IGNORECASE,
    )
    hit = next((s for s in sentences if budget_re.search(s)), None)
    results.append(("primary-budget", hit is not None,
                    f"{budget}-token budget stated as primary/reference" if hit else f"budget {budget} never stated as primary"))
    rival_re = re.compile(
        rf"\b(?!{budget})(\d{{2,4}})[-\s]token (budget|run|study|configuration)?\s*(is|was|as|remains)\s*(the\s+)?(primary|reference budget)",
        re.IGNORECASE,
    )
    offender = next((s for s in sentences if rival_re.search(s)), None)
    results.append(("contradiction:primary-budget", offender is None,
                    "no rival budget claimed as primary" if offender is None
                    else f"rival budget claimed primary: {offender[:120]}"))

    # 7. Human-validation kappas present AND neither value tightly attributed
    #    to the wrong scorer, in EITHER label-value order. Label-first: a
    #    tight "regex … 0.59" / "classifier … 0.11". Value-first: "0.59 for
    #    the regex" — flagged only when the value is not already owned by the
    #    correct label immediately to its left ("classifier κ 0.59 vs regex"
    #    legitimately puts "regex" right after the classifier's value).
    validation = claims["validation"]
    cls_k = f"{validation['classifier_human_kappa']:.2f}"
    rgx_k = f"{validation['regex_human_kappa']:.2f}"
    hit = _binding_unit(live, (cls_k,), (rgx_k,), ("regex", "human"))
    results.append(("human-validation-kappa", hit is not None,
                    f"classifier κ {cls_k} vs regex κ {rgx_k} present" if hit
                    else f"human-validation κ pair {cls_k}/{rgx_k} not bound in any unit"))
    # Label-first: the digit-blocking class means an intervening CORRECT
    # value stops the scan ("regex κ 0.11 … 0.59" cannot cross the 0.11).
    # Value-first: attribution requires the preposition DIRECTLY after the
    # value ("0.59 for/of/by/with the regex") — comparatives insert a word
    # first ("0.59 versus the regex's 0.11", "0.59 compared with the regex",
    # "(κ 0.59) than the regex") and so do not match.
    swap_res = (
        re.compile(rf"regex[^,;.0-9]{{0,25}}{re.escape(cls_k)}(?!\d)", re.IGNORECASE),
        re.compile(rf"(classifier|judge)s?\b[^,;.0-9]{{0,25}}{re.escape(rgx_k)}(?!\d)", re.IGNORECASE),
        re.compile(rf"{re.escape(cls_k)}(?!\d)\s*(\((κ|kappa)\))?\s*\b(for|of|by|from|with)\b\s*(the\s+)?regex", re.IGNORECASE),
        re.compile(rf"{re.escape(rgx_k)}(?!\d)\s*(\((κ|kappa)\))?\s*\b(for|of|by|from|with)\b\s*(the\s+)?(classifier|judge)", re.IGNORECASE),
    )
    offender = next((s for s in sentences if any(p.search(s) for p in swap_res)), None)
    results.append(("contradiction:kappa-ownership", offender is None,
                    "no κ value attributed to the wrong scorer" if offender is None
                    else f"κ ownership swapped: {offender[:120]}"))

    # 8. Retired point estimates only appear with their era made explicit.
    #    Applies to live prose; the revision-history appendix is history.
    for token, scope in RETIRED_SCOPED_TOKENS.items():
        offenders = [
            unit for unit in live
            if token in _BRACKETED.sub("", unit) and scope not in unit
        ]
        results.append((
            f"retired-scope:{token}",
            not offenders,
            f"every live '{token}' is {scope}-scoped (CI bounds exempt)" if not offenders
            else f"'{token}' presented without {scope}-context: {offenders[0][:120]}",
        ))

    return results


# Family signatures: does the SHAPE of a waivable claim appear, regardless of
# whether its values are correct? A waiver excuses absence; if the family is
# present, the waiver would otherwise conceal a present-but-divergent claim
# (e.g. "classifier κ 0.50 vs regex κ 0.20 against human labels" under the
# human-validation waiver). Claims without a signature rely on their
# unwaivable contradiction half instead.
_CLAIM_FAMILY_SIGNATURES = {
    "human-validation-kappa": lambda u: (
        _has_any(u, ("κ", "kappa")) and _has_any(u, ("human",))
        and _has_any(u, ("regex", "classifier", "judge"))
        and re.search(r"0\.\d", u) is not None
    ),
}


def _claim_family_present(claim: str, units: list[str]) -> bool:
    """Signatures are evaluated per UNIT (paragraph/table row), not per
    sentence — splitting the context and the values across sentences of one
    paragraph must not hide the family. Cross-paragraph splits remain a
    documented residual."""
    signature = _CLAIM_FAMILY_SIGNATURES.get(claim)
    if signature is None:
        return False
    normed = [_norm(u) for u in units]
    live: list[str] = []
    for unit in normed:
        if _HISTORY_MARKER in unit:
            break
        live.append(unit)
    return any(signature(u) for u in live)


def _validate_zip(surface: dict[str, Any], root: Path) -> tuple[bool, str]:
    archive_path = root / surface["path"]
    source = (root / surface["source"]).read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        try:
            bundled = archive.read(surface["zip_member"])
        except KeyError:
            return False, f"missing zip member {surface['zip_member']}"
    if bundled != source:
        return False, f"{surface['zip_member']} differs from {surface['source']}"
    return True, f"{surface['zip_member']} byte-matches source"


def validate_surface(
    surface: dict[str, Any], registry: dict[str, Any], root: Path = ROOT
) -> list[tuple[str, bool, str]]:
    path = root / surface["path"]
    results: list[tuple[str, bool, str]] = []
    local_root_absent = surface.get("availability") == "local_optional" and not (root / "fyp_submission").exists()
    if not path.exists():
        if local_root_absent:
            return [("availability", True, "local submission tree absent; optional on clone")]
        return [("availability", False, "required surface is missing")]

    text: str | None = None
    for profile in surface.get("profiles", []):
        if profile == "pdf":
            ok = path.read_bytes().startswith(b"%PDF-")
            results.append((profile, ok, "valid PDF signature" if ok else "invalid PDF signature"))
            continue
        if profile == "zip_matches_tex":
            ok, detail = _validate_zip(surface, root)
            results.append((profile, ok, detail))
            continue
        if profile == "byte_matches_source":
            source = root / surface["source"]
            ok = path.read_bytes() == source.read_bytes()
            results.append((profile, ok, "byte-matches generated source" if ok else f"differs from {surface['source']}"))
            continue
        if profile == "fresh_from_source":
            source = root / surface["source"]
            ok = path.stat().st_mtime_ns >= source.stat().st_mtime_ns
            results.append((profile, ok, "not older than source" if ok else f"older than {surface['source']}"))
            continue
        if text is None:
            text = _surface_text(path)
        if profile == "registry_consumer":
            tokens = ("loadClaimRegistry", "CLAIMS.render")
            missing = [token for token in tokens if token not in text]
            results.append((profile, not missing, "imports and consumes registry" if not missing else f"missing {missing[0]}"))
        elif profile == "volatile_free":
            live_text = text
            if path.suffix == ".docx" and "Appendix G: Document Revision History" in live_text:
                live_text = live_text.split("Appendix G: Document Revision History", 1)[0]
            hits = find_volatile_claims(live_text)
            results.append((profile, not hits, "no volatile counts/Git state" if not hits else hits[0]))
        elif profile == "markdown_primary":
            ok, detail = _validate_markdown(text, registry)
            results.append((profile, ok, detail))
        elif profile in {"docx_primary", "latex_primary"}:
            waived = dict(surface.get("waived_claims", {}))
            surface_units = _surface_units(path)
            claim_results = _validate_primary_claims(surface_units, registry)
            claim_names = {claim for claim, _, _ in claim_results}
            for claim, rationale in waived.items():
                # A waiver may only excuse the ABSENCE of a positive claim.
                # Contradiction rules are never waivable, a waiver must name a
                # real claim and say why, and it goes stale the moment the
                # claim is satisfied (then it must be removed, not carried).
                if claim.startswith("contradiction:"):
                    results.append((f"{profile}.waiver:{claim}", False, "contradiction rules cannot be waived"))
                elif claim not in claim_names:
                    results.append((f"{profile}.waiver:{claim}", False, "waiver names an unknown claim"))
                elif not str(rationale).strip():
                    results.append((f"{profile}.waiver:{claim}", False, "waiver has no rationale"))
            for claim, ok, detail in claim_results:
                if claim in waived and not claim.startswith("contradiction:") and str(waived[claim]).strip():
                    if ok:
                        results.append((f"{profile}.{claim}", False,
                                        "waiver is stale: the claim is now satisfied — remove the waiver"))
                    elif _claim_family_present(claim, surface_units):
                        results.append((f"{profile}.{claim}", False,
                                        "waiver conceals a present-but-divergent claim: the claim family "
                                        "appears without the registry values"))
                    else:
                        results.append((f"{profile}.{claim}", True, f"waived: {str(waived[claim]).split('.')[0]}"))
                else:
                    results.append((f"{profile}.{claim}", ok, detail))
        elif profile == "generated_deck_pairs":
            ok, detail = validate_deck_pairs(text, registry)
            results.append((profile, ok, detail))
        elif profile == "generated_defense_asr":
            ok, detail = validate_defense(text, registry)
            results.append((profile, ok, detail))
        elif profile == "snapshot_banner":
            ok = "DATA SNAPSHOT" in text
            results.append((profile, ok, "snapshot visibly bannered" if ok else "missing DATA SNAPSHOT banner"))
        elif profile == "project_log_current":
            current = text.split("### Historical 2026-07-02 snapshot", 1)[0]
            hits = find_volatile_claims(current)
            results.append((profile, not hits, "current-state prose has no volatile counts/Git state" if not hits else hits[0]))
        else:
            results.append((profile, False, "unknown validation profile"))
    if not surface.get("profiles"):
        results.append(("registered", True, f"classified as {surface['lifecycle']}"))
    return results


def run_checks(root: Path = ROOT, manifest_path: Path = MANIFEST) -> list[tuple[str, str, str]]:
    manifest = load_manifest(manifest_path)
    registry = build_registry(root)
    checks: list[tuple[str, str, str]] = []
    fresh, detail = registry_is_fresh(root)
    checks.append(("PASS" if fresh else "FAIL", "registry:fresh", detail))

    unregistered = sorted(unregistered_surfaces(manifest, root))
    checks.append((
        "PASS" if not unregistered else "FAIL",
        "surfaces:coverage",
        f"{len(manifest['surfaces'])} registered" if not unregistered else f"unregistered: {unregistered[0]}",
    ))

    stale_blocks = sync_surfaces(write=False, root=root)
    checks.append((
        "PASS" if not stale_blocks else "FAIL",
        "surfaces:generated-blocks",
        "all generated blocks current" if not stale_blocks else f"stale: {stale_blocks[0]}",
    ))

    for surface in manifest["surfaces"]:
        for profile, ok, result_detail in validate_surface(surface, registry, root):
            checks.append(("PASS" if ok else "FAIL", f"{surface['id']}:{profile}", result_detail))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    checks = run_checks()
    if not args.quiet:
        width = max(len(name) for _, name, _ in checks)
        for status, name, detail in checks:
            marker = "ok  " if status == "PASS" else "FAIL"
            print(f"{marker}  {name:<{width}}  {detail}")
        failures = sum(status == "FAIL" for status, _, _ in checks)
        print("-" * 60)
        print(f"{len(checks)} surface checks: {len(checks) - failures} pass, {failures} fail")
    return 1 if any(status == "FAIL" for status, _, _ in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
