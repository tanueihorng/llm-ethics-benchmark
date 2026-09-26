# Source-verification follow-up — 2026-09-08

The seven previously limited methodological source checks now have claim-relevant full-text support. Four supplementary references were added to the report; original entries are retained and renumbered. This closes support gaps for the wording used, not access to every original paper. No claim is made that every mathematical derivation or page was audited.

| Current references | Checked source passages | Supported claim and limits |
|---|---|---|
| [47], [48] | [Efron and Tibshirani (1985), pp. 16 and 23, equations 7.1–7.2](https://link.springer.com/content/pdf/10.2333/bhmk.12.17_1.pdf) | Resample joint observations as units; percentile interval endpoints. Application to matched model responses is the study implementation. The earlier 1993 book download is only an 11-page introductory excerpt. |
| [49], [50] | [Fagerland et al. (2013), p. 3, equations 3–4](https://bmcmedresmethodol.biomedcentral.com/counter/pdf/10.1186/1471-2288-13-91.pdf) | Exact conditional McNemar uses discordant pairs and a binomial null with probability 0.5. Its conservatism is explicitly reported. Authors prefer other variants; citation does not imply they recommend the exact test. |
| [51] | [Benjamini and Hochberg (1995), pp. 291–293, equation 1 and Theorem 1](https://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf) | Original full text: FDR definition and step-up procedure; independence assumption in the theorem. Relevant pages visually read; full proof not audited. |
| [52] | [Cohen (1986), Current Contents Social & Behavioral Sciences, no. 3, p. 18](https://garfield.library.upenn.edu/classics1986/A1986AXF2600001.pdf) | Author retrospective explains chance-corrected agreement and a worked kappa calculation. Corroborates the report definition alongside Artstein and Poesio [56]; not a copy of Cohen (1960). Audit support only, not an added bibliography entry. |
| [53], [54] | [SAS/STAT 15.2 User’s Guide (2020), pp. 7870–7873; p. 7873 formula visually checked](https://documentation.sas.com/api/docsets/statug/15.2/content/power.pdf?locale=en) | Documents Connor normal approximation. Repository MDE uses (z_0.975 + z_0.8)*sqrt(discordance/n), simplifying alternative variance. Report now identifies approximation and unadjusted 0.05 level; does not claim exact power for BH-corrected tests. |
| [57] | [Landis and Koch (1977), p. 165](https://jjcurtin.github.io/book_iaml/pdfs/landis_1977_kappa.pdf) | Original text calls verbal category divisions arbitrary. Supports the report’s caution about universal categories. |
| [59], [60] | [Lakens (2017), pp. 355–357 and 359–360](https://pure.tue.nl/ws/portalfiles/portal/80918653/lakeequi2017.pdf) | Full-text conceptual TOST support: both one-sided tests, 90% interval at alpha .05, justification of bounds. Does not independently validate the study’s binary normal approximation. Post hoc status and that approximation remain explicit in section 4.6. |

## Reproducible evidence

`source-followup.json` records seven valid PDF files, their source URLs, hashes and scope. PDFs are retained locally under `Sources/statistical_methods/` (ignored by git). The repo files `tmp/source_audit_external/ref_46_mcnemar.pdf` and `ref_54_schuirmann.pdf` are HTML, not papers; neither was used as full-text evidence. Original McNemar (1947), Cohen (1960), Connor (1987) and Schuirmann (1987) full texts remain unavailable. Bibliographic identity checks from the earlier audit remain valid.

The report has 61 references and 113 expanded citation occurrences, all referenced and ordered by first use. The independent numerical checker reproduces all 20 primary comparisons and all 140 numeric table cells. Original source DOCX SHA256 remains `38985d6cc293436f6f543a8063aa365cb0c8d3df70eb5c1e46a73de266f38a54`. The updated render has 29 physical pages; all 32 cached TOC page values still match. All pages were visually inspected in contact sheets; no new clipping or overflow was found.

Repository checks: report inventory regenerated with 39 documents, 25 change records and zero problems. Claim-surface gate passed 679/680; its sole failure is the pre-existing T69 unregistered `docs/statistics_explainer.html`, unrelated to this report edit.

## Direct original-source retrieval — later follow-up

Two additional original-source gaps are now closed: [Schuirmann (1987)](https://errorstatistics.com/wp-content/uploads/2014/06/schuirmann.pdf), pp. 658–661, directly supports the report’s TOST explanation; [Efron and Tibshirani (1993)](https://www.cmat.edu.uy/~mordecki/seminario/seminario1.pdf), printed pp. 47, 86–87, 113 and 170–171, directly supports replacement resampling, preserved pairs and percentile intervals. Full PDFs and SHA256 values are recorded in `source-followup.json`. The original book scan contains 451 PDF pages. Earlier excerpt-only and Schuirmann-unavailable statements above describe the previous retrieval stage and are superseded here.

McNemar (1947), Cohen (1960) and Connor (1987) originals remain outstanding. Fresh publisher searches and the repository search did not yield those PDFs. The cited Rhode Island College Cohen link still returns HTML. The NTU library’s documented OpenAthens route reaches a Microsoft sign-in page; an authenticated library session is needed to proceed. Search-result transcriptions are not counted as verified original scans.

The affected report paragraphs and bibliography entries were read directly from v2 again. No discrepancy requiring a report edit was found in the two newly checked originals. DOCX and builder remain unchanged; no report build was needed. This is direct verification of the claims that cite these sources, not a certification of every statement in each publication.
