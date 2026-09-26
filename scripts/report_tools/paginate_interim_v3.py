"""Write TOC page numbers for interim progress v3 from a LibreOffice render.

Usage: python scripts/report_tools/paginate_interim_v3.py  (then rebuild v3)
Word remains the pagination authority (D62); this gives the repeatable estimate.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[2]
DOCX = ROOT / 'docs/rewritten_fyp_interim_progress_v3.docx'
OUT = ROOT / 'scripts/report_tools/interim_progress_v3_pagination.json'


def main():
    d = docx.Document(DOCX)
    toc = [p.text.split('\t')[0] for p in d.paragraphs[24:80] if '\t' in p.text]
    toc = toc[:toc.index('References') + 1]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', tmp, str(DOCX)],
                       check=True, capture_output=True)
        pdf = Path(tmp) / (DOCX.stem + '.pdf')
        pages = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], check=True,
                               capture_output=True, text=True).stdout.split('\f')
    body = [i for i, p in enumerate(pages) if 'Table of Contents' not in p]
    entries = []
    for title in toc:
        hits = [i for i in body if any(line.strip() == title for line in pages[i].splitlines())]
        if not hits:
            sys.exit(f'heading not found in render: {title}')
        page = hits[-1] if title == 'References' else hits[0]
        printed = [line.strip() for line in pages[page].splitlines() if re.fullmatch(r'\d+', line.strip())]
        entries.append({'title': title, 'page': int(printed[-1])})
    OUT.write_text(json.dumps(entries, indent=1) + '\n')
    print(f'Wrote {len(entries)} entries to {OUT}')


if __name__ == '__main__':
    main()
