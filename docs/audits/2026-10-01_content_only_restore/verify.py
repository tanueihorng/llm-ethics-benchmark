"""Verify original designs, retained content corrections and protected files."""
from pathlib import Path
import collections
import hashlib
import io
import json
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
checks = []

def digest(value):
    return hashlib.sha256(value).hexdigest()

def design(path):
    if path.suffix == '.html':
        text = path.read_text()
        soup = BeautifulSoup(text, 'html.parser')
        structure = []
        for node in soup.find_all(True):
            attrs = {k:v for k,v in node.attrs.items()
                     if k not in ('title', 'aria-label')
                     and not (node.name == 'meta' and k == 'content')}
            structure.append((node.name, attrs))
        return {
            'stylesheets': digest('\n'.join(re.findall(r'<style[^>]*>.*?</style>', text, re.S)).encode()),
            'layout_elements_and_attributes': digest(json.dumps(structure, sort_keys=True).encode()),
        }
    prefix = 'word/' if path.suffix == '.docx' else 'ppt/'
    signatures = {}
    with zipfile.ZipFile(path) as package:
        for name in sorted(package.namelist()):
            if not name.startswith(prefix) or name.endswith('/'):
                continue
            content = package.read(name)
            if name.endswith('.xlsx'):
                # A rebuild changes nested creation/modification timestamps.
                # Compare every workbook data/style part, normalizing only
                # those two metadata fields after identifying their diff.
                nested = {}
                with zipfile.ZipFile(io.BytesIO(content)) as workbook:
                    for part in sorted(workbook.namelist()):
                        value = workbook.read(part)
                        if part == 'docProps/core.xml':
                            metadata = ET.fromstring(value)
                            for item in metadata:
                                if item.tag in ('{http://purl.org/dc/terms/}created',
                                                '{http://purl.org/dc/terms/}modified'):
                                    item.text = 'BUILD_TIME'
                            value = ET.tostring(metadata)
                        nested[part] = digest(value)
                content = json.dumps(nested, sort_keys=True).encode()
            elif name.endswith('.xml'):
                tree = ET.fromstring(content)
                for node in tree.iter():
                    if node.tag.endswith('}t'):
                        node.text = 'TEXT'
                content = ET.tostring(tree)
            signatures[name] = digest(content)
    return signatures

def check(name, passed, evidence):
    checks.append({'name':name, 'passed':bool(passed), 'evidence':evidence})

if __name__ == '__main__':
    baseline = json.loads((AUDIT / 'original_designs.json').read_text())
    for rel, original in baseline.items():
        current = design(ROOT / rel)
        changed = sorted(k for k in original.keys() | current.keys()
                         if original.get(k) != current.get(k))
        check('original design: ' + rel, not changed,
              {'changed_design_parts':changed,
               'coverage':'All HTML styles/element attributes; all Word/PPTX part inventories, formatting XML, relationships and embedded media; text values excluded. Embedded chart workbooks compare every part; only their creation/modification timestamps are normalized.'})
        if rel.endswith('.html'):
            check('no added overlay: ' + rel,
                  'current-evidence-update' not in (ROOT / rel).read_text(),
                  'Added details element and its stylesheet absent.')

    protected = json.loads((ROOT / 'docs/audits/2026-10-01_nonreport_staleness/protected_before.json').read_text())
    current = {}
    for dirname in ('submission', 'fyp_submission'):
        for path in sorted((ROOT / dirname).rglob('*')):
            if path.is_file():
                stat = path.stat()
                current[str(path.relative_to(ROOT))] = {
                    'sha256':digest(path.read_bytes()), 'mtime_ns':stat.st_mtime_ns, 'size':stat.st_size}
    check('protected submission folders', protected == current,
          {'files':len(protected), 'coverage':'SHA-256, mtime_ns, size, additions and deletions'})
    (AUDIT / 'protected_after.json').write_text(json.dumps(current, indent=2) + '\n')

    before = json.loads((ROOT / 'docs/audits/2026-10-01_nonreport_staleness/documents_before.json').read_text())
    companions = {'docs/FYP_Agent_Harness_Architecture_Guide_2026-06-09.docx',
                  'docs/Agentic_AI_Workflow_Report_2026-06-26.docx',
                  'docs/Codex_Meetup_Agent_Harness_Report.docx'}
    reports = [r for r in before if r['path'].endswith(('.docx', '.pdf', '.tex'))
               and r['path'] not in companions]
    changed = [r['path'] for r in reports
               if digest((ROOT / r['path']).read_bytes()) != r['sha256']
               or (ROOT / r['path']).stat().st_mtime_ns != r['mtime_ns']]
    check('excluded research reports preserved', not changed,
          {'files':len(reports), 'changed':changed})

    field = (ROOT / 'docs/project_field_guide.html').read_text()
    check('content retained: post-hoc timing', 'analysis was added post hoc' in field
          and 'pre-declared ±5 pp' not in field, 'Existing list item; presence and omission.')
    status = (ROOT / 'docs/fyp_status_2026-07.html').read_text()
    check('content retained: retired INT8 increase', '128-token comparison' in status
          and 'classifier ΔASR is +0.005' in status, 'Existing status-deck text.')
    check('content retained: completed judges', 'Additional judges completed' in status
          and 'An optional third scorer' not in status, 'Existing judge card; presence and omission.')
    for rel in sorted(companions):
        with zipfile.ZipFile(ROOT / rel) as package:
            strings = []
            for name in package.namelist():
                if name.startswith('word/') and name.endswith('.xml'):
                    tree = ET.fromstring(package.read(name))
                    strings.extend(n.text or '' for n in tree.iter() if n.tag.endswith('}t'))
        text = ' '.join(strings)
        check('existing Word text updated: ' + rel,
              'private' in text and ('historical' in text or 'snapshot' in text)
              and ('1 October 2026' in text or 'updated Oct 2026' in text),
              'Current privacy/date and historical framing within existing paragraphs or footer.')

    syntax = []
    temp = ROOT / 'tmp/content_only_restore_2026-10-01/js-check'
    temp.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / 'docs').glob('*.html')):
        soup = BeautifulSoup(path.read_text(), 'html.parser')
        for index, script in enumerate(soup.find_all('script')):
            kind = script.get('type', '').lower()
            if script.get('src') or kind not in ('', 'text/javascript', 'application/javascript', 'module'):
                continue
            target = temp / (path.stem + f'-{index}' + ('.mjs' if kind == 'module' else '.js'))
            target.write_text(script.get_text())
            result = subprocess.run(['node', '--check', str(target)], capture_output=True, text=True)
            syntax.append({'file':str(path.relative_to(ROOT)), 'script':index,
                           'passed':result.returncode == 0, 'diagnostic':result.stderr[:300]})
    check('HTML JavaScript syntax', all(s['passed'] for s in syntax),
          {'script_blocks':len(syntax), 'failures':[s for s in syntax if not s['passed']]})
    output = {'passed':sum(c['passed'] for c in checks), 'total':len(checks), 'checks':checks}
    (AUDIT / 'verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f"{output['passed']}/{output['total']} checks pass; original designs compared for {len(baseline)} artifacts; {len(protected)} protected files.")
    raise SystemExit(0 if output['passed'] == output['total'] else 1)
