from pathlib import Path
import json, hashlib, zipfile, xml.etree.ElementTree as ET, subprocess, re, collections
from bs4 import BeautifulSoup
ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
design_check = __import__('runpy').run_path(str(ROOT/'docs/audits/2026-10-01_content_only_restore/verify.py'))['design']
original_designs = json.loads((ROOT/'docs/audits/2026-10-01_content_only_restore/original_designs.json').read_text())
checks = []
def check(name, ok, detail):
    checks.append({'name': name, 'passed': bool(ok), 'evidence': detail})

before = json.loads((WORK/'protected_before.json').read_text())
after = {}
for dirname in ('submission','fyp_submission'):
    for p in sorted((ROOT/dirname).rglob('*')):
        if p.is_file():
            after[str(p.relative_to(ROOT))] = {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mtime_ns':p.stat().st_mtime_ns,'size':p.stat().st_size}
check('both submission trees: identities and timestamps', before == after, f'{len(before)} independently inventoried files; SHA-256, mtime_ns, size, additions/deletions')
(WORK/'protected_after.json').write_text(json.dumps(after,indent=2)+'\n')

baseline = json.loads((WORK/'documents_before.json').read_text())
edited = {e['path'] for e in json.loads((WORK/'edits.json').read_text())}
companions = {'docs/FYP_Agent_Harness_Architecture_Guide_2026-06-09.docx','docs/Codex_Meetup_Agent_Harness_Report.docx','docs/Agentic_AI_Workflow_Report_2026-06-26.docx'}
reports = [x for x in baseline if x['path'].endswith(('.docx','.pdf','.tex')) and x['path'] not in companions]
changed_reports = [x['path'] for x in reports if hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest() != x['sha256'] or (ROOT/x['path']).stat().st_mtime_ns != x['mtime_ns']]
check('excluded reports and research PDFs preserved', not changed_reports, {'files':len(reports),'changed':changed_reports})

# Recompute the primary BH family independently of the rendered registry.
mc = json.loads((ROOT/'results_512/analysis/multiple_comparisons.json').read_text())
ordered = sorted(enumerate(mc['contrasts']),key=lambda item:item[1]['p_value'])
n = len(ordered); qvals={}; running=1.0
for rank in range(n,0,-1):
    index, row=ordered[rank-1]
    running=min(running,row['p_value']*n/rank)
    qvals[index]=running
survivors=[row for i,row in enumerate(mc['contrasts']) if qvals[i]<0.05]
check('current note primary family numbers', len(mc['contrasts'])==20 and len(survivors)==3 and all('asr' not in row['metric'] for row in survivors), {'source':'multiple_comparisons.json','independent_BH_survivors':[(r['pair_id'],r['metric']) for r in survivors]})
eq = json.loads((ROOT/'results_512/analysis/capability_equivalence.json').read_text())['equivalence']
check('post-hoc timing and equivalence denominator', len(eq['contrasts'])==10 and sum(row['equivalent'] for row in eq['contrasts'])==10 and eq['margin_primary']==0.05 and 'post hoc' in eq['margin_declaration'].lower(), 'capability_equivalence.json: contrast list, margin_primary, margin_declaration')
for filename in ('response_quality_jev_validation.json','quant_effect_jev_validation.json'):
    result=json.loads((ROOT/'results_512/analysis'/filename).read_text())
    check('failed pilot retained: '+filename, result['gate_passed'] is False, 'gate_passed=false in current primary artifact')

html_paths = sorted((ROOT/'docs').glob('*.html'))
script_dir=ROOT/'tmp/nonreport_staleness_2026-10-01/js-check';script_dir.mkdir(exist_ok=True)
syntax=[]
for path in html_paths:
    soup=BeautifulSoup(path.read_text(),'html.parser')
    if path.name != 'fyp_style_fingerprints.html':
        rel=str(path.relative_to(ROOT))
        check('original HTML design: '+path.name, not soup.select('#current-evidence-update') and design_check(path)==original_designs[rel], 'Original stylesheet and element attributes match; added overlay absent; current claims stay in existing text.')
    for index,script in enumerate(soup.find_all('script')):
        if script.get('src') or script.get('type','').lower() not in ('','text/javascript','application/javascript','module'): continue
        target=script_dir/(path.stem+f'-{index}'+('.mjs' if script.get('type')=='module' else '.js'))
        target.write_text(script.get_text())
        result=subprocess.run(['node','--check',str(target)],capture_output=True,text=True)
        syntax.append({'file':str(path.relative_to(ROOT)),'script':index,'passed':result.returncode==0,'diagnostic':result.stderr[:500]})
check('standalone HTML JavaScript parses', all(x['passed'] for x in syntax), {'html_files':len(html_paths),'script_blocks':len(syntax),'failures':[x for x in syntax if not x['passed']]})

def package_text(path,prefix):
    with zipfile.ZipFile(path) as z:
        return '\n'.join(' '.join(node.text or '' for node in ET.fromstring(z.read(name)).iter() if node.tag.endswith('}t')) for name in z.namelist() if name.startswith(prefix) and name.endswith('.xml'))
for rel, count in (('docs/fyp_project_explainer_deck.pptx',21),('docs/viva/FYP_Viva_Deck.pptx',17)):
    with zipfile.ZipFile(ROOT/rel) as z:
        slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
        check('slide inventory: '+rel,len(slides)==count,f'{count} slide parts, counted independently of extraction')
    notes=package_text(ROOT/rel,'ppt/notesSlides/notesSlide')
    if 'explainer' in rel:
        text=package_text(ROOT/rel,'ppt/slides/slide')
        check('explainer significance presence and omission', 'Llama’s nominal decrease' in text and 'Every INT8 and NF4 ΔASR non-significant' not in text and 'Anywhere.' not in text, 'produced slide XML: nominal result retained; blanket-null wording absent')
        check('NF4 codebook notes corrected', '16-value codebook' in notes and '4-value codebook' not in notes,'notes XML, presence and omission; QLoRA / official bitsandbytes NF4 map')
    else:
        text=package_text(ROOT/rel,'ppt/slides/slide')
        check('bounded MDE on produced viva slide', 'detection threshold, not an equivalence bound' in text and 'I can rule out large effects' not in text,'produced slide XML, presence and omission')
guide=package_text(ROOT/'docs/FYP_Agent_Harness_Architecture_Guide_2026-06-09.docx','word/document')
check('current Word guide context regenerated', 'June architecture snapshot' in guide and 'repository is now private' in guide and 'PROJECT_LOG records current results and tasks' in guide, 'produced word/document.xml')

for rel in sorted(companions - {'docs/FYP_Agent_Harness_Architecture_Guide_2026-06-09.docx'}):
    text=package_text(ROOT/rel,'word/document')
    check('dated companion: '+rel, '1 October 2026' in text and 'post hoc' in text and 'private' in text and 'historical' in text.lower(), 'produced DOCX: existing paragraphs carry dated current context, historical framing, post-hoc disclosure and repository policy')
archive=ROOT/'docs/archive/Codex_Meetup_Agent_Harness_Report_2026-06-14_snapshot.docx'
original=archive
original_identity=next(x['sha256'] for x in baseline if x['path']=='docs/Codex_Meetup_Agent_Harness_Report.docx')
check('meetup historical source byte preservation', hashlib.sha256(archive.read_bytes()).hexdigest()==original_identity, 'archived source matches original hash from independent baseline inventory')
with zipfile.ZipFile(original) as a, zipfile.ZipFile(ROOT/'docs/Codex_Meetup_Agent_Harness_Report.docx') as b:
    media=[n for n in a.namelist() if n.startswith('word/media/') and not n.endswith('/')]
    check('meetup original diagrams preserved', bool(media) and all(n in b.namelist() and a.read(n)==b.read(n) for n in media), f'{len(media)} original embedded diagram parts compared byte-for-byte')

# The independent baseline inventory supplies the coverage denominator, not the extractor.
coverage=[]
for item in baseline:
    rel=item['path']; suffix=Path(rel).suffix.lower()
    if rel in edited or rel in companions or rel in ('docs/viva/FYP_Viva_Deck.html','docs/viva/FYP_Viva_Deck.pptx','docs/fyp_project_explainer_deck.pptx','docs/FYP_Agent_Harness_Architecture_Guide_2026-06-09.docx'):
        status='updated'; reason='Current context or verified stale wording corrected; source rebuilt where applicable.'
    elif rel.startswith(('docs/archive/','docs/audits/','docs/report_versions/changes/')) or Path(rel).name in ('CLAUDE_CODE_SESSION_ARCHIVE.md','REPORT_CLAIM_AUDIT_v5.md'):
        status='preserved historical'; reason='Dated archive/audit/version-diff record, not a live claim; previous audit bodies not reused.'
    elif suffix in ('.docx','.pdf','.tex'):
        status='excluded report or reference'; reason='User excluded reports; registered variants, drafts and third-party research preserved.'
    elif rel.startswith(('.agents/','.codex/','.claude/','.understand-anything/')):
        status='tool reference'; reason='Agent/tool/design metadata; no functional harness or skill change in this sweep.'
    else:
        status='reviewed, no stale target found'; reason='Staleness scan and current/historical classification; not a full citation, prose, pixel or line-by-line code audit.'
    coverage.append({'path':rel,'status':status,'reason':reason,'sha256':hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()})
(WORK/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
(WORK/'verification.json').write_text(json.dumps({'checks':checks,'syntax':syntax,'coverage_counts':dict(collections.Counter(x['status'] for x in coverage))},indent=2)+'\n')
print(f'{sum(x["passed"] for x in checks)}/{len(checks)} focused checks pass; {len(coverage)} independently inventoried documents; {len(before)} protected files unchanged.')
for c in checks:
    if not c['passed']:print('FAIL',c['name'],c['evidence'])
raise SystemExit(0 if all(c['passed'] for c in checks) else 1)
