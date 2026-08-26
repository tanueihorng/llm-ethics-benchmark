# SINGAPURA · A Miniature Singapore

An interactive Three.js miniature of Marina Bay — and a sky-scale model of the
FYP study *safety–capability trade-offs in quantized small language models*.

**The five landmarks are the five matched model pairs** (fp16 baseline vs NF4
4-bit). Every landmark card carries that pair's real measured deltas (ΔASR /
ΔORR / ΔMMLU / ΔARC, judge-primary ASR, Benjamini–Hochberg FDR q=0.05)
transcribed from `results_512/analysis/`, plus its canonical interpretation
label. The **? panel** holds the Chronicle: the landmark↔pair map, the headline
null (no FDR-surviving ASR change anywhere), and the study's honesty caveats
(power floor, scorer dependence). The 12 collectible ✦ probes end with the
conclusion itself.

| Landmark | Pair | Verdict |
|---|---|---|
| Marina Bay Sands | Qwen3-1.7B | broad degradation (MMLU −9.0 pp†, ASR +0.0) |
| Singapore Flyer | Mistral-7B | alignment improvement, unconfirmed (ASR −2.0 pp) |
| The Merlion | Llama-3.2-3B | capability collapse masquerading as safety (ARC −3.2 pp†) |
| Gardens by the Bay | Phi-4-mini | alignment degradation, directional; ORR survivor is scorer-dependent |
| ArtScience Museum | Qwen3-4B | alignment degradation, unconfirmed (+4.0 pp, p=.096) |

† survives Benjamini–Hochberg FDR (q < 0.05). Numbers are read from the primary
artifacts, never invented — see `src/lore.js` for the provenance note.

![stack](https://img.shields.io/badge/three.js-r169-blue) ![tests](https://img.shields.io/badge/tests-e2e%2018%20%C3%97%202-green)

## The city

Marina Bay water at the centre (planar **real-time reflections** of the skyline
on high quality), ringed by the promenade; Marina Bay Sands (three curved,
leaning towers with glass-strip facades and a planted sky-park deck), the
ArtScience Museum lotus, the Singapore Flyer (slowly turning, lit rim),
Gardens by the Bay (six lattice supertrees that glow along their trunks at
night, OCBC skyway, two conservatory domes), the Merlion (with its constant
water jet and splash), the Esplanade durians, the Helix Bridge and a dense
procedural CBD — every building carries a **procedural facade texture** whose
windows individually light up after dark (day map + night emissive map).
Mid-rise districts on street grids and an instanced low-rise suburb ring fill
the city out to the horizon. Gulls by day, an airliner on approach, tugboats
on the bay — and at 23:30 the MBS light show sweeps beams across the sky.
Glass reflects a PMREM environment baked from the atmospheric sky model.

## Run

Double-click `AETHERIA.command` (starts a local server + opens the browser), or:

```bash
cd world3d
npm install
npm run dev          # http://127.0.0.1:5199
```

Production bundle + end-to-end tests (headless Chrome via puppeteer-core):

```bash
npm run build
npm run test:e2e     # builds, serves, runs 18 checks × (prod + dev), saves screenshots/
```

## Controls

| Input | Action |
|---|---|
| drag / scroll / right-drag | orbit / zoom / pan |
| click a landmark (or keys `1`–`5`) | open its card + codex + cinematic fly-to |
| fly close to a glowing ✦ | collect an evaluation probe (12 around the bay) |
| `space` | pause/resume the day cycle |
| time slider | scrub dawn → dusk → night |
| `?` / `h` / `esc` | chronicle & help / hide UI / close panels |

Collect all 12 probes to resolve the evaluation — the toast is the paper's
conclusion.

`?quality=low` forces the lightweight pipeline (pixel ratio 1, no bloom, no
shadows) — used by the e2e suite so software-rendered CI stays fast.

## Layout

```
src/main.js         bootstrap, render loop, quality, debug API (window.__WORLD__)
src/world.js        Marina Bay water, promenade, parks, Fort Canning, roads, trees
src/props.js        MBS, ArtScience Museum, Flyer, Gardens, Merlion, Esplanade,
                    Helix Bridge, CBD skyline, city districts + suburb ring
src/textures.js     procedural canvas textures: facades (day + night emissive),
                    roads, ground, water normals, supertree lattice
src/life.js         gulls, airliner, tugboats, water sparkle
src/lights.js       the MBS night light show (rotating beams)
src/quest.js        12-probe discovery quest (pickup, bursts, toast)
src/lore.js         FYP meaning layer: per-pair codices + the Chronicle
src/sky.js          sky dome shader, stars, sun & moon
src/timeOfDay.js    hour → lighting/colour state
src/interactions.js raycast picking, tooltips, camera flights
src/ui.js           dock, card, help, loader, probe HUD, toast DOM
src/pois.js         landmark definitions (positions + hit spheres)
tests/e2e.mjs       end-to-end suite (rendering, controls, picking, travel,
                    cycle, light-show gating, quest, codex numbers, chronicle)
```

Everything is generated from a fixed seed at load (no network assets), so the
city is identical on every visit and the e2e assertions are deterministic.
Screenshots in `screenshots/` regenerate on every `test:e2e` run (gitignored).
