// End-to-end suite for AETHERIA.
// Runs the real page in headless Chrome and exercises rendering, camera
// controls, landmark picking, travel flights and the day/night cycle.
//
//   node tests/e2e.mjs [--dev | --prod | --url http://...] [--keep-server]
//
// Default: production build + `vite preview`, then the dev server.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SHOTS = path.join(ROOT, 'screenshots');
mkdirSync(SHOTS, { recursive: true });

const args = process.argv.slice(2);
const MODE = args.includes('--dev') ? 'dev' : args.includes('--prod') ? 'prod' : 'both';
const URL_OVERRIDE = args.includes('--url') ? args[args.indexOf('--url') + 1] : null;

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
  ].filter(Boolean);
  for (const c of candidates) if (existsSync(c)) return c;
  throw new Error('No Chrome/Chromium/Edge/Brave found; set CHROME_PATH');
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForPort(port, timeoutMs = 30000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const net = await import('node:net');
      await new Promise((res, rej) => {
        const s = net.default.connect({ port, host: '127.0.0.1' }, () => { s.destroy(); res(); });
        s.on('error', rej);
        s.setTimeout(1200, () => { s.destroy(); rej(new Error('timeout')); });
      });
      return;
    } catch {
      await sleep(250);
    }
  }
  throw new Error(`server on :${port} did not start`);
}

// ── server management ──────────────────────────────────────────────────
const VITE_BIN = path.join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js');

function spawnServer(viteArgs) {
  const child = spawn(process.execPath, [VITE_BIN, ...viteArgs], { cwd: ROOT });
  child.stdout.on('data', () => {});
  child.stderr.on('data', (d) => process.env.E2E_VERBOSE && process.stderr.write(d));
  return child;
}

async function buildProd() {
  console.log('  building production bundle…');
  await new Promise((res, rej) => {
    const p = spawn(process.execPath, [VITE_BIN, 'build'], { cwd: ROOT });
    let out = '';
    p.stdout.on('data', (d) => { out += d; });
    p.stderr.on('data', (d) => { out += d; });
    p.on('exit', (code) => (code === 0 ? res() : rej(new Error(`vite build failed:\n${out.slice(-2000)}`))));
  });
}

// ── the actual checks ──────────────────────────────────────────────────
async function runSuite(browser, baseUrl) {
  const results = [];
  const add = (name, pass, detail = '') => {
    results.push({ name, pass, detail });
    console.log(`   ${pass ? '✓' : '✗'} ${name}${detail ? ` — ${detail}` : ''}`);
  };

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });

  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => consoleErrors.push(String(err)));

  const url = `${baseUrl}/?quality=low&cycle=0&hour=9&intro=0`;
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });

  // 1 · world becomes ready
  try {
    await page.waitForFunction(() => window.__WORLD__?.ready === true, { timeout: 30000 });
    add('world boots and reports ready', true);
  } catch {
    add('world boots and reports ready', false, 'no ready flag within 30s');
    return results;
  }

  // 1b · controls settle (no flight active) before interaction tests
  try {
    await page.waitForFunction(() => window.__WORLD__.isFlying() === false, { timeout: 15000 });
    add('controls settle before interaction', true);
  } catch {
    add('controls settle before interaction', false, 'a flight was still active');
  }

  // 2 · canvas renders non-blank frames
  const stats = await page.evaluate(() => window.__WORLD__.captureStats());
  add('framebuffer has real imagery',
    stats.distinct > 60 && stats.meanLum > 4 && stats.blackRatio < 0.85,
    `distinct=${stats.distinct} meanLum=${stats.meanLum.toFixed(1)} black=${(stats.blackRatio * 100).toFixed(0)}%`);

  // 3 · orbit drag rotates the camera
  const azBefore = await page.evaluate(() => window.__WORLD__.controls.getAzimuthalAngle());
  await page.mouse.move(720, 450);
  await page.mouse.down();
  await page.mouse.move(600, 470, { steps: 12 });
  await page.mouse.up();
  await sleep(400);
  const azAfter = await page.evaluate(() => window.__WORLD__.controls.getAzimuthalAngle());
  add('drag orbits the camera', Math.abs(azAfter - azBefore) > 0.05,
    `Δazimuth=${(azAfter - azBefore).toFixed(3)}`);

  // 4 · wheel zooms
  const distBefore = await page.evaluate(() => {
    const c = window.__WORLD__.camera.position;
    const t = window.__WORLD__.controls.target;
    return c.distanceTo(t);
  });
  await page.mouse.move(720, 450);
  await page.mouse.wheel({ deltaY: -480 });
  await sleep(500);
  const distAfter = await page.evaluate(() => {
    const c = window.__WORLD__.camera.position;
    const t = window.__WORLD__.controls.target;
    return c.distanceTo(t);
  });
  add('scroll zooms the camera', distAfter < distBefore - 1,
    `${distBefore.toFixed(1)} → ${distAfter.toFixed(1)}`);

  // reset to the default vantage so the pick tests are deterministic
  await page.evaluate(() => {
    const w = window.__WORLD__;
    w.controls.target.set(0, 30, 40);
    w.camera.position.set(230, 160, 540);
    w.controls.update();
  });
  await sleep(300);

  // 5 · hover tooltip on a projected landmark
  const proj = await page.evaluate(() => {
    const w = window.__WORLD__;
    const v = w.pois[2].pos.clone().project(w.camera);   // the Merlion
    if (v.z > 1) return null;
    return { x: (v.x * 0.5 + 0.5) * innerWidth, y: (-v.y * 0.5 + 0.5) * innerHeight };
  });
  if (proj) {
    await page.mouse.move(proj.x, proj.y);
    await sleep(300);
    const tipVisible = await page.evaluate(() =>
      !document.getElementById('tooltip').classList.contains('hidden')
      && document.getElementById('tt-name').textContent.length > 0);
    add('hovering a landmark shows its tooltip', tipVisible);
    // 6 · click opens card + flies
    await page.mouse.click(proj.x, proj.y);
    const opened = await page.waitForFunction(
      () => window.__WORLD__.cardOpen() === true
        && document.getElementById('card-title').textContent.includes('Merlion'),
      { timeout: 5000 },
    ).then(() => true).catch(() => false);
    add('clicking a landmark opens its card', opened);
    const arrived = await page.waitForFunction(() => window.__WORLD__.isFlying() === false, { timeout: 8000 })
      .then(() => true).catch(() => false);
    const nearTarget = await page.evaluate(() => {
      const w = window.__WORLD__;
      const poi = w.pois[2];
      return w.camera.position.distanceTo(poi.pos);
    });
    add('travel flight lands near the landmark',
      arrived && nearTarget < 100, `distance≈${nearTarget.toFixed(1)}`);
  } else {
    add('hovering a landmark shows its tooltip', false, 'landmark behind camera — reposition first');
  }

  // 7 · keyboard travel to another landmark
  await page.keyboard.press('Escape');
  await page.keyboard.press('4');   // Gardens by the Bay
  const card4 = await page.waitForFunction(() =>
    document.getElementById('card-title').textContent.includes('Gardens'), { timeout: 5000 })
    .then(() => true).catch(() => false);
  add('keyboard 1–5 travels to a landmark', card4);

  // 8 · time slider drives the clock readout
  await page.evaluate(() => {
    const s = document.getElementById('time-slider');
    s.value = '13.4';
    s.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await sleep(200);
  const readout = await page.evaluate(() => document.getElementById('time-readout').textContent);
  add('time slider updates clock readout', /13:/.test(readout) && /Sun|High/.test(readout), readout);

  // 9 · day vs night environment really changes
  const envDay = await page.evaluate(() => {
    const w = window.__WORLD__;
    w.setHour(12);
    return { sun: w.renderer.shadowMap.enabled ? null : null, hour: w.getHour() };
  });
  void envDay;
  const dayStats = await page.evaluate(() => {
    const w = window.__WORLD__;
    w.setHour(12);
    return new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(() => res(w.captureStats()))));
  });
  const nightStats = await page.evaluate(() => {
    const w = window.__WORLD__;
    w.setHour(0);
    return new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(() => res(w.captureStats()))));
  });
  add('day/night cycle changes the rendered scene',
    Math.abs(dayStats.meanLum - nightStats.meanLum) > 3 || Math.abs(dayStats.distinct - nightStats.distinct) > 15,
    `day lum=${dayStats.meanLum.toFixed(1)} vs night lum=${nightStats.meanLum.toFixed(1)}`);

  // 10 · screenshots (day / golden hour / night) — full page, UI included
  const shots = [
    ['day-noon', 12, null],
    ['golden-hour', 17.0, 1],      // the Flyer, sun still up
    ['night', 23.5, null],         // custom panorama: the whole lit waterfront
  ];
  for (const [name, hr, fly] of shots) {
    await page.evaluate((hour, f) => {
      const w = window.__WORLD__;
      w.setHour(hour);
      if (f !== null) { w.openCard(f); w.flyTo(f); }
      return new Promise((res) => setTimeout(res, f !== null ? 2600 : 400));
    }, hr, fly);
    if (name === 'night') {
      // postcard vantage: SE high over the water, whole skyline in frame
      await page.evaluate(() => {
        const w = window.__WORLD__;
        w.closeCard();
        w.camera.position.set(470, 210, -450);
        w.controls.target.set(-40, 60, 110);
        w.controls.update();
      });
      await sleep(600);
    }
    await page.screenshot({ path: path.join(SHOTS, `${name}.png`) });
  }
  add('screenshots captured', true, 'screenshots/day-noon.png, golden-hour.png, night.png');

  // 11 · the light show plays only after dark
  const showDay = await page.evaluate(() => {
    const w = window.__WORLD__;
    w.setHour(12);
    return new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(() => res(w.lightShowIntensity()))));
  });
  const showNight = await page.evaluate(() => {
    const w = window.__WORLD__;
    w.setHour(23.5);
    return new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(() => res(w.lightShowIntensity()))));
  });
  add('night light show plays only after dark',
    showDay < 0.05 && showNight > 0.3,
    `day=${showDay.toFixed(2)} night=${showNight.toFixed(2)}`);

  // 12 · sky-shard quest: fly to every shard, counter fills, toast appears
  const total = await page.evaluate(() => window.__WORLD__.quest.total);
  const positions = await page.evaluate(() => window.__WORLD__.quest.positions);
  for (const p of positions) {
    await page.evaluate((pos) => {
      const w = window.__WORLD__;
      w.camera.position.set(pos[0] + 0.6, pos[1] + 0.6, pos[2] + 0.6);
    }, p);
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  }
  await sleep(300);
  const collected = await page.evaluate(() => window.__WORLD__.quest.collected());
  const toastShown = await page.evaluate(() =>
    !document.getElementById('toast').classList.contains('hidden'));
  const hudCount = await page.evaluate(() => document.getElementById('shard-count').textContent);
  add('sky-shard quest completes with toast',
    total === 12 && collected === total && toastShown && hudCount === String(total),
    `${collected}/${total} collected, toast=${toastShown}`);
  const toastText = await page.evaluate(() => document.getElementById('toast').textContent);
  add('quest toast carries the study conclusion', toastText.includes('null holds'), JSON.stringify(toastText.slice(0, 60)));

  // 13 · landmark cards carry the FYP codex (real measured deltas)
  const cx = await page.evaluate(() => {
    const w = window.__WORLD__;
    w.openCard(0);
    const el = document.getElementById('card-codex');
    return { text: el.textContent, visible: el.style.display !== 'none' };
  });
  const cxOk = cx.visible
    && cx.text.includes('QWEN3-1.7B')
    && cx.text.includes('ΔASR')
    && cx.text.includes('0.0 pp')
    && cx.text.includes('−9.0 pp')
    && cx.text.includes('†')
    && cx.text.includes('Broad Degradation');
  add('landmark card shows the pair codex (real deltas)', cxOk,
    cx.text.replace(/\s+/g, ' ').slice(0, 90));
  await page.evaluate(() => window.__WORLD__.closeCard());

  // 14 · the chronicle explains the landmark↔pair mapping and headline finding
  const ch = await page.evaluate(() => {
    document.getElementById('btn-help').click();
    return document.getElementById('chronicle').textContent;
  });
  const chOk = ch.includes('Chronicle of the Quantized City')
    && ch.includes('no harmful-compliance (ASR) change survives FDR')
    && ch.includes('Marina Bay Sands') && ch.includes('Mistral-7B')
    && ch.includes('results_512/analysis/multiple_comparisons.json');
  add('chronicle maps landmarks to model pairs', chOk);
  await page.evaluate(() => document.getElementById('help-close').click());

  // 11 · performance floor (software GL is slow; this catches catastrophic jank)
  const perf = await page.evaluate(() => new Promise((res) => {
    let frames = 0;
    const t0 = performance.now();
    const tick = () => {
      frames++;
      if (frames >= 60) res({ fps: (frames / ((performance.now() - t0) / 1000)).toFixed(1) });
      else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }));
  add('performance floor (>3.5 fps under SwiftShader)', Number(perf.fps) > 3.5, `${perf.fps} fps`);

  // 12 · zero console errors across the whole session
  add('zero console errors / page errors', consoleErrors.length === 0,
    consoleErrors.length ? consoleErrors.slice(0, 3).join(' | ') : 'clean');

  await page.close();
  return results;
}

// ── driver ─────────────────────────────────────────────────────────────
async function main() {
  const exePath = findChrome();
  console.log(`browser: ${exePath}`);
  const browser = await puppeteer.launch({
    executablePath: exePath,
    headless: true,
    userDataDir: path.join(ROOT, '.chrome-tmp'),
    args: [
      '--no-sandbox', '--disable-setuid-sandbox',
      '--disable-gpu', '--enable-unsafe-swiftshader',
      '--disable-crashpad', '--disable-crash-reporter',
      '--no-first-run', '--disable-dev-shm-usage',
      '--hide-scrollbars',
      '--window-size=1440,900',
    ],
  });

  const modes = URL_OVERRIDE ? ['url'] : MODE === 'both' ? ['prod', 'dev'] : [MODE];
  let failures = 0;

  for (const mode of modes) {
    let server = null;
    let baseUrl;
    if (URL_OVERRIDE) {
      baseUrl = URL_OVERRIDE;
    } else if (mode === 'prod') {
      await buildProd();
      server = spawnServer(['preview', '--host', '127.0.0.1', '--port', '5201', '--strictPort']);
      await waitForPort(5201);
      baseUrl = 'http://127.0.0.1:5201';
    } else {
      server = spawnServer(['--host', '127.0.0.1', '--port', '5199', '--strictPort']);
      await waitForPort(5199);
      baseUrl = 'http://127.0.0.1:5199';
    }

    console.log(`\n━━━ suite against ${mode}: ${baseUrl} ━━━`);
    try {
      const results = await runSuite(browser, baseUrl);
      failures += results.filter((r) => !r.pass).length;
    } catch (err) {
      console.error('suite crashed:', err);
      failures++;
    } finally {
      if (server) server.kill('SIGTERM');
      await sleep(300);
    }
  }

  await browser.close();
  console.log(failures === 0 ? '\nALL CHECKS PASSED ✅' : `\n${failures} CHECK(S) FAILED ❌`);
  process.exitCode = failures === 0 ? 0 : 1;
}

main().catch((e) => { console.error(e); process.exit(1); });
