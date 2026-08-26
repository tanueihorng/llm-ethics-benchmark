// Quick visual probe: loads the page, waits for ready, saves screenshots,
// prints console errors and capture stats. Not part of the pass/fail suite.
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
mkdirSync(path.join(ROOT, 'screenshots'), { recursive: true });
const exe = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
if (!existsSync(exe)) throw new Error('chrome not found: ' + exe);

const browser = await puppeteer.launch({
  executablePath: exe,
  headless: true,
  userDataDir: path.join(ROOT, '.chrome-tmp'),
  args: [
    '--no-sandbox', '--disable-setuid-sandbox',
    '--disable-gpu', '--enable-unsafe-swiftshader',
    '--disable-crashpad', '--disable-crash-reporter',
    '--no-first-run', '--disable-dev-shm-usage',
    '--window-size=1440,900',
  ],
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });

const errors = [];
page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) errors.push(`[${m.type()}] ${m.text()}`); });
page.on('pageerror', (e) => errors.push('[pageerror] ' + e));

await page.goto(process.argv[2] || 'http://127.0.0.1:5201/?quality=low&cycle=0&hour=9', { waitUntil: 'domcontentloaded' });
try {
  await page.waitForFunction(() => window.__WORLD__?.ready === true, { timeout: 30000 });
  console.log('ready ✓');
} catch { console.log('READY TIMEOUT'); }

for (const [name, hour, fly] of [
  ['probe-day', 9.5, null],
  ['probe-golden', 17.6, 1],
  ['probe-night', 23.4, 3],
]) {
  await page.evaluate((h, f) => {
    const w = window.__WORLD__;
    w.setHour(h);
    if (f !== null && f !== undefined) { w.openCard(f); w.flyTo(f); }
    return new Promise((res) => setTimeout(res, f !== null && f !== undefined ? 2600 : 400));
  }, hour, fly ?? null);
  const stats = await page.evaluate(() => window.__WORLD__.captureStats());
  await page.screenshot({ path: path.join(ROOT, 'screenshots', `${name}.png`) });
  console.log(name, JSON.stringify(stats));
}

console.log('console issues:', errors.length ? '\n' + errors.join('\n') : 'none');
await browser.close();
