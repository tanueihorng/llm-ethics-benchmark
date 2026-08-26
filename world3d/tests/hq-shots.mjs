// One-off: high-quality screenshots (reflections + envmap on) for visual review.
import puppeteer from 'puppeteer-core';
import { existsSync } from 'node:fs';

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
  ].filter(Boolean);
  for (const c of candidates) if (existsSync(c)) return c;
  throw new Error('No Chrome found; set CHROME_PATH');
}
const CHROME = findChrome();

const url = process.argv[2] ?? 'http://127.0.0.1:5201/';
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu', '--enable-unsafe-swiftshader', '--disable-crashpad', '--user-data-dir=.chrome-tmp'],
  defaultViewport: { width: 1440, height: 900 },
});
const page = await browser.newPage();
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForFunction(() => window.__WORLD__?.ready === true, { timeout: 60000 });
await new Promise((r) => setTimeout(r, 2500));

const shots = [
  ['hq-day', 12, null],
  ['hq-golden', 17.2, 1],
  ['hq-night', 23.5, null],
];
for (const [name, hr, fly] of shots) {
  await page.evaluate((hour, f) => {
    const w = window.__WORLD__;
    w.setHour(hour);
    if (f !== null) { w.openCard(f); w.flyTo(f); }
    return new Promise((res) => setTimeout(res, f !== null ? 3000 : 600));
  }, hr, fly);
  if (name === 'hq-night') {
    await page.evaluate(() => {
      const w = window.__WORLD__;
      w.closeCard();
      w.camera.position.set(470, 210, -450);
      w.controls.target.set(-40, 60, 110);
      w.controls.update();
    });
    await new Promise((r) => setTimeout(r, 800));
  }
  await page.screenshot({ path: `screenshots/${name}.png` });
  console.log(`saved screenshots/${name}.png`);
}
await browser.close();
