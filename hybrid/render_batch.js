// Batched SVG -> PNG renderer: one Chrome session, many renders.
// render.js launches a browser per invocation (~1s overhead each); a 48-frame
// corpus animation needs hundreds of renders, so batching is mandatory.
// Usage: NODE_PATH=<spike>/node_modules node render_batch.js <manifest.json>
//   manifest: [{"svg": "/abs/path.svg", "png": "/abs/out.png", "bg": "#fff"}, ...]
const puppeteer = require('puppeteer-core');
const fs = require('fs');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

async function main() {
  const manifest = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 512, height: 512, deviceScaleFactor: 1 });
  for (const job of manifest) {
    const svg = fs.readFileSync(job.svg, 'utf8');
    const bg = job.bg || '#fff';
    await page.setContent(`<body style="margin:0;background:${bg}">${svg}</body>`);
    await page.screenshot({ path: job.png, clip: { x: 0, y: 0, width: 512, height: 512 } });
  }
  await browser.close();
  console.log(`rendered ${manifest.length}`);
}
main().catch(e => { console.error(e); process.exit(1); });
