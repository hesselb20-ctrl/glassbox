// Review helper: render several stills from one bundle without re-bundling.
// Usage: node scripts/stills.mjs <outDir> <composition> <frame> [<frame> ...] [--props=file.json] [--scale=0.5]
// Browser: set REMOTION_BROWSER to a chrome-headless-shell.exe on a short path (see README).
import {selectComposition, renderStill} from '@remotion/renderer';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const opts = Object.fromEntries(args.filter((a) => a.startsWith('--')).map((a) => a.slice(2).split('=')));
const [outDir, id, ...frames] = args.filter((a) => !a.startsWith('--'));
const serveUrl = path.resolve('build');
const browserExecutable = process.env.REMOTION_BROWSER || null;
const inputProps = opts.props ? JSON.parse(fs.readFileSync(opts.props, 'utf8')) : {};
fs.mkdirSync(outDir, {recursive: true});
const composition = await selectComposition({serveUrl, id, inputProps, browserExecutable});
for (const fr of frames) {
  const output = path.join(outDir, `${id}-${String(fr).padStart(5, '0')}.png`);
  await renderStill({composition, serveUrl, frame: Number(fr), output, inputProps, browserExecutable, scale: Number(opts.scale ?? 1)});
  console.log(output);
}
