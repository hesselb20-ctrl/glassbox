// Render every README figure (the gif-* compositions in src/Gifs.tsx) from one bundle.
// Usage: node scripts/gifs.mjs <workDir> [--stills] [--only=hero,backends] [--out=../media]
//   --stills   only render the last frame of each composition as PNG into <workDir> (layout review)
//   otherwise  render each composition to <workDir>/<id>.mp4 and convert it to <out>/<id>.gif with ffmpeg
// Browser: set REMOTION_BROWSER to a chrome-headless-shell.exe on a short path (see README).
import {getCompositions, renderMedia, renderStill} from '@remotion/renderer';
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const opts = Object.fromEntries(args.filter((a) => a.startsWith('--')).map((a) => a.slice(2).split('=')));
const [workDir] = args.filter((a) => !a.startsWith('--'));
const serveUrl = path.resolve('build');
const browserExecutable = process.env.REMOTION_BROWSER || null;
const outDir = path.resolve(opts.out ?? '../media');
const only = opts.only ? opts.only.split(',') : null;
fs.mkdirSync(workDir, {recursive: true});

const comps = (await getCompositions(serveUrl, {browserExecutable})).filter(
  (c) => c.id.startsWith('gif-') && (!only || only.includes(c.id.slice(4))),
);
for (const composition of comps) {
  const name = composition.id.slice(4);
  if ('stills' in opts) {
    const output = path.join(workDir, `${name}.png`);
    await renderStill({composition, serveUrl, frame: composition.durationInFrames - 1, output, browserExecutable});
    console.log(output);
    continue;
  }
  const mp4 = path.join(workDir, `${name}.mp4`);
  await renderMedia({composition, serveUrl, codec: 'h264', crf: 12, outputLocation: mp4, browserExecutable, chromiumOptions: {gl: 'swangle'}, concurrency: 6});
  const gif = path.join(outDir, `${name}.gif`);
  // 15 fps, a small palette (the theme is black, white, greys and one signal colour), frame differencing for the held last frame
  const vf = 'fps=15,split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle';
  execFileSync('ffmpeg', ['-y', '-v', 'error', '-i', mp4, '-vf', vf, '-loop', '0', gif]);
  console.log(`${gif}  ${(fs.statSync(gif).size / 1e6).toFixed(2)} MB`);
}
