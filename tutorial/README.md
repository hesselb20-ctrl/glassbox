# glassbox tutorial video

The tutorial video and every README figure (all GIFs), drawn in code with [Remotion](https://www.remotion.dev)
(React + SVG + CSS, no images, no audio). Every number on screen comes from `src/data.json`, which
`make_data.py` writes from the benchmark and demo runs.

| Composition | Size | Output |
|---|---|---|
| `Tutorial` | 1920x1080, 30 fps, 132 s with every section in data.json (sections without data get a short "not run yet" card) | `media/tutorial.mp4` |
| `Preview` | 960x540, about 11 s, converted to a 15 fps GIF | `media/preview.gif` |
| `gif-<name>` (15 figures) | 1280 px wide, a scene clip plus a held last frame | `media/<name>.gif` |
| `Hero` (still) | 1600x800 | not used by the README any more |

## Layout

- `src/Root.tsx`: registers the three compositions.
- `src/Tutorial.tsx`: the timed brief (scene list with seconds). Pending runs get a 5 s "not run yet" card;
  when `demo27b.available` / `ladder.available` are true they get their full 14 s / 10 s slot.
- `src/scenes/*.tsx`: one component per scene; each fades itself in and out. `Upgrade.tsx` holds the three scenes
  on glassbox-bench (a sealed task solved), null controls and the model-diff audit (26 s together).
- `src/components.tsx`: shared pieces (glass cube, circuit drawing, cards, pills).
- `src/theme.ts`: palette and type scale, mirrored from `glassbox/style.py`. The theme is black, white, hairlines,
  square corners and one signal colour (orange, for failures and refuted predictions only).
- `src/Gifs.tsx`: the list of README figures. Each one replays a stretch of a scene and holds its last frame.
- `src/scenes/Figures.tsx`: scenes that exist only as figures (architecture, backend table, prediction record).
- `scripts/gifs.mjs`: renders every figure and converts it to a GIF with ffmpeg.
- `src/types.ts`: the shape of `data.json`; every field is optional.
- `scripts/stills.mjs`: renders many review stills from one bundle.

## Render

From this folder, after `npm install`:

```sh
python make_data.py                      # refresh src/data.json (optional)
npx remotion bundle src/index.ts --out-dir=build

# video: JPEG sequence, then ffmpeg (Remotion's own H.264 read back slightly tinted)
npx remotion render build Tutorial out/frames --sequence --image-format=jpeg --jpeg-quality=95 --gl=swangle --concurrency=6 --browser-executable="$REMOTION_BROWSER"
ffmpeg -y -framerate 30 -i out/frames/element-%04d.jpeg -vf "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p" -c:v libx264 -preset slow -crf 20 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -movflags +faststart -map_metadata -1 ../media/tutorial.mp4

# README GIF (every 2nd frame = 15 fps)
npx remotion render build Preview out/preview.mp4 --gl=swangle --browser-executable="$REMOTION_BROWSER"
ffmpeg -y -i out/preview.mp4 -vf "fps=15,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" ../media/preview.gif

# every README figure as a GIF (REMOTION_BROWSER must be set); --only=hero,backends for a subset, --stills for last frames
node scripts/gifs.mjs out/gifwork --out=../media

# hero still
npx remotion still build Hero ../media/hero.png --gl=swangle --browser-executable="$REMOTION_BROWSER"

# review stills (several frames, one browser): REMOTION_BROWSER must be set
node scripts/stills.mjs out/review Tutorial 0 300 900 1300 1700 --scale=0.5
```

The frame file names depend on the frame count (`element-%04d.jpeg` for fewer than 10,000 frames).
`--gl=swangle` renders on the CPU, so the GPUs stay free for model runs.

To check the pending scenes before their runs finish, render with a props file of the form
`{"data": {...}}` (`--props=file.json`); keep such test data out of `src/data.json`.

## Windows: long project paths

When the project folder is nested deeply (absolute path near 260 characters), Remotion's downloaded
browser (`node_modules/.remotion/chrome-headless-shell/...`) fails to start with
`Failed to launch the browser process ... ENOENT`, although the file exists. `npm install` itself worked
in the long path. The fix: copy the folder
`node_modules/.remotion/chrome-headless-shell/win64/chrome-headless-shell-win64` to a short path and point
`REMOTION_BROWSER` at the `chrome-headless-shell.exe` in it, then pass
`--browser-executable="$REMOTION_BROWSER"` (or `browserExecutable` in the Node API, as `scripts/stills.mjs`
does). For example a short folder such as C:/tools/chs.
From a normal-depth folder the default browser works and the flag can be left out.
