// One palette and type scale, mirrored from glassbox/style.py: black, white, hairlines, one signal colour.
import {Easing, interpolate, useCurrentFrame} from 'remotion';

export const C = {
  navy: '#000000', // page background (the key keeps its old name)
  panel: '#0a0a0a',
  grid: '#262626',
  text: '#f2f2f2',
  muted: '#8a8a8a',
  accent: '#ffffff',
  dim: '#3d3d3d',
  warn: '#ff5a36',
};

export const MONO = '"JetBrains Mono", "Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace';
export const SANS = 'Inter, "Segoe UI", "Helvetica Neue", Arial, sans-serif';

// Type scale for a 1920x1080 frame (all >= 22 px so a frame shown 800 px wide stays readable).
export const T = {
  hero: 120,
  h1: 64,
  h2: 46,
  h3: 34,
  body: 30,
  small: 26,
  tiny: 22,
};

export const FPS = 30;

export const ease = Easing.bezier(0.33, 0, 0.2, 1);

/** 0 -> 1 between frames a and b, eased and clamped. */
export const prog = (f: number, a: number, b: number) =>
  interpolate(f, [a, b], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease});

/** Scene envelope: fade in over `inF` frames and out over `outF` frames. */
export const envelope = (f: number, dur: number, inF = 14, outF = 14) => {
  const i = inF > 0 ? prog(f, 0, inF) : 1;
  const o = outF > 0 ? 1 - prog(f, dur - outF, dur) : 1;
  return Math.min(i, o);
};

/** No glow in this theme; kept so scenes can ask for one without caring. */
export const glow = (_color = C.accent, _r = 14) => 'none';

/** Cut text at a word boundary so it fits `max` characters. */
export const truncate = (s: string | undefined, max: number) => {
  if (!s) return '';
  if (s.length <= max) return s;
  const cut = s.slice(0, max - 1);
  const sp = cut.lastIndexOf(' ');
  return (sp > max * 0.6 ? cut.slice(0, sp) : cut).replace(/[\s,;:(—-]+$/, '') + '…';
};

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits).replace(/\.0+$/, '')}%`;

/** The scene's frame: an explicit override (used for time-remapped clips in Preview) or the timeline frame. */
export const useSceneFrame = (override?: number) => {
  const f = useCurrentFrame();
  return override ?? f;
};
