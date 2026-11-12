import React from 'react';
import {AbsoluteFill, Sequence, useCurrentFrame} from 'remotion';
import {Backdrop, Circuit, Cube, Pill} from './components';
import {C, FPS, MONO, SANS, prog} from './theme';
import {Data, SceneProps} from './types';
import {GlassReveal, TAGLINE} from './scenes/GlassReveal';
import {ResearchLoop} from './scenes/ResearchLoop';
import {CircuitScene} from './scenes/CircuitScene';
import {circuitFreqs} from './scenes/shared';

// Tight cut for the README GIF: box to glass, the Fourier bars, the circuit lighting up.
// Each clip replays a stretch of a Tutorial scene at a higher speed (the scene's rame prop is remapped).
type Clip = {C: React.FC<SceneProps>; sceneDur: number; from: number; to: number; speed: number};

const CLIPS: Clip[] = [
  {C: GlassReveal, sceneDur: 8 * FPS, from: 6, to: 170, speed: 2.5},
  {C: ResearchLoop, sceneDur: 22 * FPS, from: 286, to: 410, speed: 2},
  {C: CircuitScene, sceneDur: 14 * FPS, from: 18, to: 384, speed: 3.6},
];
const XF = 8;
const clipLen = (c: Clip) => Math.round((c.to - c.from) / c.speed);
export const previewFrames = CLIPS.reduce((a, c) => a + clipLen(c), 0) - XF * (CLIPS.length - 1);

const Remap: React.FC<{clip: Clip; data: Data; len: number}> = ({clip, data, len}) => {
  const t = useCurrentFrame();
  const frame = Math.min(clip.to, Math.round(clip.from + t * clip.speed));
  const op = Math.min(prog(t, 0, XF), 1 - prog(t, len - XF, len));
  return (
    <AbsoluteFill style={{opacity: op}}>
      <clip.C data={data} dur={clip.sceneDur} frame={frame} />
    </AbsoluteFill>
  );
};

export const Preview: React.FC<{data: Data}> = ({data}) => {
  let at = 0;
  return (
    <AbsoluteFill style={{backgroundColor: C.navy}}>
      <div style={{width: 1920, height: 1080, transform: 'scale(0.5)', transformOrigin: '0 0', position: 'absolute'}}>
        <Backdrop>
          {CLIPS.map((c, i) => {
            const len = clipLen(c);
            const from = at;
            at += len - XF;
            return (
              <Sequence key={i} from={from} durationInFrames={len}>
                <Remap clip={c} data={data} len={len} />
              </Sequence>
            );
          })}
        </Backdrop>
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ hero still (1600x800)
export const Hero: React.FC<{data: Data}> = ({data}) => (
  <Backdrop step={50}>
    <div style={{position: 'absolute', left: 110, top: 150}}>
      <Cube w={580} h={400} glass={1}>
        <Circuit lit={5} width={520} freqs={circuitFreqs(data)} labels={false} shimmer={1} />
      </Cube>
    </div>
    <div style={{position: 'absolute', left: 880, top: 236, width: 660}}>
      <div style={{fontFamily: MONO, fontSize: 124, fontWeight: 700, color: C.text, letterSpacing: -3, lineHeight: 1}}>
        glass<span style={{color: C.accent, textShadow: `0 0 26px ${C.accent}66`}}>box</span>
      </div>
      <div style={{fontFamily: SANS, fontSize: 42, color: C.muted, marginTop: 22, lineHeight: 1.3}}>{TAGLINE}</div>
      <div style={{display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 44}}>
        {['activation patching', 'ablation', 'logit lens', 'probes', 'steering'].map((t) => (
          <Pill key={t} color={C.dim} size={22} style={{color: C.muted}}>
            {t}
          </Pill>
        ))}
      </div>
    </div>
  </Backdrop>
);
