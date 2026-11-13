import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {Backdrop} from './components';
import {FPS} from './theme';
import {Data, SceneProps} from './types';
import {GlassReveal} from './scenes/GlassReveal';
import {ResearchLoop} from './scenes/ResearchLoop';
import {CircuitScene} from './scenes/CircuitScene';
import {Verify} from './scenes/Verify';
import {Demo27b} from './scenes/Demo27b';
import {Ladder} from './scenes/Ladder';
import {Scorecard, ZeroCost} from './scenes/Closing';
import {AuditScene, BenchScene, NullScene} from './scenes/Upgrade';
import {Architecture, Backends, Prereg} from './scenes/Figures';

// Every README figure is a short GIF: a stretch of a scene played faster, then held on its last frame so it can be read.
// `h` crops the 1920x1080 scene from the top (scenes rarely use the full height). Rendered at 2/3 scale: 1280 px wide.
export type Gif = {id: string; C: React.FC<SceneProps>; sceneSec: number; from: number; to: number; speed: number; hold: number; h: number};

export const GIF_SCALE = 2 / 3;

export const GIFS: Gif[] = [
  {id: 'hero', C: GlassReveal, sceneSec: 8, from: 6, to: 200, speed: 2, hold: 4, h: 920},
  {id: 'how-it-works', C: Architecture, sceneSec: 7, from: 0, to: 150, speed: 1.5, hold: 5, h: 930},
  {id: 'free-stack', C: ZeroCost, sceneSec: 6, from: 0, to: 110, speed: 1.5, hold: 4, h: 800},
  {id: 'backends', C: Backends, sceneSec: 7, from: 0, to: 140, speed: 1.5, hold: 5, h: 990},
  {id: 'scorecard', C: Scorecard, sceneSec: 6, from: 0, to: 160, speed: 1.5, hold: 5, h: 1000},
  {id: 'fourier', C: ResearchLoop, sceneSec: 22, from: 240, to: 410, speed: 2, hold: 4, h: 1080},
  {id: 'circuit-modadd', C: CircuitScene, sceneSec: 14, from: 18, to: 396, speed: 3, hold: 5, h: 1080},
  {id: 'notebook', C: ResearchLoop, sceneSec: 22, from: 410, to: 640, speed: 2, hold: 5, h: 1040},
  {id: 'prereg-record', C: Prereg, sceneSec: 7, from: 0, to: 150, speed: 1.5, hold: 5, h: 1020},
  {id: 'null-controls', C: NullScene, sceneSec: 8, from: 0, to: 215, speed: 2, hold: 5, h: 1000},
  {id: 'bench-scorecard', C: BenchScene, sceneSec: 9, from: 0, to: 250, speed: 2, hold: 5, h: 860},
  {id: 'steering-27b', C: Demo27b, sceneSec: 14, from: 0, to: 400, speed: 3, hold: 5, h: 1080},
  {id: 'scale-ladder', C: Ladder, sceneSec: 10, from: 0, to: 280, speed: 2.5, hold: 5, h: 1080},
  {id: 'model-diff', C: AuditScene, sceneSec: 9, from: 0, to: 250, speed: 2, hold: 5, h: 1000},
  {id: 'verification', C: Verify, sceneSec: 12, from: 0, to: 340, speed: 2.5, hold: 5, h: 980},
];

export const gifFrames = (g: Gif) => Math.round((g.to - g.from) / g.speed) + Math.round(g.hold * FPS);
export const gifSize = (g: Gif) => ({width: Math.round(1920 * GIF_SCALE), height: Math.round((g.h * GIF_SCALE) / 2) * 2});

// Props are serialised by the renderer, so a composition gets the figure's id and looks the scene up here.
export const GifClip: React.FC<{data: Data; id: string}> = ({data, id}) => {
  const gif = GIFS.find((g) => g.id === id) ?? GIFS[0];
  const t = useCurrentFrame();
  const frame = Math.min(gif.to, Math.round(gif.from + t * gif.speed));
  return (
    <AbsoluteFill style={{backgroundColor: '#000000'}}>
      <div style={{width: 1920, height: 1080, transform: `scale(${GIF_SCALE})`, transformOrigin: '0 0', position: 'absolute'}}>
        <Backdrop>
          <gif.C data={data} dur={Math.round(gif.sceneSec * FPS)} frame={frame} />
        </Backdrop>
      </div>
    </AbsoluteFill>
  );
};
