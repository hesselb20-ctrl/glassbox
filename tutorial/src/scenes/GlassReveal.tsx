import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Circuit, Cube} from '../components';
import {C, MONO, SANS, T, envelope, prog, useSceneFrame} from '../theme';
import {SceneProps} from '../types';
import {TokenFlow, coldOpenFlows} from './ColdOpen';
import {BOX, BOX_DX, BOX_MID_Y, circuitFreqs} from './shared';

export const TAGLINE = 'Turn black boxes into glass boxes.';

export const GlassReveal: React.FC<SceneProps & {coldDur?: number}> = ({data, dur, coldDur = 240, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur, 0, 16);
  const glass = prog(f, 12, 96);
  const lit = prog(f, 120, 220) * 5;
  const title = prog(f, 100, 130);
  const tag = prog(f, 128, 156);
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{opacity: 1 - prog(f, 0, 24)}}>
        <TokenFlow
          f={f + coldDur}
          flows={coldOpenFlows(data.modadd?.p)}
          y={BOX_MID_Y}
          boxLeft={BOX.left}
          boxRight={BOX.left + BOX.w + BOX_DX}
        />
      </div>
      <div style={{position: 'absolute', left: BOX.left, top: BOX.top}}>
        <Cube w={BOX.w} h={BOX.h} glass={glass}>
          <Circuit lit={lit} width={BOX.w - 70} freqs={circuitFreqs(data)} labels={false} shimmer={glass} />
        </Cube>
      </div>
      <div style={{position: 'absolute', top: 640, width: '100%', textAlign: 'center'}}>
        <div
          style={{
            fontFamily: MONO,
            fontSize: T.hero,
            fontWeight: 700,
            color: C.text,
            letterSpacing: -2,
            opacity: title,
            transform: `translateY(${(1 - title) * 16}px)`,
          }}
        >
          glass<span style={{color: C.accent, textShadow: `0 0 24px ${C.accent}66`}}>box</span>
        </div>
        <div style={{fontFamily: SANS, fontSize: 44, color: C.muted, marginTop: 6, opacity: tag}}>{TAGLINE}</div>
      </div>
    </AbsoluteFill>
  );
};
