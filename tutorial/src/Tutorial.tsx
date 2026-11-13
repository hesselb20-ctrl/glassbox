import React from 'react';
import {AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig} from 'remotion';
import {Backdrop} from './components';
import {C, FPS} from './theme';
import {Data, SceneProps} from './types';
import {ColdOpen} from './scenes/ColdOpen';
import {GlassReveal} from './scenes/GlassReveal';
import {ResearchLoop} from './scenes/ResearchLoop';
import {CircuitScene} from './scenes/CircuitScene';
import {Verify} from './scenes/Verify';
import {Demo27b} from './scenes/Demo27b';
import {Ladder} from './scenes/Ladder';
import {AuditScene, BenchScene, NullScene} from './scenes/Upgrade';
import {EndCard, Scorecard, ZeroCost} from './scenes/Closing';

export type Scene = {id: string; seconds: number; C: React.FC<SceneProps & {coldDur?: number}>};

/** The timed brief. Pending runs get a short "not run yet" card instead of their full slot. */
export const scenes = (data: Data): Scene[] => [
  {id: 'cold-open', seconds: 8, C: ColdOpen},
  {id: 'glass', seconds: 8, C: GlassReveal},
  {id: 'loop', seconds: 22, C: ResearchLoop},
  {id: 'circuit', seconds: 14, C: CircuitScene},
  {id: 'verify', seconds: 12, C: Verify},
  {id: 'demo27b', seconds: data.demo27b?.available ? 14 : 5, C: Demo27b},
  {id: 'ladder', seconds: data.ladder?.available && data.ladder?.rungs?.length ? 10 : 5, C: Ladder},
  {id: 'bench', seconds: data.bench?.available ? 9 : 4, C: BenchScene},
  {id: 'nulls', seconds: data.nulls?.available ? 8 : 4, C: NullScene},
  {id: 'audit', seconds: data.audit?.available ? 9 : 4, C: AuditScene},
  {id: 'scorecard', seconds: 6, C: Scorecard},
  {id: 'zero', seconds: 6, C: ZeroCost},
  {id: 'end', seconds: 6, C: EndCard},
];

export const totalFrames = (data: Data) => scenes(data).reduce((a, s) => a + Math.round(s.seconds * FPS), 0);

const Progress: React.FC = () => {
  const f = useCurrentFrame();
  const {durationInFrames, width} = useVideoConfig();
  return (
    <div style={{position: 'absolute', left: 0, bottom: 0, height: 4, width: (f / (durationInFrames - 1)) * width, background: C.dim}} />
  );
};

export const Tutorial: React.FC<{data: Data}> = ({data}) => {
  let at = 0;
  const list = scenes(data);
  const coldDur = Math.round(list[0].seconds * FPS);
  return (
    <Backdrop>
      {list.map((s) => {
        const dur = Math.round(s.seconds * FPS);
        const from = at;
        at += dur;
        return (
          <Sequence key={s.id} from={from} durationInFrames={dur} name={s.id}>
            <s.C data={data} dur={dur} coldDur={coldDur} />
          </Sequence>
        );
      })}
      <AbsoluteFill>
        <Progress />
      </AbsoluteFill>
    </Backdrop>
  );
};
