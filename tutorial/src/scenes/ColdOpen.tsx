import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Cube} from '../components';
import {C, MONO, SANS, T, envelope, prog, useSceneFrame} from '../theme';
import {SceneProps} from '../types';
import {BOX, BOX_DX, BOX_MID_Y} from './shared';

const FS = 28;
const CH = FS * 0.6;
const PAD = 30;
const GAP = 12;
const LINE_H = 66;
const pillW = (s: string) => s.length * CH + PAD;

type Flow = {lines: string[][]; start: number; answer: string};

/**
 * A prompt builds up token by token (in reading order) left of the box, then slides into it;
 * a moment later the answer slides out of the right side.
 */
export const TokenFlow: React.FC<{f: number; flows: Flow[]; y: number; boxLeft: number; boxRight: number}> = ({
  f,
  flows,
  y,
  boxLeft,
  boxRight,
}) => {
  const out: React.ReactNode[] = [];
  flows.forEach((fl, k) => {
    const nTok = fl.lines.reduce((a, l) => a + l.length, 0);
    const appearEnd = fl.start + nTok * 3;
    const slide0 = appearEnd + 14;
    const slide = prog(f, slide0, slide0 + 34);
    const ans0 = slide0 + 46;
    const next = flows[k + 1];
    const ansFade = next ? 1 - prog(f, next.start + 20, next.start + 40) : 1;
    const blockW = Math.max(...fl.lines.map((l) => l.reduce((a, t) => a + pillW(t) + GAP, -GAP)));
    const top0 = y - (fl.lines.length * LINE_H) / 2;
    let idx = 0;
    if (f >= fl.start && slide < 1) {
      out.push(
        <div key={`p${k}`} style={{position: 'absolute', left: 0, top: 0, width: boxLeft, height: 1080, overflow: 'hidden'}}>
          <div style={{position: 'absolute', left: boxLeft - 50 - blockW + slide * slide * (blockW + 90), top: top0}}>
            {fl.lines.map((line, li) => {
              let x = 0;
              return line.map((tok) => {
                const i = idx++;
                const on = prog(f, fl.start + i * 3, fl.start + i * 3 + 8);
                const w = pillW(tok);
                const node = (
                  <div
                    key={`${li}-${i}`}
                    style={{
                      position: 'absolute',
                      left: x + (1 - on) * -10,
                      top: li * LINE_H,
                      width: w,
                      height: 50,
                      borderRadius: 0,
                      border: `1px solid ${C.grid}`,
                      background: C.panel,
                      color: C.text,
                      fontFamily: MONO,
                      fontSize: FS,
                      lineHeight: '48px',
                      textAlign: 'center',
                      opacity: on,
                    }}
                  >
                    {tok}
                  </div>
                );
                x += w + GAP;
                return node;
              });
            })}
          </div>
        </div>,
      );
    }
    const t = prog(f, ans0, ans0 + 30);
    if (f >= ans0 && ansFade > 0) {
      out.push(
        <div key={`a${k}`} style={{position: 'absolute', left: boxRight, top: y - 30, width: 1920 - boxRight, height: 60, overflow: 'hidden'}}>
          <div
            style={{
              position: 'absolute',
              left: -340 + t * 390,
              height: 56,
              padding: '0 22px',
              borderRadius: 0,
              border: `1px solid ${C.muted}`,
              background: C.panel,
              color: C.text,
              fontFamily: MONO,
              fontSize: 30,
              lineHeight: '54px',
              whiteSpace: 'nowrap',
              opacity: ansFade,
            }}
          >
            {fl.answer}
          </div>
        </div>,
      );
    }
  });
  return <>{out}</>;
};

export const coldOpenFlows = (p?: number): Flow[] => [
  {
    lines: [
      ['When', 'Mary', 'and', 'John'],
      ['went', 'to', 'the', 'store,'],
      ['John', 'gave', 'a', 'drink', 'to'],
    ],
    start: -40,
    answer: 'Mary',
  },
  {lines: [['a', '+', 'b', '=']], start: 104, answer: p ? `(a+b) mod ${p}` : '(a+b) mod p'},
];

export const ColdOpen: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur, 0, 0); // frame 0 is a finished picture; the cut to the glass scene is seamless
  const p = data.modadd?.p;
  const l1 = prog(f, 40, 64);
  const l2 = prog(f, 92, 118);
  return (
    <AbsoluteFill style={{opacity: env}}>
      <TokenFlow f={f} flows={coldOpenFlows(p)} y={BOX_MID_Y} boxLeft={BOX.left} boxRight={BOX.left + BOX.w + BOX_DX} />
      <div style={{position: 'absolute', left: BOX.left, top: BOX.top}}>
        <Cube w={BOX.w} h={BOX.h} glass={0} />
      </div>
      <div
        style={{
          position: 'absolute',
          top: 690,
          width: '100%',
          textAlign: 'center',
          fontFamily: SANS,
          fontSize: T.h1,
          fontWeight: 600,
          color: C.text,
          letterSpacing: -0.5,
          opacity: 1 - prog(f, dur - 26, dur - 4),
        }}
      >
        <span style={{opacity: l1}}>We built it. </span>
        <span style={{opacity: l2, color: C.muted}}>We don't know how it works.</span>
      </div>
    </AbsoluteFill>
  );
};
