import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Circuit, Heading, Kicker} from '../components';
import {C, MONO, envelope, prog, useSceneFrame} from '../theme';
import {SceneProps} from '../types';
import {circuitFreqs} from './shared';

/** Which circuit stage (0..4) a pseudocode line belongs to, from its "# 1." ... "# 4." step comments. */
const stagesOf = (lines: string[]) => {
  let s = 0;
  return lines.map((l, i) => {
    const m = l.match(/#\s*(\d)\./);
    if (m) s = Math.min(4, Number(m[1]));
    else if (i > 0 && /return|logit/.test(l)) s = 4;
    return s;
  });
};

export const typingWindow = (dur: number) => ({start: Math.round(dur * 0.07), end: Math.round(dur * 0.84)});

/** Pseudocode with a typewriter; returns the lit level for the circuit at this frame. */
export const usePseudocode = (code: string, f: number, start: number, end: number) => {
  const lines = code.split('\n');
  const stages = stagesOf(lines);
  const total = lines.reduce((a, l) => a + l.length + 6, 0);
  const typed = prog(f, start, end) * total;
  let acc = 0;
  let cur = 0;
  const shown = lines.map((l, i) => {
    const n = Math.max(0, Math.min(l.length, typed - acc));
    if (typed >= acc) cur = i;
    acc += l.length + 6;
    return l.slice(0, Math.floor(n));
  });
  // lit = stage of the line being typed, plus how far through that stage's lines we are
  const st = stages[cur] ?? 0;
  const same = stages.map((s, i) => (s === st ? i : -1)).filter((i) => i >= 0);
  const firstI = same[0];
  const lastI = same[same.length - 1];
  let a0 = 0;
  for (let i = 0; i < firstI; i++) a0 += lines[i].length + 6;
  let a1 = a0;
  for (let i = firstI; i <= lastI; i++) a1 += lines[i].length + 6;
  const frac = Math.max(0, Math.min(1, (typed - a0) / Math.max(1, a1 - a0)));
  const done = typed >= total - 1;
  return {lines, shown, cur, lit: done ? 5 : st + frac, done};
};

const CodeLine: React.FC<{text: string; active: boolean}> = ({text, active}) => {
  const h = text.indexOf('#');
  const code = h >= 0 ? text.slice(0, h) : text;
  const comment = h >= 0 ? text.slice(h) : '';
  return (
    <div style={{display: 'flex', alignItems: 'center'}}>
      <div style={{width: 6, height: 30, borderRadius: 0, marginRight: 18, background: active ? C.accent : 'transparent'}} />
      <span style={{color: C.text}}>{code}</span>
      <span style={{color: active ? C.text : C.muted, opacity: active ? 0.9 : 1}}>{comment}</span>
    </div>
  );
};

export const CircuitScene: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const m = data.modadd ?? {};
  const {start, end} = typingWindow(dur);
  const pc = usePseudocode(m.pseudocode ?? '', f, start, end);
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 50}}>
        <Kicker>The recovered algorithm</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          The circuit lights up, path by path, <span style={{color: C.muted}}>as the pseudocode writes itself.</span>
        </Heading>
      </div>
      <div style={{position: 'absolute', left: 330, top: 150}}>
        <Circuit lit={pc.lit} width={1260} freqs={circuitFreqs(data)} p={m.p} />
      </div>
      <div
        style={{
          position: 'absolute',
          left: 110,
          top: 760 - 70,
          width: 1700,
          height: 330,
          background: C.navy,
          border: `1px solid ${C.grid}`,
          borderRadius: 0,
          padding: '18px 20px',
          boxSizing: 'border-box',
          fontFamily: MONO,
          fontSize: 23,
          lineHeight: '36px',
          whiteSpace: 'pre',
          overflow: 'hidden',
        }}
      >
        {pc.shown.map((s, i) => (
          <CodeLine key={i} text={s} active={!pc.done ? i === pc.cur : /return/.test(pc.lines[i])} />
        ))}
      </div>
    </AbsoluteFill>
  );
};
