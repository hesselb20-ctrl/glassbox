import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Card, Heading, Kicker, Pill} from '../components';
import {C, MONO, SANS, envelope, prog, truncate, useSceneFrame} from '../theme';
import {SceneProps} from '../types';

// Scenes that exist only as README figures (rendered to GIF, see Gifs.tsx): the architecture, the backend
// capability table and the prediction record. Same palette and type scale as the video.

const Rule: React.FC<{x: number; y: number; w: number; on: number; label?: string}> = ({x, y, w, on, label}) => (
  <div style={{position: 'absolute', left: x, top: y, width: w, height: 1}}>
    <div style={{width: `${on * 100}%`, height: 1, background: C.text}} />
    {label && (
      <div style={{position: 'absolute', left: 0, right: 0, top: -34, textAlign: 'center', fontFamily: MONO, fontSize: 22, color: C.muted, letterSpacing: 4, opacity: on}}>
        {label}
      </div>
    )}
  </div>
);

// ------------------------------------------------------------------------------------------ how it works
export const Architecture: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const backends = Object.keys(data.backends?.backends ?? {});
  const tools = ['preregister', 'run_experiment', 'verify_hypothesis', 'write_report'];
  const cols = [
    {x: 110, w: 430, kicker: '01 / scientist', title: 'Claude Code', lines: ['forms hypotheses', 'writes the prediction first', 'reads numbers and plots', 'rejects and revises']},
    {x: 745, w: 430, kicker: '02 / lab', title: 'glassbox', lines: ['experiment primitives', 'hash-chained notebook', 'null-calibrated verification', 'reports with cited claims']},
    {x: 1380, w: 430, kicker: '03 / model', title: 'four backends', lines: backends.length ? backends : ['local-full', 'local-gguf', 'local-4bit', 'ndif']},
  ];
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56, right: 110}}>
        <Kicker>How it works</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          The scientist proposes. <span style={{color: C.muted}}>The lab runs it, scores it and keeps the record.</span>
        </Heading>
      </div>
      {cols.map((c, i) => {
        const on = prog(f, 10 + i * 26, 30 + i * 26);
        return (
          <Card key={c.title} style={{position: 'absolute', left: c.x, top: 250, width: c.w, height: 420, padding: 34, opacity: on, transform: `translateY(${(1 - on) * 14}px)`}} border={i === 1 ? C.text : C.grid}>
            <div style={{fontFamily: MONO, fontSize: 22, color: C.muted, letterSpacing: 4, textTransform: 'uppercase'}}>{c.kicker}</div>
            <div style={{fontFamily: SANS, fontSize: 50, fontWeight: 500, color: C.text, letterSpacing: -1.5, marginTop: 14}}>{c.title}</div>
            <div style={{height: 1, background: C.grid, margin: '26px 0 22px'}} />
            {c.lines.map((l, k) => (
              <div key={l} style={{fontFamily: i === 2 ? MONO : SANS, fontSize: 27, color: C.text, lineHeight: 1.75, opacity: prog(f, 26 + i * 26 + k * 5, 38 + i * 26 + k * 5)}}>
                <span style={{color: C.dim, fontFamily: MONO, marginRight: 14}}>{String(k + 1).padStart(2, '0')}</span>
                {l}
              </div>
            ))}
          </Card>
        );
      })}
      <Rule x={540} y={460} w={205} on={prog(f, 34, 52)} label="MCP" />
      <Rule x={1175} y={460} w={205} on={prog(f, 60, 78)} label="HOOKS" />
      <div style={{position: 'absolute', left: 110, right: 110, top: 740, display: 'flex', gap: 14, alignItems: 'center', opacity: prog(f, 92, 110)}}>
        <span style={{fontFamily: MONO, fontSize: 22, color: C.muted, letterSpacing: 4, marginRight: 14}}>TOOLS</span>
        {tools.map((t, i) => (
          <Pill key={t} color={i === 0 ? C.text : C.muted} size={24} style={{textTransform: 'none', opacity: prog(f, 96 + i * 6, 108 + i * 6)}}>
            {t}
          </Pill>
        ))}
      </div>
      <div style={{position: 'absolute', left: 110, right: 110, top: 836, fontFamily: SANS, fontSize: 30, color: C.muted, opacity: prog(f, 124, 140)}}>
        <span style={{color: C.text}}>No experiment without a prediction.</span> No claim without an experiment that ran.
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------------------------------ backends
const LABEL: Record<string, string> = {
  attribution_patching: 'attribution patching',
  activation_patching: 'activation patching',
  path_patching: 'path patching',
  ablation: 'ablation',
  logit_lens: 'logit lens',
  direct_logit_attribution: 'direct logit attribution',
  probe: 'linear probes',
  attention_patterns: 'attention patterns',
  direction: 'difference-of-means direction',
  steering: 'steering',
  periodicity: 'Fourier analysis',
};

export const Backends: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const B = data.backends ?? {};
  const names = Object.keys(B.backends ?? {});
  const prims = Object.keys(B.primitives ?? {}).filter((p) => !['dataset', 'baseline'].includes(p)).slice(0, 11);
  const rowH = 54;
  const colW = 300;
  const left = 110;
  const labelW = 520;
  const top = 250;
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56, right: 110}}>
        <Kicker>Four backends</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          One interface. <span style={{color: C.muted}}>Quantized results are screening only.</span>
        </Heading>
      </div>
      {names.map((n, j) => {
        const b = B.backends?.[n] ?? {};
        const on = prog(f, 8 + j * 8, 22 + j * 8);
        return (
          <div key={n} style={{position: 'absolute', left: left + labelW + j * colW, top: top - 84, width: colW - 16, opacity: on}}>
            <div style={{fontFamily: MONO, fontSize: 28, fontWeight: 700, color: C.text}}>{n}</div>
            <div style={{fontFamily: MONO, fontSize: 20, color: b.verifies ? C.text : C.muted, letterSpacing: 2, marginTop: 6, textTransform: 'uppercase'}}>
              {b.verifies ? 'verified' : 'screening'}
            </div>
          </div>
        );
      })}
      <div style={{position: 'absolute', left, top: top - 4, width: labelW + names.length * colW - 16, height: 1, background: C.text, opacity: prog(f, 20, 34)}} />
      {prims.map((p, i) => {
        const on = prog(f, 34 + i * 5, 46 + i * 5);
        return (
          <div key={p} style={{position: 'absolute', left, top: top + i * rowH, width: labelW + names.length * colW - 16, height: rowH, borderBottom: `1px solid ${C.grid}`, opacity: on}}>
            <div style={{position: 'absolute', left: 0, top: 12, fontFamily: SANS, fontSize: 27, color: C.text}}>{LABEL[p] ?? p.replace(/_/g, ' ')}</div>
            {names.map((n, j) => {
              const ok = B.primitives?.[p]?.[n];
              const cell = prog(f, 40 + i * 5 + j * 3, 50 + i * 5 + j * 3);
              return (
                <div key={n} style={{position: 'absolute', left: labelW + j * colW, top: 19, opacity: cell}}>
                  {ok ? <div style={{width: 16, height: 16, background: C.text}} /> : <div style={{width: 16, height: 1, marginTop: 8, background: C.dim}} />}
                </div>
              );
            })}
          </div>
        );
      })}
      <div style={{position: 'absolute', left, right: 110, top: top + prims.length * rowH + 34, fontFamily: SANS, fontSize: 27, color: C.muted, opacity: prog(f, 110, 126)}}>
        An unsupported primitive raises an error that names a backend that works.
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------------------------------ prediction record
export const Prereg: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const P = data.prereg ?? {};
  const total = (P.confirmed ?? 0) + (P.refuted ?? 0) + (P.inconclusive ?? 0) || 1;
  const bars: [string, number, string][] = [
    ['confirmed', P.confirmed ?? 0, C.text],
    ['refuted', P.refuted ?? 0, C.warn],
    ['inconclusive', P.inconclusive ?? 0, C.dim],
  ];
  const W = 1700;
  const grow = prog(f, 14, 60);
  let x = 0;
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56, right: 110}}>
        <Kicker>Pre-registration · {P.investigations} investigations</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          The prediction is written first. <span style={{color: C.muted}}>The lab scores it.</span>
        </Heading>
      </div>
      <div style={{position: 'absolute', left: 110, top: 236, display: 'flex', gap: 90}}>
        {bars.map(([label, v, c], i) => (
          <div key={label} style={{opacity: prog(f, 8 + i * 8, 22 + i * 8)}}>
            <div style={{fontFamily: MONO, fontSize: 22, color: C.muted, letterSpacing: 4, textTransform: 'uppercase'}}>{label}</div>
            <div style={{fontFamily: MONO, fontSize: 92, fontWeight: 700, color: c === C.dim ? C.muted : c, lineHeight: 1.1}}>{Math.round(v * grow)}</div>
          </div>
        ))}
        <div style={{opacity: prog(f, 40, 56)}}>
          <div style={{fontFamily: MONO, fontSize: 22, color: C.muted, letterSpacing: 4, textTransform: 'uppercase'}}>hit rate</div>
          <div style={{fontFamily: MONO, fontSize: 92, fontWeight: 700, color: C.text, lineHeight: 1.1}}>{Math.round((P.hit_rate ?? 0) * 100 * grow)}%</div>
        </div>
      </div>
      <div style={{position: 'absolute', left: 110, top: 420, width: W, height: 20, background: C.panel}}>
        {bars.map(([label, v, c]) => {
          const w = (v / total) * W * grow;
          const el = <div key={label} style={{position: 'absolute', left: x, top: 0, width: Math.max(0, w - 3), height: 20, background: c}} />;
          x += w;
          return el;
        })}
      </div>
      <div style={{position: 'absolute', left: 110, top: 486, fontFamily: MONO, fontSize: 22, color: C.muted, letterSpacing: 4, opacity: prog(f, 60, 74)}}>
        REFUTED PREDICTIONS STAY IN THE REPORT
      </div>
      {(P.examples ?? []).slice(0, 3).map((e, i) => {
        const on = prog(f, 70 + i * 16, 88 + i * 16);
        return (
          <div key={i} style={{position: 'absolute', left: 110, top: 536 + i * 132, width: W, height: 112, borderTop: `1px solid ${C.grid}`, opacity: on, transform: `translateY(${(1 - on) * 10}px)`}}>
            <div style={{position: 'absolute', left: 0, top: 18, width: 6, height: 76, background: C.warn}} />
            <div style={{position: 'absolute', left: 30, top: 14, right: 0, fontFamily: SANS, fontSize: 27, color: C.text}}>{truncate(e.hypothesis, 118)}</div>
            <div style={{position: 'absolute', left: 30, top: 62, right: 0, fontFamily: MONO, fontSize: 23, color: C.muted}}>
              predicted {e.predicted} <span style={{color: C.warn, marginLeft: 26}}>observed {e.observed}</span>
              <span style={{color: C.dim, marginLeft: 26}}>{e.where}</span>
            </div>
          </div>
        );
      })}
      {P.audit && (
        <div style={{position: 'absolute', left: 110, right: 110, top: 946, fontFamily: SANS, fontSize: 27, color: C.muted, opacity: prog(f, 124, 140)}}>
          Model-diff audit, scored separately: {P.audit.confirmed} confirmed, <span style={{color: C.warn}}>{P.audit.refuted} refuted</span>. The refuted ones changed the conclusion.
        </div>
      )}
    </AbsoluteFill>
  );
};
