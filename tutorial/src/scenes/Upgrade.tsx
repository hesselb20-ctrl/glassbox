import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Card, Heading, Kicker, NotRunYet, Pill} from '../components';
import {C, MONO, SANS, envelope, glow, prog, useSceneFrame} from '../theme';
import {SceneProps} from '../types';

// Three short scenes: a sealed bench task is solved, a finding is checked against its null distribution,
// and an edit between two versions of a model is located. Every number comes from data.json.

const Stat: React.FC<{label: string; value: string; color?: string; on: number}> = ({label, value, color = C.text, on}) => (
  <div style={{opacity: on, transform: `translateY(${(1 - on) * 12}px)`}}>
    <div style={{fontFamily: MONO, fontSize: 22, color: C.muted}}>{label}</div>
    <div style={{fontFamily: MONO, fontSize: 46, fontWeight: 700, color, marginTop: 4}}>{value}</div>
  </div>
);

// ------------------------------------------------------------------------------------------ sealed bench task
export const BenchScene: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const B = data.bench ?? {};
  const ex = B.example;
  if (!B.available || !ex) {
    return (
      <AbsoluteFill style={{opacity: env}}>
        <NotRunYet kicker="glassbox-bench" title="Planted circuits with sealed answer keys" what="No graded seed yet." />
      </AbsoluteFill>
    );
  }
  const L = ex.n_layers ?? 0;
  const H = ex.n_heads ?? 0;
  const claimed = new Set(ex.claimed ?? []);
  const truth = new Set(ex.true ?? []);
  const cell = 92;
  const gap = 14;
  const found = prog(f, dur * 0.28, dur * 0.5); // the scientist's claim lights up
  const opened = prog(f, dur * 0.58, dur * 0.7); // the key is opened
  const names: string[][] = [];
  for (let l = 0; l < L; l++) names.push([...Array.from({length: H}, (_, h) => `${l}.${h}`), `mlp${l}`]);
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56, right: 110}}>
        <Kicker>glassbox-bench · seed {ex.seed} · {ex.family}</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          A circuit is planted. <span style={{color: C.muted}}>The answer key is sealed until grading.</span>
        </Heading>
      </div>
      <Card style={{position: 'absolute', left: 110, top: 230, width: 800, height: 570, padding: 34}}>
        <div style={{fontFamily: MONO, fontSize: 22, color: C.muted}}>the model the scientist gets: rotated, with distractors</div>
        <div style={{marginTop: 26}}>
          {names
            .slice()
            .reverse()
            .map((row, ri) => (
              <div key={ri} style={{display: 'flex', gap, marginBottom: gap, alignItems: 'center'}}>
                <div style={{fontFamily: MONO, fontSize: 22, color: C.muted, width: 96}}>layer {L - 1 - ri}</div>
                {row.map((n) => {
                  const isClaim = claimed.has(n);
                  const isTrue = truth.has(n);
                  const lit = isClaim ? found : 0;
                  const mlp = n.startsWith('mlp');
                  return (
                    <div
                      key={n}
                      style={{
                        width: mlp ? cell * 1.5 : cell,
                        height: cell,
                        marginLeft: mlp ? 18 : 0,
                        borderRadius: mlp ? 14 : cell / 2,
                        border: `2px solid ${lit > 0.5 ? C.accent : C.dim}`,
                        background: lit > 0 ? `rgba(61,245,208,${0.16 * lit})` : 'transparent',
                        boxShadow: lit > 0.5 ? glow(C.accent, 12 * lit) : 'none',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontFamily: MONO,
                        fontSize: 22,
                        color: lit > 0.5 ? C.accent : C.muted,
                        position: 'relative',
                      }}
                    >
                      {n}
                      {isTrue && (
                        <div
                          style={{
                            position: 'absolute',
                            inset: -9,
                            borderRadius: mlp ? 20 : cell,
                            border: `2px dashed ${C.text}`,
                            opacity: opened,
                          }}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
        </div>
        <div style={{display: 'flex', gap: 26, marginTop: 14, fontFamily: MONO, fontSize: 22, color: C.muted}}>
          <span style={{opacity: found}}>
            <span style={{color: C.accent}}>filled</span> claimed by the scientist
          </span>
          <span style={{opacity: opened}}>
            <span style={{color: C.text}}>dashed</span> the sealed key
          </span>
        </div>
      </Card>
      <div style={{position: 'absolute', left: 980, top: 236, width: 830}}>
        <div style={{fontFamily: MONO, fontSize: 22, color: C.muted, opacity: prog(f, 10, 26)}}>planted algorithm (RASP, compiled with Tracr)</div>
        <Card style={{marginTop: 12, padding: 22, opacity: prog(f, 10, 26)}}>
          <div style={{fontFamily: MONO, fontSize: 24, color: opened > 0.5 ? C.text : C.dim, filter: `blur(${(1 - opened) * 9}px)`}}>{ex.rasp ?? ''}</div>
          <div style={{marginTop: 12}}>
            <Pill color={opened > 0.5 ? C.accent : C.warn}>{opened > 0.5 ? 'key opened for grading' : 'sealed outside the repo'}</Pill>
          </div>
        </Card>
        <div style={{display: 'flex', gap: 54, marginTop: 44}}>
          <Stat label="circuit recall" value={(ex.recall ?? 0).toFixed(2)} color={C.accent} on={prog(f, dur * 0.7, dur * 0.76)} />
          <Stat label="precision" value={(ex.precision ?? 0).toFixed(2)} color={C.accent} on={prog(f, dur * 0.73, dur * 0.79)} />
          <Stat label="mechanism match" value={(ex.mechanism_match ?? 0).toFixed(2)} color={C.accent} on={prog(f, dur * 0.76, dur * 0.82)} />
        </div>
        <div style={{fontFamily: SANS, fontSize: 28, color: C.text, marginTop: 40, lineHeight: 1.4, opacity: prog(f, dur * 0.8, dur * 0.86)}}>
          {ex.experiments} experiments, {ex.minutes} minutes. Over {B.seeds} seeds: {B.compiled_solved} of {B.compiled_tasks} planted circuits recovered.
        </div>
        <div style={{fontFamily: SANS, fontSize: 24, color: C.muted, marginTop: 14, lineHeight: 1.4, opacity: prog(f, dur * 0.84, dur * 0.9)}}>
          Every task at this difficulty was solved, so the bench is at its ceiling. The seal is an honour system on one PC.
        </div>
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------------------------------ null distribution
export const NullScene: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const N = data.nulls ?? {};
  const panels = N.panels ?? [];
  if (!N.available || panels.length === 0) {
    return (
      <AbsoluteFill style={{opacity: env}}>
        <NotRunYet kicker="Null controls" title="Is the effect bigger than chance?" what="No null calibration yet." />
      </AbsoluteFill>
    );
  }
  const W = 540;
  const Hh = 380;
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56, right: 110}}>
        <Kicker>Null controls</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          A finding counts only <span style={{color: C.accent}}>above the 95th percentile</span> of its nulls.
        </Heading>
      </div>
      {panels.slice(0, 3).map((p, i) => {
        const counts = p.counts ?? [];
        const lo = p.lo ?? 0;
        const hi = p.hi ?? 1;
        const max = Math.max(...counts, 1);
        const x = (v: number) => ((v - lo) / (hi - lo)) * W;
        const bw = W / counts.length;
        const grow = prog(f, 14 + i * 8, 50 + i * 8);
        const line95 = prog(f, dur * 0.36, dur * 0.46);
        const eff = prog(f, dur * 0.5 + i * 6, dur * 0.62 + i * 6);
        return (
          <Card key={i} style={{position: 'absolute', left: 110 + i * 580, top: 240, width: 550, height: 590, padding: 0, overflow: 'hidden'}}>
            <div style={{padding: '24px 28px 0'}}>
              <div style={{fontFamily: SANS, fontSize: 30, fontWeight: 600, color: C.text}}>{p.name}</div>
              <div style={{fontFamily: MONO, fontSize: 22, color: C.muted, marginTop: 4}}>{p.n} null samples</div>
            </div>
            <svg width={550} height={Hh + 60} style={{position: 'absolute', left: 5, top: 120}}>
              {counts.map((c, k) => (
                <rect key={k} x={k * bw + 1} y={Hh - (c / max) * (Hh - 40) * grow} width={bw - 2} height={(c / max) * (Hh - 40) * grow} fill={C.dim} />
              ))}
              <line x1={0} x2={W} y1={Hh} y2={Hh} stroke={C.grid} strokeWidth={2} />
              <line x1={x(p.p95 ?? 0)} x2={x(p.p95 ?? 0)} y1={Hh - (Hh - 10) * line95} y2={Hh} stroke={C.muted} strokeWidth={2.5} strokeDasharray="8 6" />
              <line x1={x(p.effect ?? 0)} x2={x(p.effect ?? 0)} y1={Hh - (Hh - 10) * eff} y2={Hh} stroke={C.accent} strokeWidth={5} />
              <text x={x(p.p95 ?? 0) + 10} y={26} fill={C.muted} fontFamily={MONO} fontSize={22} opacity={line95}>
                p95 {(p.p95 ?? 0).toFixed(2)}
              </text>
              <text x={x(p.effect ?? 0) - 12} y={70} fill={C.accent} fontFamily={MONO} fontSize={26} fontWeight={700} textAnchor="end" opacity={eff}>
                circuit {(p.effect ?? 0).toFixed(2)}
              </text>
              <text x={W / 2} y={Hh + 40} fill={C.muted} fontFamily={MONO} fontSize={22} textAnchor="middle">
                drop in the metric when ablated
              </text>
            </svg>
          </Card>
        );
      })}
      <div style={{position: 'absolute', left: 110, right: 110, top: 876, fontFamily: SANS, fontSize: 30, color: C.text, opacity: prog(f, dur * 0.72, dur * 0.8)}}>
        Grey: random component sets of the same size, and the same circuit in models with random or shuffled weights.
        {N.null_runs ? (
          <span style={{color: C.muted}}>
            {' '}
            Full investigations repeated on null models: {N.null_runs}. Counted as a discovery: {N.false_discoveries}
            {N.false_discoveries === 1 ? ' (shuffled labels on real weights; it found the real circuit).' : '.'}
          </span>
        ) : null}
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------------------------------ model diff
export const AuditScene: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const A = data.audit ?? {};
  if (!A.available) {
    return (
      <AbsoluteFill style={{opacity: env}}>
        <NotRunYet kicker="Auditing a model edit" title="Where do two versions of a model differ?" what="The audit has not finished yet." />
      </AbsoluteFill>
    );
  }
  const roles: [string, string][] = [
    ['attn_out', 'attention out'],
    ['mlp_out', 'MLP out'],
    ['attn_in', 'attention in'],
    ['mlp_in', 'MLP in'],
  ];
  const nL = (A.over_floor?.mlp_out ?? []).length || 1;
  const W = 1470;
  const cw = W / nL;
  // colour: multiple of the matrix's own quantization noise floor, log scale from 1x to 100x
  const shade = (e: number | null | undefined) => {
    if (e == null) return C.panel;
    const t = Math.max(0, Math.min(1, Math.log10(Math.max(e, 1)) / 2));
    return e > 5 ? `rgba(61,245,208,${0.3 + 0.7 * t})` : `rgba(58,71,128,${0.28 + 0.9 * t})`;
  };
  const sweep = prog(f, dur * 0.14, dur * 0.5) * nL;
  const bracket = prog(f, dur * 0.5, dur * 0.58);
  const x0 = 230 + (A.edited_from ?? 0) * cw;
  const x1 = 230 + ((A.edited_to ?? 0) + 1) * cw;
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56, right: 110}}>
        <Kicker>Auditing a model edit · {A.label}</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          Two versions of a 27B model. <span style={{color: C.muted}}>Where is the edit?</span>
        </Heading>
      </div>
      <div style={{position: 'absolute', left: 110, top: 212, fontFamily: MONO, fontSize: 22, color: C.muted, opacity: prog(f, 8, 22)}}>
        {A.matrices} weight matrices compared one by one · bright = structured difference, more than 5x the quantization noise floor
      </div>
      <div style={{position: 'absolute', left: 110, top: 300}}>
        {roles.map(([key, label], ri) => (
          <div key={key} style={{display: 'flex', alignItems: 'center', marginBottom: 10}}>
            <div style={{width: 230, fontFamily: MONO, fontSize: 24, color: ri < 2 ? C.text : C.muted}}>{label}</div>
            <div style={{display: 'flex'}}>
              {(A.over_floor?.[key] ?? []).map((e, l) => (
                <div key={l} style={{width: cw - 2, height: 70, marginRight: 2, borderRadius: 3, background: l < sweep ? shade(e) : C.panel}} />
              ))}
            </div>
          </div>
        ))}
        <div style={{marginLeft: 230, width: W, display: 'flex', justifyContent: 'space-between', fontFamily: MONO, fontSize: 22, color: C.muted}}>
          <span>layer 0</span>
          <span>layer {nL - 1}</span>
        </div>
        <div
          style={{
            position: 'absolute',
            left: x0 - 4,
            width: x1 - x0 + 4,
            top: -14,
            height: 178,
            border: `2px solid ${C.accent}`,
            borderRadius: 8,
            opacity: bracket,
            boxShadow: glow(C.accent, 10),
          }}
        />
        <div style={{position: 'absolute', left: x0, top: -50, fontFamily: MONO, fontSize: 24, color: C.accent, opacity: bracket, whiteSpace: 'nowrap'}}>
          {A.edited_matrices} matrices, layers {A.edited_from} to {A.edited_to}: one shared direction
        </div>
      </div>
      <div style={{position: 'absolute', left: 110, right: 110, top: 706, display: 'flex', gap: 64}}>
        <Stat label="refusal rate, official" value={(A.refusal_official ?? 0).toFixed(2)} on={prog(f, dur * 0.56, dur * 0.62)} />
        <Stat label="refusal rate, edited" value={(A.refusal_abliterated ?? 0).toFixed(2)} color={C.warn} on={prog(f, dur * 0.59, dur * 0.65)} />
        <Stat
          label={`edit direction vs refusal direction, layer ${A.cosine_layer}`}
          value={`cos ${(A.cosine ?? 0).toFixed(2)}`}
          color={A.beats_nulls ? C.accent : C.warn}
          on={prog(f, dur * 0.66, dur * 0.72)}
        />
        <Stat label="null p95" value={(A.cosine_null ?? 0).toFixed(2)} color={C.muted} on={prog(f, dur * 0.69, dur * 0.75)} />
      </div>
      <div style={{position: 'absolute', left: 110, right: 110, top: 868, fontFamily: SANS, fontSize: 28, color: C.text, lineHeight: 1.4, opacity: prog(f, dur * 0.76, dur * 0.84)}}>
        {A.refuted} of {(A.confirmed ?? 0) + (A.refuted ?? 0)} pre-registered predictions were refuted: the edit is not everywhere. The location comes
        from an exploratory follow-up.{' '}
        <span style={{color: C.muted}}>Measurement only: no completions stored, no weights changed.</span>
      </div>
    </AbsoluteFill>
  );
};
