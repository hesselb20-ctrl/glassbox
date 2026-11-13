import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Card, Heading, Kicker, NotRunYet, Pill} from '../components';
import {C, MONO, SANS, T, envelope, pct, prog, truncate, useSceneFrame} from '../theme';
import {SceneProps} from '../types';

export const demoAvailable = (d: SceneProps['data']) => Boolean(d.demo27b?.available);

const GpuCard: React.FC<{name: string; on: number}> = ({name, on}) => (
  <Card style={{padding: 22, display: 'flex', alignItems: 'center', gap: 22, opacity: on, transform: `translateY(${(1 - on) * 12}px)`}}>
    <svg width={92} height={56}>
      <rect x={2} y={6} width={86} height={44} rx={6} fill="none" stroke={C.muted} strokeWidth={2} />
      <circle cx={30} cy={28} r={13} fill="none" stroke={C.dim} strokeWidth={2} />
      <circle cx={62} cy={28} r={13} fill="none" stroke={C.dim} strokeWidth={2} />
      {[14, 26, 38, 50, 62, 74].map((x) => (
        <rect key={x} x={x} y={50} width={6} height={6} fill={C.dim} />
      ))}
    </svg>
    <div style={{fontFamily: MONO, fontSize: 25, color: C.text}}>{name}</div>
  </Card>
);

const LayerSweep: React.FC<{
  values: number[];
  best?: number;
  steer: number[];
  width: number;
  height: number;
  draw: number;
}> = ({values, best, steer, width, height, draw}) => {
  const n = values.length;
  const padL = 64;
  const padB = 44;
  const w = width - padL;
  const h = height - padB - 10;
  const step = w / n;
  const bw = Math.max(2, step * 0.72);
  const shown = Math.floor(draw * n + 0.0001);
  const yOf = (v: number) => 10 + h - Math.max(0, Math.min(1, v)) * h;
  return (
    <svg width={width} height={height} style={{overflow: 'visible'}}>
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <line x1={padL} x2={width} y1={yOf(t)} y2={yOf(t)} stroke={C.grid} strokeWidth={1.2} />
          <text x={padL - 12} y={yOf(t) + 8} textAnchor="end" fontFamily={MONO} fontSize={20} fill={C.muted}>
            {t}
          </text>
        </g>
      ))}
      {values.slice(0, shown).map((v, i) => {
        const isBest = best === i;
        const isSteer = steer.includes(i);
        return (
          <rect
            key={i}
            x={padL + i * step + (step - bw) / 2}
            y={yOf(v)}
            width={bw}
            height={Math.max(1.5, 10 + h - yOf(v))}
            rx={2}
            fill={isBest ? C.accent : isSteer ? C.muted : C.dim}
          />
        );
      })}
      <text x={padL} y={height - 6} fontFamily={MONO} fontSize={20} fill={C.muted}>
        layer 0
      </text>
      <text x={width} y={height - 6} textAnchor="end" fontFamily={MONO} fontSize={20} fill={C.muted}>
        {n - 1}
      </text>
    </svg>
  );
};

export const Demo27b: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const d = data.demo27b ?? {};
  if (!d.available) {
    return (
      <AbsoluteFill style={{opacity: env}}>
        <NotRunYet
          kicker="27B on a home PC"
          title="A 27B model on two consumer GPUs"
          what="Layer sweep, a direction found, and steering that changes the output. This run has not finished yet."
        />
      </AbsoluteFill>
    );
  }
  const sep = d.separation_by_layer ?? [];
  const best = d.best_layer ?? (sep.length ? sep.indexOf(Math.max(...sep)) : undefined);
  const gens = d.generations ?? [];
  const st = d.steering ?? {};
  const t1 = Math.round(dur * 0.15);
  const t2 = Math.round(dur * 0.5);
  const t3 = Math.round(dur * 0.58);
  const gIdx = gens.length > 1 && f > t3 + (dur - t3) / 2 ? 1 : 0;
  const g = gens[gIdx];
  const gOn = prog(f, t3, t3 + 16) * (gens.length > 1 ? (gIdx === 0 ? 1 - prog(f, t3 + (dur - t3) / 2 - 12, t3 + (dur - t3) / 2) : prog(f, t3 + (dur - t3) / 2, t3 + (dur - t3) / 2 + 12)) : 1);
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56, right: 110}}>
        <Kicker>On a home PC</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          {d.model ?? 'A 27B model'} <span style={{color: C.muted}}>on two consumer GPUs</span>
        </Heading>
        {d.label && (
          <div style={{marginTop: 16}}>
            <Pill color={C.text} size={T.small}>
              {d.label}
            </Pill>
          </div>
        )}
      </div>

      <div style={{position: 'absolute', left: 110, top: 290, width: 560, display: 'flex', flexDirection: 'column', gap: 20}}>
        {(d.gpus ?? []).slice(0, 3).map((gpu, i) => (
          <GpuCard key={gpu} name={gpu} on={prog(f, 8 + i * 12, 26 + i * 12)} />
        ))}
        {d.tag && (
          <div style={{marginTop: 6, opacity: prog(f, 40, 56)}}>
            <Pill color={C.muted}>{d.tag}</Pill>
          </div>
        )}
        {st.alphas && st.metric && st.alphas.length > 0 && (
          <Card style={{marginTop: 18, padding: 24, opacity: prog(f, t3 - 10, t3 + 10)}}>
            <Kicker>{st.metric_name ?? 'metric'} by steering strength</Kicker>
            {st.alphas.slice(0, 6).map((a, i) => {
              const v = st.metric?.[i] ?? 0;
              const mx = Math.max(...(st.metric ?? [1]).map((x) => Math.abs(x))) || 1;
              return (
                <div key={i} style={{display: 'flex', alignItems: 'center', gap: 14, marginTop: 12}}>
                  <div style={{width: 110, fontFamily: MONO, fontSize: 22, color: C.muted}}>alpha {a}</div>
                  <div style={{flex: 1, height: 14, background: C.grid, borderRadius: 0}}>
                    <div style={{width: `${(Math.abs(v) / mx) * 100 * prog(f, t3 + i * 8, t3 + 20 + i * 8)}%`, height: '100%', background: C.muted, borderRadius: 0}} />
                  </div>
                  <div style={{width: 90, textAlign: 'right', fontFamily: MONO, fontSize: 22, color: C.text}}>{v}</div>
                </div>
              );
            })}
          </Card>
        )}
      </div>

      <Card style={{position: 'absolute', left: 720, top: 290, width: 1090, height: 330, padding: 30}}>
        <div style={{display: 'flex', alignItems: 'baseline'}}>
          <Kicker>Layer sweep · probe accuracy</Kicker>
          <div style={{flex: 1}} />
          {best != null && (
            <div style={{fontFamily: MONO, fontSize: 25, color: C.accent, whiteSpace: 'nowrap', opacity: prog(f, t2 - 20, t2)}}>
              direction found · layer {best}
              {sep[best] != null ? ` · ${pct(sep[best])}` : ''}
            </div>
          )}
        </div>
        <div style={{marginTop: 16}}>
          <LayerSweep values={sep} best={best} steer={d.steer_layers ?? []} width={1030} height={230} draw={prog(f, t1, t2 - 24)} />
        </div>
      </Card>

      <div style={{position: 'absolute', left: 720, top: 650, width: 1090, height: 350, opacity: gOn}}>
        {g && (
          <Card style={{width: 1090, height: 350, padding: 30}}>
            <div style={{display: 'flex', alignItems: 'baseline', gap: 16}}>
              <Kicker>Steering</Kicker>
              <div style={{flex: 1}} />
              {g.alpha != null && <Pill color={C.accent}>alpha {g.alpha}</Pill>}
              {st.metric_name && <Pill color={C.muted}>{st.metric_name}</Pill>}
            </div>
            <div style={{fontFamily: MONO, fontSize: 24, color: C.muted, marginTop: 16}}>{truncate(g.prompt, 80)}</div>
            <div style={{display: 'flex', gap: 24, marginTop: 18}}>
              <div style={{flex: 1, border: `1px solid ${C.grid}`, borderRadius: 0, padding: 18}}>
                <div style={{fontFamily: MONO, fontSize: T.tiny, color: C.muted}}>before</div>
                <div style={{fontFamily: SANS, fontSize: 26, color: C.text, marginTop: 8, lineHeight: 1.35}}>{truncate(g.before, 110)}</div>
              </div>
              <div style={{flex: 1, border: `1px solid ${C.accent}`, borderRadius: 0, padding: 18}}>
                <div style={{fontFamily: MONO, fontSize: T.tiny, color: C.accent}}>after</div>
                <div style={{fontFamily: SANS, fontSize: 26, color: C.text, marginTop: 8, lineHeight: 1.35}}>{truncate(g.after, 110)}</div>
              </div>
            </div>
          </Card>
        )}
        {!g && st.alphas && st.metric && (
          <Card style={{width: 1090, height: 350, padding: 30}}>
            <Kicker>Steering · {st.metric_name ?? 'metric'} by alpha</Kicker>
            <div style={{display: 'flex', gap: 30, marginTop: 30, flexWrap: 'wrap'}}>
              {st.alphas.map((a, i) => (
                <div key={i} style={{fontFamily: MONO, fontSize: 26, color: C.text}}>
                  <span style={{color: C.muted}}>{a}:</span> {st.metric?.[i]}
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </AbsoluteFill>
  );
};
