import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Card, Heading, Kicker, Pill} from '../components';
import {C, MONO, SANS, T, envelope, prog, truncate, useSceneFrame} from '../theme';
import {SceneProps} from '../types';
import {FourierBars} from './FourierBars';
import {firstId} from './shared';

// Stage start frames (hypothesis, experiment, plot, update), as a fraction of the scene length.
const STAGES = [0.03, 0.26, 0.44, 0.74];

type Line = {t: string; at: number; kind?: 'prompt' | 'call' | 'out' | 'cont' | 'muted'};

const Terminal: React.FC<{f: number; lines: Line[]; style?: React.CSSProperties}> = ({f, lines, style}) => {
  const visible = lines.filter((l) => f >= l.at);
  const last = visible[visible.length - 1];
  return (
    <div
      style={{
        background: '#050816',
        border: `1.5px solid ${C.grid}`,
        borderRadius: 16,
        overflow: 'hidden',
        boxSizing: 'border-box',
        ...style,
      }}
    >
      <div
        style={{
          height: 50,
          background: C.panel,
          borderBottom: `1.5px solid ${C.grid}`,
          display: 'flex',
          alignItems: 'center',
          padding: '0 20px',
          gap: 10,
        }}
      >
        {[0, 1, 2].map((i) => (
          <div key={i} style={{width: 14, height: 14, borderRadius: 7, background: C.dim}} />
        ))}
        <div style={{fontFamily: MONO, fontSize: T.tiny, color: C.muted, marginLeft: 14}}>claude code · glassbox MCP</div>
      </div>
      <div style={{padding: '22px 26px', fontFamily: MONO, fontSize: 23, lineHeight: '37px', whiteSpace: 'pre'}}>
        {visible.map((l, i) => {
          const n = Math.floor((f - l.at) * 2.6);
          const txt = l.t.slice(0, Math.max(0, n));
          const color =
            l.kind === 'prompt' ? C.text : l.kind === 'out' ? C.accent : l.kind === 'muted' ? C.muted : C.text;
          const bullet = l.kind === 'call' ? txt.slice(0, 1) : '';
          return (
            <div key={i} style={{color, opacity: l.kind === 'cont' ? 0.85 : 1}}>
              {bullet ? (
                <>
                  <span style={{color: C.accent}}>{bullet}</span>
                  {txt.slice(1)}
                </>
              ) : (
                txt
              )}
              {l === last && n < l.t.length + 8 && Math.floor(f / 8) % 2 === 0 ? (
                <span style={{color: C.muted}}>▌</span>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const LoopNodes: React.FC<{active: number; width: number}> = ({active, width}) => {
  const names = ['hypothesis', 'experiment', 'plot', 'update'];
  const step = width / names.length;
  return (
    <svg width={width} height={96} style={{overflow: 'visible'}}>
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={C.dim} />
        </marker>
      </defs>
      {names.map((nm, i) => {
        const cx = step * i + step / 2;
        const on = i === active;
        const past = i < active;
        return (
          <g key={nm}>
            {i < names.length - 1 && (
              <line x1={cx + 86} x2={cx + step - 92} y1={32} y2={32} stroke={C.dim} strokeWidth={2} markerEnd="url(#arr)" />
            )}
            <rect
              x={cx - 82}
              y={10}
              width={164}
              height={44}
              rx={22}
              fill={on ? C.accent : C.panel}
              stroke={on ? C.accent : past ? C.muted : C.dim}
              strokeWidth={1.8}
            />
            <text
              x={cx}
              y={39}
              textAnchor="middle"
              fontFamily={MONO}
              fontSize={21}
              fill={on ? C.navy : past ? C.text : C.muted}
              fontWeight={on ? 700 : 400}
            >
              {nm}
            </text>
          </g>
        );
      })}
      <path
        d={`M ${width - step / 2} 56 C ${width - step / 2} 92, ${step / 2} 92, ${step / 2} 58`}
        fill="none"
        stroke={C.dim}
        strokeWidth={2}
        strokeDasharray="5 7"
        markerEnd="url(#arr)"
      />
    </svg>
  );
};

export const ResearchLoop: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const m = data.modadd ?? {};
  const S = STAGES.map((s) => Math.round(s * dur));
  const p = m.p;
  const expId = firstId(m.result) ?? 'E0003';
  const revId = firstId(m.revised_example?.text);
  const backend = (m.result ?? '').match(/\((local-full|local-4bit|local-gguf|ndif)\)/)?.[1] ?? 'local-full';
  const top3 = (m.top ?? []).slice(0, 3).map((t) => t.frequency);
  const hypName = (m.hypothesis ?? '').split(':')[0];

  const lines: Line[] = [
    {t: '> /investigate modular addition', at: S[0] - 6, kind: 'prompt'},
    {t: `● load_model(backend="${backend}",`, at: S[0] + 18, kind: 'call'},
    {t: '    model="benchmarks/modadd/model")', at: S[0] + 30, kind: 'cont'},
    {t: '  ⎿ investigation opened', at: S[0] + 52, kind: 'out'},
    {t: '● notebook_append(kind="hypothesis",', at: S[0] + 76, kind: 'call'},
    {t: `    text="${truncate(hypName, 26)}…")`, at: S[0] + 88, kind: 'cont'},
    {t: '● run_experiment("periodicity",', at: S[1] + 4, kind: 'call'},
    {t: `    {"source": "W_E", "period": ${p ?? 'p'}})`, at: S[1] + 16, kind: 'cont'},
    {t: `  ⎿ ${expId} · top k: ${top3.join(', ')}`, at: S[2] - 26, kind: 'out'},
    {t: `● get_plot("${expId}")`, at: S[2] + 4, kind: 'call'},
    {t: `● notebook_append(kind="result", refs=["${expId}"])`, at: S[2] + 120, kind: 'call'},
    {t: '  ⋮', at: S[3] - 10, kind: 'muted'},
  ];
  if (revId) {
    lines.push({t: '● run_experiment("ablation", {"mode": "mean"})', at: S[3] + 6, kind: 'call'});
    lines.push({t: `  ⎿ ${revId}`, at: S[3] + 40, kind: 'out'});
  }
  if (m.revised_example?.target) {
    lines.push({t: `● notebook_append(target="${m.revised_example.target}",`, at: S[3] + 62, kind: 'call'});
    lines.push({t: '    status="revised")', at: S[3] + 74, kind: 'cont'});
  }

  const active = f < S[1] ? 0 : f < S[2] ? 1 : f < S[3] ? 2 : 3;
  const vis = (i: number) =>
    prog(f, S[i], S[i] + 16) * (i < 3 ? 1 - prog(f, S[i + 1] - 12, S[i + 1] + 2) : 1);

  const cardStyle: React.CSSProperties = {position: 'absolute', left: 0, top: 0, width: 860, height: 690, padding: 38};
  const [hypMain, hypAgainst] = (m.hypothesis ?? '').split(/\s(?=Against:)/);
  const freqs = m.frequencies ?? [];
  const share = m.share ?? [];
  const keys = m.key_frequencies ?? [];
  const runP = prog(f, S[1] + 20, S[2] - 30);

  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56}}>
        <Kicker>The research loop</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          Claude Code is the scientist. <span style={{color: C.muted}}>glassbox is the lab.</span>
        </Heading>
      </div>

      <Terminal f={f} lines={lines} style={{position: 'absolute', left: 110, top: 200, width: 800, height: 800}} />

      <div style={{position: 'absolute', left: 950, top: 196}}>
        <LoopNodes active={active} width={860} />
      </div>

      <div style={{position: 'absolute', left: 950, top: 310, width: 860, height: 690}}>
        {/* 1. hypothesis */}
        <Card style={{...cardStyle, opacity: vis(0)}}>
          <Kicker>Hypothesis · lab notebook</Kicker>
          <div style={{fontFamily: SANS, fontSize: 31, lineHeight: 1.45, color: C.text, marginTop: 22}}>
            {truncate(hypMain, 230)}
          </div>
          {hypAgainst && (
            <div style={{fontFamily: SANS, fontSize: 28, lineHeight: 1.45, color: C.muted, marginTop: 24}}>
              {truncate(hypAgainst, 150)}
            </div>
          )}
        </Card>

        {/* 2. experiment */}
        <Card style={{...cardStyle, opacity: vis(1)}}>
          <Kicker>Experiment</Kicker>
          <div style={{fontFamily: MONO, fontSize: 38, color: C.text, marginTop: 26}}>
            run_experiment(<span style={{color: C.accent}}>"periodicity"</span>)
          </div>
          <div style={{fontFamily: MONO, fontSize: 26, color: C.muted, marginTop: 18, lineHeight: 1.6}}>
            source: W_E (token embedding)
            <br />
            period: {p ?? 'p'}
          </div>
          <div style={{marginTop: 50, height: 10, borderRadius: 5, background: C.grid, overflow: 'hidden'}}>
            <div style={{width: `${runP * 100}%`, height: '100%', background: C.muted}} />
          </div>
          <div style={{marginTop: 26, display: 'flex', gap: 16, alignItems: 'center', opacity: prog(f, S[2] - 34, S[2] - 22)}}>
            <Pill color={C.accent} size={T.small}>
              {expId}
            </Pill>
            {m.tag && (
              <Pill color={C.muted} size={T.small}>
                {m.tag}
              </Pill>
            )}
          </div>
          <div style={{fontFamily: SANS, fontSize: 28, color: C.muted, marginTop: 34, lineHeight: 1.45}}>
            Every run is logged in the notebook with its id; every later claim cites one.
          </div>
        </Card>

        {/* 3. plot */}
        <Card style={{...cardStyle, opacity: vis(2)}}>
          <Kicker>get_plot · W_E Fourier power by frequency</Kicker>
          <div style={{marginTop: 18}}>
            <FourierBars
              freqs={freqs}
              share={share}
              keys={keys}
              labelTop={top3}
              width={784}
              height={340}
              grow={(i) => prog(f, S[2] + 8 + i * 1.2, S[2] + 34 + i * 1.2)}
            />
          </div>
          <div style={{display: 'flex', gap: 14, alignItems: 'center', marginTop: 10}}>
            <div style={{width: 18, height: 18, borderRadius: 4, background: C.accent}} />
            <div style={{fontFamily: MONO, fontSize: T.tiny, color: C.muted}}>key frequencies ({keys.length})</div>
          </div>
          <div
            style={{
              fontFamily: SANS,
              fontSize: 25,
              lineHeight: 1.42,
              color: C.text,
              marginTop: 18,
              opacity: prog(f, S[2] + 90, S[2] + 110),
            }}
          >
            {truncate(m.result, 250)}
          </div>
        </Card>

        {/* 4. update */}
        <Card style={{...cardStyle, opacity: vis(3)}} border={C.muted}>
          <div style={{display: 'flex', gap: 14, alignItems: 'center'}}>
            <Kicker>Hypothesis updated</Kicker>
            <div style={{flex: 1}} />
            {m.revised_example?.target && <Pill color={C.muted}>{m.revised_example.target}</Pill>}
            <Pill color={C.accent}>revised</Pill>
          </div>
          <div style={{fontFamily: SANS, fontSize: 29, lineHeight: 1.45, color: C.text, marginTop: 26}}>
            {truncate(m.revised_example?.text, 430)}
          </div>
        </Card>
      </div>
    </AbsoluteFill>
  );
};
