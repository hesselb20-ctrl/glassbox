import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Card, Heading, Kicker} from '../components';
import {C, MONO, SANS, T, envelope, pct, prog, useSceneFrame} from '../theme';
import {SceneProps} from '../types';

const AgreementGrid: React.FC<{n: number; agree: number; reveal: number; width: number}> = ({n, agree, reveal, width}) => {
  // One cell per held-out input (capped so cells stay visible); mismatches spread evenly and drawn in warn.
  const cells = Math.min(n, 1000);
  const cols = Math.ceil(Math.sqrt(cells * 1.6));
  const rows = Math.ceil(cells / cols);
  const step = width / cols;
  const size = step * 0.78;
  const miss = Math.round((1 - agree) * cells);
  const missEvery = miss > 0 ? cells / miss : Infinity;
  const shown = Math.floor(reveal * cells);
  return (
    <svg width={width} height={rows * step}>
      {Array.from({length: cells}).map((_, i) => {
        const r = Math.floor(i / cols);
        const c = i % cols;
        const isMiss = miss > 0 && Math.floor(i % missEvery) === 0 && i / missEvery < miss;
        const on = i < shown;
        return (
          <rect
            key={i}
            x={c * step}
            y={r * step}
            width={size}
            height={size}
            rx={2}
            fill={on ? (isMiss ? C.warn : C.accent) : C.grid}
            opacity={on ? 0.9 : 0.6}
          />
        );
      })}
    </svg>
  );
};

const Bar: React.FC<{label: string; sub: string; value: number; color: string; grow: number}> = ({label, sub, value, color, grow}) => (
  <div style={{marginTop: 34}}>
    <div style={{display: 'flex', alignItems: 'baseline'}}>
      <div style={{fontFamily: SANS, fontSize: 29, color: C.text}}>{label}</div>
      <div style={{flex: 1}} />
      <div style={{fontFamily: MONO, fontSize: 34, color, fontWeight: 700, opacity: grow > 0.05 ? 1 : 0}}>
        {pct(value * Math.min(1, grow))}
      </div>
    </div>
    <div style={{fontFamily: MONO, fontSize: T.tiny, color: C.muted, marginTop: 4}}>{sub}</div>
    <div style={{marginTop: 12, height: 26, borderRadius: 0, background: C.grid, overflow: 'hidden'}}>
      <div
        style={{
          width: `${Math.max(value * grow * 100, grow > 0 ? 0.6 : 0)}%`,
          height: '100%',
          background: color,
          borderRadius: 0,
          
        }}
      />
    </div>
  </div>
);

export const Verify: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const v = data.modadd?.verification ?? {};
  const n = v.n ?? 0;
  const agree = v.agreement ?? 0;
  const reveal = prog(f, 20, 130);
  const count = Math.round(reveal * agree * n);
  const sd = v.subspace_dims;
  const td = v.total_dims;
  const dims = sd != null && td != null ? `${sd} of ${td} dims` : 'key-frequency subspace';
  const b0 = Math.round(dur * 0.42);
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56}}>
        <Kicker>Verification</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          Run the pseudocode against the model <span style={{color: C.muted}}>on inputs it never saw.</span>
        </Heading>
      </div>

      <Card style={{position: 'absolute', left: 110, top: 230, width: 800, height: 700, padding: 40}}>
        <Kicker>Predicted vs actual · {n} held-out inputs</Kicker>
        <div style={{marginTop: 26}}>
          <AgreementGrid n={n} agree={agree} reveal={reveal} width={720} />
        </div>
        <div style={{display: 'flex', alignItems: 'baseline', gap: 20, marginTop: 26}}>
          <div style={{fontFamily: MONO, fontSize: 72, fontWeight: 700, color: C.accent, textShadow: 'none'}}>
            {pct(reveal * agree)}
          </div>
          <div style={{fontFamily: MONO, fontSize: 30, color: C.text}}>
            {count} / {n} agree
          </div>
        </div>
      </Card>

      <Card style={{position: 'absolute', left: 950, top: 230, width: 860, height: 700, padding: 44, opacity: prog(f, b0 - 20, b0)}}>
        <Kicker>Ablation · the key-frequency subspace</Kicker>
        <div style={{display: 'flex', alignItems: 'baseline', gap: 18, marginTop: 20}}>
          <div style={{fontFamily: MONO, fontSize: 44, fontWeight: 700, color: C.accent, whiteSpace: 'nowrap'}}>{dims}</div>
          <div style={{fontFamily: SANS, fontSize: 28, color: C.muted}}>of the embedding</div>
        </div>
        <Bar
          label="Keep only the subspace"
          sub={dims}
          value={v.acc_only_subspace ?? 0}
          color={C.accent}
          grow={prog(f, b0 + 6, b0 + 46)}
        />
        <Bar
          label="Remove the subspace"
          sub="the behaviour breaks"
          value={v.acc_subspace_removed ?? 0}
          color={C.warn}
          grow={prog(f, b0 + 46, b0 + 86)}
        />
        <Bar
          label="Remove a random subspace"
          sub="control"
          value={v.acc_random_removed ?? 0}
          color={C.muted}
          grow={prog(f, b0 + 86, b0 + 126)}
        />
      </Card>
    </AbsoluteFill>
  );
};
