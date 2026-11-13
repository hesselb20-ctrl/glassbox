import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Card, Circuit, Cube, Heading, Kicker, NotRunYet, Pill} from '../components';
import {C, MONO, SANS, T, envelope, prog, truncate, useSceneFrame} from '../theme';
import {SceneProps} from '../types';
import {TAGLINE} from './GlassReveal';
import {circuitFreqs} from './shared';

// ---------------------------------------------------------------- benchmark scorecard
export const Scorecard: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur, 12, 12);
  const sc = data.scorecard ?? {};
  const rows = sc.rows ?? [];
  if (!sc.available || !rows.length) {
    return (
      <AbsoluteFill style={{opacity: env}}>
        <NotRunYet kicker="Blind benchmarks" title="Benchmark scorecard" what="The blind benchmark runs have not finished yet." />
      </AbsoluteFill>
    );
  }
  const ioi = sc.ioi;
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56}}>
        <Kicker>Blind benchmarks</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          Known circuits, rediscovered <span style={{color: C.muted}}>without hints.</span>
        </Heading>
      </div>
      <div style={{position: 'absolute', left: 110, top: 220, width: 1700}}>
        <div
          style={{
            display: 'flex',
            fontFamily: MONO,
            fontSize: T.tiny,
            color: C.muted,
            padding: '0 32px 14px',
            letterSpacing: 1,
            textTransform: 'uppercase',
          }}
        >
          <div style={{flex: 1}}>Benchmark</div>
          <div style={{width: 470}}>found / partial / missed</div>
          <div style={{width: 170, textAlign: 'right'}}>Experiments</div>
          <div style={{width: 150, textAlign: 'right'}}>Minutes</div>
        </div>
        {rows.slice(0, 5).map((r, i) => {
          const on = prog(f, 8 + i * 10, 24 + i * 10);
          const total = r.total ?? (r.found ?? 0) + (r.partial ?? 0) + (r.missed ?? 0);
          const cells = [
            ...Array(r.found ?? 0).fill(C.accent),
            ...Array(r.partial ?? 0).fill(C.muted),
            ...Array(r.missed ?? 0).fill(C.warn),
          ].slice(0, 12);
          return (
            <Card
              key={i}
              style={{
                display: 'flex',
                alignItems: 'center',
                padding: '24px 32px',
                marginBottom: 18,
                opacity: on,
                transform: `translateX(${(1 - on) * -20}px)`,
              }}
            >
              <div style={{flex: 1}}>
                <div style={{fontFamily: SANS, fontSize: 30, color: C.text}}>{truncate(r.title ?? r.name, 52)}</div>
                {r.status && (
                  <div style={{marginTop: 8}}>
                    <Pill color={r.status === 'verified' ? C.accent : C.muted} size={20}>
                      {r.status}
                    </Pill>
                  </div>
                )}
              </div>
              <div style={{width: 470, display: 'flex', alignItems: 'center', gap: 8}}>
                {cells.map((c, k) => (
                  <div key={k} style={{width: 30, height: 30, borderRadius: 0, background: c, opacity: prog(f, 20 + i * 10 + k * 2, 28 + i * 10 + k * 2)}} />
                ))}
                <div style={{fontFamily: MONO, fontSize: 25, color: C.text, marginLeft: 14}}>
                  {r.found ?? 0}/{r.partial ?? 0}/{r.missed ?? 0}
                  <span style={{color: C.muted}}> of {total}</span>
                </div>
              </div>
              <div style={{width: 170, textAlign: 'right', fontFamily: MONO, fontSize: 30, color: C.text}}>{r.experiments ?? '-'}</div>
              <div style={{width: 150, textAlign: 'right', fontFamily: MONO, fontSize: 30, color: C.text}}>{r.minutes ?? '-'}</div>
            </Card>
          );
        })}
        {ioi?.found != null && ioi?.published != null && (
          <div style={{fontFamily: MONO, fontSize: 25, color: C.muted, marginTop: 10, padding: '0 32px', opacity: prog(f, 50, 66)}}>
            IOI: <span style={{color: C.accent}}>{ioi.found}</span> of {ioi.published} published heads found
          </div>
        )}
      </div>
      <div
        style={{
          position: 'absolute',
          left: 110,
          right: 110,
          top: 900,
          fontFamily: SANS,
          fontSize: 34,
          color: C.text,
          opacity: prog(f, 40, 58),
        }}
      >
        blind = no hints given; every claim backed by an experiment
      </div>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- $0 card
export const ZeroCost: React.FC<SceneProps> = ({dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur, 12, 12);
  const items = ['your own GPUs', 'NDIF', 'your Claude Code subscription'];
  return (
    <AbsoluteFill style={{opacity: env, alignItems: 'center', justifyContent: 'center', flexDirection: 'column'}}>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 230,
          fontWeight: 700,
          color: C.accent,
          textShadow: 'none',
          lineHeight: 1,
          opacity: prog(f, 4, 22),
          transform: `scale(${0.96 + 0.04 * prog(f, 4, 30)})`,
        }}
      >
        $0
      </div>
      <div style={{fontFamily: SANS, fontSize: 40, color: C.muted, marginTop: 20, opacity: prog(f, 18, 34)}}>
        extra to run the lab
      </div>
      <div style={{display: 'flex', alignItems: 'center', gap: 26, marginTop: 64}}>
        {items.map((it, i) => (
          <React.Fragment key={it}>
            {i > 0 && <div style={{fontFamily: MONO, fontSize: 40, color: C.dim, opacity: prog(f, 30 + i * 12, 42 + i * 12)}}>+</div>}
            <Card style={{padding: '22px 34px', opacity: prog(f, 30 + i * 12, 44 + i * 12)}}>
              <div style={{fontFamily: SANS, fontSize: 34, color: C.text}}>{it}</div>
            </Card>
          </React.Fragment>
        ))}
      </div>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- end card
export const EndCard: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur, 14, 0);
  const freqs = circuitFreqs(data);
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 150, top: 300}}>
        <Cube w={480} h={330} glass={1}>
          <Circuit lit={5} width={420} freqs={freqs} labels={false} shimmer={1} />
        </Cube>
      </div>
      <div style={{position: 'absolute', left: 820, top: 300, width: 1000}}>
        <div style={{fontFamily: MONO, fontSize: 110, fontWeight: 700, color: C.text, letterSpacing: -2, lineHeight: 1}}>
          glass<span style={{color: C.accent, textShadow: 'none'}}>box</span>
        </div>
        <div style={{fontFamily: SANS, fontSize: 40, color: C.muted, marginTop: 18}}>{TAGLINE}</div>
        <div
          style={{
            marginTop: 56,
            background: C.navy,
            border: `1px solid ${C.grid}`,
            borderRadius: 0,
            padding: '22px 26px',
            fontFamily: MONO,
            fontSize: 25,
            color: C.text,
            lineHeight: 1.6,
            opacity: prog(f, 16, 32),
          }}
        >
          {(data.install ?? '').split(/\s&&\s/).map((part, i) => (
            <div key={i}>
              <span style={{color: C.accent}}>$ </span>
              {part}
            </div>
          ))}
        </div>
        {data.repo && (
          <div style={{fontFamily: MONO, fontSize: 36, color: C.accent, marginTop: 40, opacity: prog(f, 30, 46)}}>{data.repo}</div>
        )}
      </div>
    </AbsoluteFill>
  );
};
