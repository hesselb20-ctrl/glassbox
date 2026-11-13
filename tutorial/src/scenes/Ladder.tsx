import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Card, Heading, Kicker, NotRunYet, Pill} from '../components';
import {C, MONO, SANS, envelope, prog, truncate, useSceneFrame} from '../theme';
import {SceneProps} from '../types';

const statusColor = (s?: string) => {
  const v = (s ?? '').toLowerCase();
  if (/fail|miss|broke|error/.test(v)) return C.warn;
  if (/verified|found|ok|pass|done/.test(v) && !/not/.test(v)) return C.accent;
  return C.muted;
};

export const Ladder: React.FC<SceneProps> = ({data, dur, frame}) => {
  const f = useSceneFrame(frame);
  const env = envelope(f, dur);
  const L = data.ladder ?? {};
  const rungs = L.rungs ?? [];
  if (!L.available || rungs.length === 0) {
    return (
      <AbsoluteFill style={{opacity: env}}>
        <NotRunYet
          kicker="Scale ladder"
          title="The same circuit, from a small model to 27B"
          what="One rung per model: small local model, 27B on the home PC, NDIF at full precision. This run has not finished yet."
        />
      </AbsoluteFill>
    );
  }
  const n = Math.min(rungs.length, 5);
  const gap = 24;
  const W = 1700;
  const cw = (W - gap * (n - 1)) / n;
  const rise = n > 1 ? Math.min(60, 160 / (n - 1)) : 0;
  const per = (dur * 0.6) / n;
  return (
    <AbsoluteFill style={{opacity: env}}>
      <div style={{position: 'absolute', left: 110, top: 56, right: 110}}>
        <Kicker>Scale ladder{L.behavior ? ` · ${L.behavior}` : ''}</Kicker>
        <Heading style={{marginTop: 10}} size={44}>
          Four predictions from the small circuit, <span style={{color: C.muted}}>tested up the ladder.</span>
        </Heading>
      </div>
      {rungs.slice(0, n).map((r, i) => {
        const on = prog(f, 16 + i * per, 34 + i * per);
        const sc = statusColor(r.status);
        return (
          <Card
            key={i}
            style={{
              position: 'absolute',
              left: 110 + i * (cw + gap),
              top: 230 + (n - 1 - i) * rise,
              width: cw,
              height: 500,
              overflow: 'hidden',
              padding: 26,
              opacity: on,
              transform: `translateY(${(1 - on) * 18}px)`,
            }}
            border={sc === C.accent ? C.accent : C.grid}
          >
            <div style={{fontFamily: MONO, fontSize: 22, color: C.muted}}>rung {i + 1}</div>
            <div style={{fontFamily: SANS, fontSize: n > 3 ? 28 : 32, fontWeight: 600, color: C.text, marginTop: 8, lineHeight: 1.2}}>
              {truncate(r.model, 40)}
            </div>
            {r.params != null && <div style={{fontFamily: MONO, fontSize: 24, color: C.muted, marginTop: 6}}>{r.params}</div>}
            <div style={{display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 16}}>
              {r.backend && <Pill color={C.muted}>{r.backend}</Pill>}
              {r.tag && <Pill color={C.muted}>{truncate(r.tag, 28)}</Pill>}
              {r.status && <Pill color={sc}>{r.status}</Pill>}
            </div>
            {r.key_result && (
              <div style={{fontFamily: SANS, fontSize: 25, color: C.text, marginTop: 18, lineHeight: 1.38}}>
                {truncate(r.key_result, n > 3 ? 110 : 150)}
              </div>
            )}
            {r.heads && r.heads.length > 0 && (
              <div style={{display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 16}}>
                {r.heads.slice(0, 8).map((h) => (
                  <Pill key={h} color={C.accent} size={20}>
                    {h}
                  </Pill>
                ))}
              </div>
            )}
          </Card>
        );
      })}
      {L.conclusion && (
        <div
          style={{
            position: 'absolute',
            left: 110,
            right: 110,
            top: 930,
            fontFamily: SANS,
            fontSize: 32,
            color: C.text,
            opacity: prog(f, dur * 0.7, dur * 0.7 + 20),
          }}
        >
          {truncate(L.conclusion, 140)}
        </div>
      )}
    </AbsoluteFill>
  );
};
