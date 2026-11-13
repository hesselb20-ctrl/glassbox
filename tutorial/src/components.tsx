import React from 'react';
import {AbsoluteFill} from 'remotion';
import {C, MONO, SANS, T} from './theme';

/** Black page with a barely visible survey grid and four registration marks. */
export const Backdrop: React.FC<{children?: React.ReactNode; step?: number}> = ({children, step = 120}) => (
  <AbsoluteFill style={{backgroundColor: C.navy}}>
    <AbsoluteFill
      style={{
        backgroundImage: `linear-gradient(#141414 1px, transparent 1px), linear-gradient(90deg, #141414 1px, transparent 1px)`,
        backgroundSize: `${step}px ${step}px`,
        opacity: 0.55,
      }}
    />
    {[
      {left: 36, top: 36},
      {right: 36, top: 36},
      {left: 36, bottom: 36},
      {right: 36, bottom: 36},
    ].map((pos, i) => (
      <div key={i} style={{position: 'absolute', width: 14, height: 14, ...pos}}>
        <div style={{position: 'absolute', left: 0, right: 0, top: 6.5, height: 1, background: C.dim}} />
        <div style={{position: 'absolute', top: 0, bottom: 0, left: 6.5, width: 1, background: C.dim}} />
      </div>
    ))}
    {children}
  </AbsoluteFill>
);

export const Pill: React.FC<{
  children: React.ReactNode;
  color?: string;
  fill?: boolean;
  size?: number;
  style?: React.CSSProperties;
}> = ({children, color = C.muted, fill = false, size = T.tiny, style}) => (
  <span
    style={{
      display: 'inline-block',
      fontFamily: MONO,
      fontSize: size,
      lineHeight: 1.2,
      color: fill ? C.navy : color,
      background: fill ? color : 'transparent',
      border: `1px solid ${color}`,
      borderRadius: 0,
      padding: `${size * 0.22}px ${size * 0.6}px`,
      whiteSpace: 'nowrap',
      textTransform: 'uppercase',
      ...style,
    }}
  >
    {children}
  </span>
);

export const Card: React.FC<{
  children: React.ReactNode;
  style?: React.CSSProperties;
  border?: string;
}> = ({children, style, border = C.grid}) => (
  <div
    style={{
      background: C.panel,
      border: `1px solid ${border}`,
      borderRadius: 0,
      padding: 32,
      boxSizing: 'border-box',
      ...style,
    }}
  >
    {children}
  </div>
);

/** Small mono label above a scene. */
export const Kicker: React.FC<{children: React.ReactNode; style?: React.CSSProperties}> = ({children, style}) => (
  <div style={{fontFamily: MONO, fontSize: T.tiny, color: C.muted, letterSpacing: 5, textTransform: 'uppercase', ...style}}>
    {children}
  </div>
);

export const Heading: React.FC<{children: React.ReactNode; style?: React.CSSProperties; size?: number}> = ({
  children,
  style,
  size = T.h2,
}) => (
  <div style={{fontFamily: SANS, fontSize: size, fontWeight: 500, color: C.text, letterSpacing: -1.4, ...style}}>
    {children}
  </div>
);

/** Plain card for a run whose data is not in data.json yet. */
export const NotRunYet: React.FC<{title: string; what: string; kicker: string}> = ({title, what, kicker}) => (
  <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
    <Card style={{width: 1100, padding: 56, textAlign: 'center'}}>
      <Kicker>{kicker}</Kicker>
      <Heading style={{marginTop: 18}}>{title}</Heading>
      <div style={{marginTop: 30}}>
        <Pill color={C.muted} size={T.small}>
          not run yet
        </Pill>
      </div>
      <div style={{marginTop: 30, fontFamily: SANS, fontSize: T.body, color: C.muted, lineHeight: 1.45}}>{what}</div>
    </Card>
  </AbsoluteFill>
);

// ---------------------------------------------------------------------------------------------
// The circuit of the modular-addition model, drawn as five stages that can light up one by one.
// lit: 0 = all dim, 1 = tokens path lit, ... 5 = everything lit (fractions animate the stroke).
// ---------------------------------------------------------------------------------------------
const VB_W = 1100;
const VB_H = 400;

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

export const Circuit: React.FC<{
  lit: number;
  width: number;
  freqs: number[];
  p?: number;
  labels?: boolean;
  shimmer?: number; // 0..1 faint accent on everything (used while the box turns to glass)
}> = ({lit, width, freqs, p, labels = true, shimmer = 0}) => {
  const height = (width * (VB_H + (labels ? 50 : 0))) / VB_W;
  const L = (stage: number) => clamp01(lit - stage);
  const nodeColor = (stage: number) => (L(stage) > 0.5 ? C.accent : C.dim);

  const tokens = [
    {id: 'a', y: 120},
    {id: 'b', y: 280},
  ];
  const embed = {x: 230, y: 50, w: 120, h: 300};
  const heads = [80, 160, 240, 320].map((y, i) => ({x: 500, y, id: `h${i}`}));
  const neurons = (freqs.length ? freqs : [0, 0, 0, 0, 0]).slice(0, 6);
  const nY = (i: number) => 200 + (i - (neurons.length - 1) / 2) * 66;
  const unembed = {x: 860, y: 110, w: 110, h: 180};

  // Edge groups: [stage, path]
  const edges: [number, string][] = [];
  tokens.forEach((t) => edges.push([0, `M 110 ${t.y} C 170 ${t.y}, 180 ${t.y}, ${embed.x} ${t.y}`]));
  heads.forEach((h) => edges.push([1, `M ${embed.x + embed.w} 200 C 420 200, 420 ${h.y}, ${h.x - 26} ${h.y}`]));
  heads.forEach((h) =>
    neurons.forEach((_, i) => edges.push([2, `M ${h.x + 26} ${h.y} C 600 ${h.y}, 610 ${nY(i)}, 676 ${nY(i)}`])),
  );
  neurons.forEach((_, i) => edges.push([3, `M 724 ${nY(i)} C 800 ${nY(i)}, 800 200, ${unembed.x} 200`]));
  edges.push([4, `M ${unembed.x + unembed.w} 200 L 1060 200`]);

  const fs = 19; // font size in viewBox units

  return (
    <svg width={width} height={height} viewBox={`0 0 ${VB_W} ${VB_H + (labels ? 50 : 0)}`} style={{overflow: 'visible'}}>
      <defs>
        {/* user-space region: straight horizontal paths have a zero-height bounding box */}
        <filter id="glow" filterUnits="userSpaceOnUse" x={-20} y={-20} width={VB_W + 40} height={VB_H + 90}>
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* dim base edges */}
      {edges.map(([, d], i) => (
        <path key={`d${i}`} d={d} fill="none" stroke={C.dim} strokeWidth={1.6} opacity={0.75 + shimmer * 0.25} />
      ))}
      {/* lit edges, drawn along their length */}
      {edges.map(([s, d], i) => {
        const v = L(s);
        if (v <= 0 && shimmer <= 0) return null;
        return (
          <path
            key={`l${i}`}
            d={d}
            fill="none"
            stroke={C.accent}
            strokeWidth={v > 0 ? 2.6 : 1.4}
            pathLength={1}
            strokeDasharray={1}
            strokeDashoffset={v > 0 ? 1 - v : 0}
            opacity={v > 0 ? 0.95 : shimmer * 0.35}
            filter={v > 0 ? 'url(#glow)' : undefined}
          />
        );
      })}

      {/* tokens */}
      {tokens.map((t) => (
        <g key={t.id}>
          <rect x={40} y={t.y - 32} width={70} height={64} rx={0} fill={C.panel} stroke={nodeColor(0)} strokeWidth={2.4} />
          <text x={75} y={t.y + 11} textAnchor="middle" fontFamily={MONO} fontSize={32} fill={C.text}>
            {t.id}
          </text>
        </g>
      ))}

      {/* embedding block with Fourier waves */}
      <rect x={embed.x} y={embed.y} width={embed.w} height={embed.h} rx={0} fill={C.panel} stroke={nodeColor(1)} strokeWidth={2.4} />
      {[0, 1, 2, 3, 4].map((i) => {
        const y0 = embed.y + 46 + i * 52;
        const k = i + 1;
        let d = '';
        for (let x = 0; x <= 80; x += 2) {
          d += `${x === 0 ? 'M' : 'L'} ${embed.x + 20 + x} ${y0 + Math.sin((x / 80) * Math.PI * 2 * k * 0.6) * 12}`;
        }
        return <path key={i} d={d} fill="none" stroke={nodeColor(1)} strokeWidth={2} opacity={0.9} />;
      })}

      {/* attention heads */}
      {heads.map((h) => (
        <g key={h.id}>
          <circle cx={h.x} cy={h.y} r={26} fill={C.panel} stroke={nodeColor(2)} strokeWidth={2.4} />
          <text x={h.x} y={h.y + 7} textAnchor="middle" fontFamily={MONO} fontSize={fs} fill={L(2) > 0.5 ? C.text : C.muted}>
            {h.id}
          </text>
        </g>
      ))}

      {/* MLP neurons, each tuned to one frequency */}
      {neurons.map((k, i) => (
        <g key={i}>
          <circle cx={700} cy={nY(i)} r={24} fill={C.panel} stroke={nodeColor(3)} strokeWidth={2.4} />
          <text x={700} y={nY(i) + 7} textAnchor="middle" fontFamily={MONO} fontSize={fs} fill={L(3) > 0.5 ? C.text : C.muted}>
            {k || ''}
          </text>
        </g>
      ))}

      {/* unembedding */}
      <rect x={unembed.x} y={unembed.y} width={unembed.w} height={unembed.h} rx={0} fill={C.panel} stroke={nodeColor(4)} strokeWidth={2.4} />
      <text x={unembed.x + unembed.w / 2} y={206} textAnchor="middle" fontFamily={MONO} fontSize={fs} fill={L(4) > 0.5 ? C.text : C.muted}>
        W_U
      </text>
      <circle cx={1066} cy={200} r={7} fill={L(4) > 0.9 ? C.accent : C.dim} filter={L(4) > 0.9 ? 'url(#glow)' : undefined} />

      {labels && (
        <g fontFamily={MONO} fontSize={fs} textAnchor="middle">
          <text x={75} y={VB_H + 30} fill={L(0) > 0.5 ? C.text : C.muted}>tokens a, b</text>
          <text x={embed.x + embed.w / 2} y={VB_H + 30} fill={L(1) > 0.5 ? C.text : C.muted}>Fourier embed</text>
          <text x={500} y={VB_H + 30} fill={L(2) > 0.5 ? C.text : C.muted}>attention</text>
          <text x={700} y={VB_H + 30} fill={L(3) > 0.5 ? C.text : C.muted}>MLP neurons</text>
          <text x={930} y={VB_H + 30} fill={L(4) > 0.5 ? C.accent : C.muted}>{p ? `(a+b) mod ${p}` : '(a+b) mod p'}</text>
        </g>
      )}
    </svg>
  );
};

// ---------------------------------------------------------------------------------------------
// A box drawn in oblique projection. glass = 0 is an opaque black box, glass = 1 is clear glass.
// Children are drawn inside the box volume.
// ---------------------------------------------------------------------------------------------
export const Cube: React.FC<{
  w: number;
  h: number;
  glass: number;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}> = ({w, h, glass, children, style}) => {
  const dx = w * 0.2;
  const dy = w * 0.14;
  const W = w + dx;
  const H = h + dy;
  const g = clamp01(glass);
  const mix = (a: number, b: number) => a + (b - a) * g;
  const edge = `rgba(${Math.round(mix(70, 245))}, ${Math.round(mix(70, 245))}, ${Math.round(mix(70, 245))}, ${mix(1, 0.7)})`;
  const front = `rgba(${Math.round(mix(6, 255))}, ${Math.round(mix(6, 255))}, ${Math.round(mix(6, 255))}, ${mix(1, 0.03)})`;
  const top = `rgba(${Math.round(mix(20, 255))}, ${Math.round(mix(20, 255))}, ${Math.round(mix(20, 255))}, ${mix(1, 0.06)})`;
  const side = `rgba(${Math.round(mix(12, 255))}, ${Math.round(mix(12, 255))}, ${Math.round(mix(12, 255))}, ${mix(1, 0.045)})`;

  return (
    <div style={{position: 'relative', width: W, height: H, ...style}}>
      {/* back edges, only visible through glass */}
      <svg width={W} height={H} style={{position: 'absolute', left: 0, top: 0, overflow: 'visible'}}>
        <g stroke={edge} strokeWidth={1.4} strokeDasharray="6 8" opacity={g * 0.7} fill="none">
          <path d={`M ${dx} 0 L ${dx} ${h}`} />
          <path d={`M ${dx} ${h} L ${W} ${h}`} />
          <path d={`M ${dx} ${h} L 0 ${H}`} />
        </g>
      </svg>
      {/* contents, centred in the volume */}
      <div
        style={{
          position: 'absolute',
          left: dx / 2,
          top: dy / 2,
          width: w,
          height: h,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: g,
        }}
      >
        {children}
      </div>
      {/* faces */}
      <svg width={W} height={H} style={{position: 'absolute', left: 0, top: 0, overflow: 'visible'}}>
        <defs>
          <linearGradient id="sheen" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#ffffff" stopOpacity={0.1 * g} />
            <stop offset="0.35" stopColor="#ffffff" stopOpacity={0} />
            <stop offset="0.62" stopColor="#ffffff" stopOpacity={0.05 * g} />
            <stop offset="1" stopColor="#ffffff" stopOpacity={0} />
          </linearGradient>
        </defs>
        <polygon points={`0,${dy} ${dx},0 ${W},0 ${w},${dy}`} fill={top} stroke={edge} strokeWidth={2} />
        <polygon points={`${w},${dy} ${W},0 ${W},${h} ${w},${H}`} fill={side} stroke={edge} strokeWidth={2} />
        <rect x={0} y={dy} width={w} height={h} fill={front} stroke={edge} strokeWidth={2} />
        <rect x={0} y={dy} width={w} height={h} fill="url(#sheen)" />
      </svg>
    </div>
  );
};
