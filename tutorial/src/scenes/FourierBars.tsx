import React from 'react';
import {C, MONO} from '../theme';

/** Bar chart of Fourier power share per frequency; key frequencies glow in the accent colour. */
export const FourierBars: React.FC<{
  freqs: number[];
  share: number[];
  keys: number[];
  labelTop: number[];
  width: number;
  height: number;
  grow: (i: number) => number; // 0..1 per bar
  fontSize?: number;
  axis?: boolean;
}> = ({freqs, share, keys, labelTop, width, height, grow, fontSize = 22, axis = true}) => {
  const n = Math.min(freqs.length, share.length);
  if (!n) return null;
  const max = Math.max(...share.slice(0, n)) || 1;
  const top = fontSize * 1.6;
  const bottom = axis ? fontSize * 1.8 : 4;
  const ch = height - top - bottom;
  const step = width / n;
  const bw = Math.max(2, step * 0.7);
  const keySet = new Set(keys);
  const uniform = 1 / n;
  const uy = top + ch - (uniform / max) * ch;
  return (
    <svg width={width} height={height} style={{overflow: 'visible'}}>
      <defs>
        <filter id="barglow" x="-100%" y="-20%" width="300%" height="140%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <line x1={0} x2={width} y1={top + ch} y2={top + ch} stroke={C.grid} strokeWidth={1.5} />
      {Array.from({length: n}).map((_, i) => {
        const k = freqs[i];
        const g = grow(i);
        const h = Math.max(1.5, (share[i] / max) * ch * g);
        const isKey = keySet.has(k);
        const x = i * step + (step - bw) / 2;
        return (
          <g key={i}>
            <rect
              x={x}
              y={top + ch - h}
              width={bw}
              height={h}
              rx={Math.min(3, bw / 3)}
              fill={isKey ? C.accent : C.dim}
              filter={isKey && g > 0.2 ? 'url(#barglow)' : undefined}
            />
            {labelTop.includes(k) && g > 0.6 && (
              <text
                x={x + bw / 2}
                y={top + ch - h - fontSize * 0.45}
                textAnchor="middle"
                fontFamily={MONO}
                fontSize={fontSize}
                fill={C.accent}
                opacity={Math.min(1, (g - 0.6) / 0.4)}
              >
                {k}
              </text>
            )}
          </g>
        );
      })}
      <line x1={0} x2={width} y1={uy} y2={uy} stroke={C.muted} strokeWidth={1.4} strokeDasharray="6 6" opacity={0.8} />
      <text x={width - step * 2.5} y={uy - 10} textAnchor="end" fontFamily={MONO} fontSize={fontSize * 0.9} fill={C.muted}>
        uniform 1/{n}
      </text>
      {axis && (
        <g fontFamily={MONO} fontSize={fontSize} fill={C.muted}>
          <text x={step / 2} y={top + ch + fontSize * 1.3} textAnchor="middle">
            {freqs[0]}
          </text>
          <text x={width / 2} y={top + ch + fontSize * 1.3} textAnchor="middle">
            frequency k
          </text>
          <text x={width - step / 2} y={top + ch + fontSize * 1.3} textAnchor="middle">
            {freqs[n - 1]}
          </text>
        </g>
      )}
    </svg>
  );
};
