import {Data} from '../types';

// Box geometry shared by the cold open and the glass reveal, so the cut between them is seamless.
export const BOX = {w: 620, h: 420, left: 588, top: 96};
export const BOX_DX = BOX.w * 0.2;
export const BOX_DY = BOX.w * 0.14;
export const BOX_MID_Y = BOX.top + BOX_DY + BOX.h / 2;

/** Key frequencies for the circuit drawing: the top frequencies from the periodicity run, ascending. */
export const circuitFreqs = (data: Data): number[] => {
  const top = data.modadd?.top ?? [];
  if (top.length) return top.map((t) => t.frequency).sort((a, b) => a - b).slice(0, 6);
  return (data.modadd?.key_frequencies ?? []).slice(0, 6);
};

/** First experiment id (E0003, V0001, ...) mentioned in a notebook text. */
export const firstId = (s?: string) => (s ?? '').match(/\b[EV]\d{4}\b/)?.[0];
