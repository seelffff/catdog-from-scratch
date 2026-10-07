import { makeMap, type FeatureMap } from './tensor';

/** Ядро 3×3 построчно, 9 чисел. */
export type Kernel3 = readonly number[];

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

/** Свёртка 3×3 без смены размера; на краях повторяется крайний пиксель. */
export function conv3x3Raw(m: FeatureMap, k: Kernel3): Float32Array {
  const { w, h, data } = m;
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let ky = -1; ky <= 1; ky++) {
        const row = clamp(y + ky, 0, h - 1) * w;
        for (let kx = -1; kx <= 1; kx++) {
          s += data[row + clamp(x + kx, 0, w - 1)] * k[(ky + 1) * 3 + kx + 1];
        }
      }
      out[y * w + x] = s;
    }
  }
  return out;
}

export const conv3x3 = (m: FeatureMap, k: Kernel3): FeatureMap => makeMap(m.w, m.h, conv3x3Raw(m, k));

export const relu = (m: FeatureMap): FeatureMap => makeMap(m.w, m.h, m.data.map(v => (v > 0 ? v : 0)));

export function maxPool2(m: FeatureMap): FeatureMap {
  const w = m.w >> 1;
  const h = m.h >> 1;
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * 2 * m.w + x * 2;
      out[y * w + x] = Math.max(m.data[i], m.data[i + 1], m.data[i + m.w], m.data[i + m.w + 1]);
    }
  }
  return makeMap(w, h, out);
}

/** Окно 3×3 вокруг (x, y) с тем же правилом краёв, что и у свёртки. */
export function window3x3(m: FeatureMap, x: number, y: number): number[] {
  const out: number[] = [];
  for (let ky = -1; ky <= 1; ky++) {
    for (let kx = -1; kx <= 1; kx++) {
      out.push(m.data[clamp(y + ky, 0, m.h - 1) * m.w + clamp(x + kx, 0, m.w - 1)]);
    }
  }
  return out;
}
