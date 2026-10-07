/*
 * Учебная CNN, которая считается в браузере, только ради визуализации.
 * Фильтры первого слоя классические (Собель, Лаплас), остальные веса фиксированные.
 * Ответ «кошка/собака» даёт настоящая модель; её активации, если она их пришлёт,
 * подменяют локальные через mergeActivations().
 */
import type { RawActivations } from '../api/types';
import { conv3x3, conv3x3Raw, maxPool2, relu, window3x3, type Kernel3 } from './ops';
import { fromNested, makeMap, type FeatureMap } from './tensor';

export const INPUT_SIZE = 64;

export interface Filter {
  name: string;
  kernel: Kernel3;
}

export const FILTERS: readonly Filter[] = [
  { name: 'вертикальные края', kernel: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
  { name: 'горизонтальные края', kernel: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
  { name: 'диагонали', kernel: [0, 1, 2, -1, 0, 1, -2, -1, 0] },
  { name: 'пятна и точки', kernel: [0, -1, 0, -1, 4, -1, 0, -1, 0] },
];

function rng(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = rng(42);
const gauss = () => (rand() + rand() + rand() + rand() - 2) * 1.732;

// Второй слой: каждая выходная карта смешивает входные через размытие и немного шума,
// чтобы фильтры выглядели обученными.
const BLUR = [1, 2, 1, 2, 4, 2, 1, 2, 1].map(v => v / 16);
const MIX = [
  [1, 1, 0, 0],
  [0, -0.5, 1, 0],
  [0.5, 0, 0, 1],
  [-0.3, 1, -0.3, 0.6],
];
const FILTERS2: Kernel3[][] = MIX.map(row => row.map(c => BLUR.map(b => c * b + gauss() * 0.04)));

const FLAT = FILTERS2.length * (INPUT_SIZE / 4) ** 2;
export const HIDDEN = 16;
export const W1: readonly Float32Array[] = Array.from({ length: HIDDEN }, () =>
  Float32Array.from({ length: FLAT }, () => gauss() / Math.sqrt(FLAT)),
);
export const W2: readonly (readonly [number, number])[] = Array.from({ length: HIDDEN }, () => [gauss(), gauss()] as const);

export type StageKey = 'conv1' | 'relu1' | 'pool1' | 'conv2' | 'pool2';

export interface Activations {
  input: FeatureMap;
  conv1: FeatureMap[];
  relu1: FeatureMap[];
  pool1: FeatureMap[];
  conv2: FeatureMap[];
  pool2: FeatureMap[];
  flat: Float32Array;
  hidden: Float32Array;
  /** Какие слои пришли от настоящей модели. */
  fromModel: ReadonlySet<StageKey | 'dense'>;
}

/** Центральный квадрат изображения в оттенках серого, 64×64, значения 0..1 с шагом 1/255. */
export function toGrayscale(source: CanvasImageSource, sw: number, sh: number): FeatureMap {
  const side = Math.min(sw, sh);
  // Уменьшаем в два шага, чтобы 64×64 не было в ступеньках.
  const mid = document.createElement('canvas');
  mid.width = mid.height = INPUT_SIZE * 4;
  const mctx = mid.getContext('2d')!;
  mctx.imageSmoothingQuality = 'high';
  mctx.drawImage(source, (sw - side) / 2, (sh - side) / 2, side, side, 0, 0, mid.width, mid.height);

  const cv = document.createElement('canvas');
  cv.width = cv.height = INPUT_SIZE;
  const ctx = cv.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(mid, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const px = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const out = new Float32Array(INPUT_SIZE * INPUT_SIZE);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.round(0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2]) / 255;
  }
  return makeMap(INPUT_SIZE, INPUT_SIZE, out);
}

/** Прямой проход учебной сети. Чистая функция, без DOM. */
export function forward(input: FeatureMap): Activations {
  const conv1 = FILTERS.map(f => conv3x3(input, f.kernel));
  const relu1 = conv1.map(relu);
  const pool1 = relu1.map(maxPool2);
  const conv2 = FILTERS2.map(row => {
    const { w, h } = pool1[0];
    const sum = new Float32Array(w * h);
    row.forEach((k, i) => {
      const part = conv3x3Raw(pool1[i], k);
      for (let j = 0; j < sum.length; j++) sum[j] += part[j];
    });
    return relu(makeMap(w, h, sum));
  });
  const pool2 = conv2.map(maxPool2);

  const flat = new Float32Array(pool2.reduce((n, m) => n + m.data.length, 0));
  let offset = 0;
  for (const m of pool2) {
    flat.set(m.data, offset);
    offset += m.data.length;
  }
  const hidden = Float32Array.from(W1, row => {
    let s = 0;
    const n = Math.min(row.length, flat.length);
    for (let i = 0; i < n; i++) s += row[i] * flat[i];
    return Math.max(0, s);
  });

  return { input, conv1, relu1, pool1, conv2, pool2, flat, hidden, fromModel: new Set() };
}

export const computeFromImage = (img: HTMLImageElement): Activations =>
  forward(toGrayscale(img, img.naturalWidth, img.naturalHeight));

export interface ConvStep {
  x: number;
  y: number;
  pixels: number[];
  kernel: Kernel3;
  sum: number;
  relu: number;
  filter: Filter;
}

/** Один шаг свёртки первого слоя в точке (x, y), в единицах пикселей 0..255. */
export function convStep(input: FeatureMap, x: number, y: number, filterIndex: number): ConvStep {
  const filter = FILTERS[filterIndex];
  const pixels = window3x3(input, x, y).map(v => Math.round(v * 255));
  const sum = pixels.reduce((s, v, i) => s + v * filter.kernel[i], 0);
  return { x, y, pixels, kernel: filter.kernel, sum, relu: Math.max(0, sum), filter };
}

const STAGE_KEYS: readonly StageKey[] = ['conv1', 'relu1', 'pool1', 'conv2', 'pool2'];

/** Подменяет локальные активации теми, что прислала модель. Битые слои пропускаются. */
export function mergeActivations(local: Activations, raw: RawActivations | null): Activations {
  if (!raw) return local;
  const out: Activations = { ...local };
  const fromModel = new Set<StageKey | 'dense'>();
  for (const key of STAGE_KEYS) {
    const maps = raw[key];
    if (!Array.isArray(maps) || !maps.length) continue;
    try {
      out[key] = maps.map(fromNested);
      fromModel.add(key);
    } catch (e) {
      console.warn(`activations.${key} пропущены:`, (e as Error).message);
    }
  }
  if (Array.isArray(raw.dense) && raw.dense.length && raw.dense.every(Number.isFinite)) {
    out.hidden = Float32Array.from(raw.dense);
    fromModel.add('dense');
  }
  out.fromModel = fromModel;
  return out;
}
