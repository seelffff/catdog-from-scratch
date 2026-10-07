import type { Prediction } from '../api/types';
import { W1, W2, type Activations } from '../net/cnn';
import { mix, palette, rgba } from './theme';

const INPUT_NODES = 20;
const HIDDEN_NODES = 16;
const LABELS = ['кошка', 'собака'] as const;

interface Edge {
  from: number;
  to: number;
  s: number;
}

interface DenseData {
  a0: number[];
  a1: number[];
  a2: [number, number];
  e01: Edge[];
  e12: Edge[];
  pulses01: Edge[];
  pulses12: Edge[];
  win: 0 | 1;
  flatLen: number;
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

function sample<T>(arr: ArrayLike<T>, n: number): T[] {
  if (arr.length <= n) return Array.from(arr);
  return Array.from({ length: n }, (_, i) => arr[Math.floor(((i + 0.5) * arr.length) / n)]);
}

function normalizeEdges(edges: Edge[]): void {
  const m = Math.max(...edges.map(e => e.s)) || 1;
  for (const e of edges) e.s /= m;
}

/** Полносвязные слои на canvas: вход (выборка) → скрытый → выход. */
export class DenseView {
  /** Прогресс анимации: −1 пусто, 0..1 вход→скрытый, 1..2 скрытый→выход. */
  progress = -1;
  private data: DenseData | null = null;

  constructor(private readonly canvas: HTMLCanvasElement) {
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  clear(): void {
    this.data = null;
    this.progress = -1;
    this.draw();
  }

  setData(acts: Activations, pred: Prediction): void {
    const flatIdx = Array.from({ length: INPUT_NODES }, (_, i) => Math.floor(((i + 0.5) * acts.flat.length) / INPUT_NODES));
    const flatMax = Math.max(...acts.flat) || 1;
    const a0 = flatIdx.map(i => acts.flat[i] / flatMax);

    const hidden = sample(Array.from(acts.hidden, Math.abs), HIDDEN_NODES);
    const hMax = Math.max(...hidden) || 1;
    const a1 = hidden.map(v => v / hMax);

    // Веса учебные: если модель прислала свой dense, связи всё равно иллюстративные.
    const e01: Edge[] = [];
    a1.forEach((_, j) => flatIdx.forEach((fi, i) => e01.push({ from: i, to: j, s: a0[i] * Math.abs(W1[j % W1.length][fi] ?? 0) })));
    const e12: Edge[] = [];
    a1.forEach((a, j) => [0, 1].forEach(k => e12.push({ from: j, to: k, s: a * Math.abs(W2[j % W2.length][k]) })));
    normalizeEdges(e01);
    normalizeEdges(e12);
    const strongest = (list: Edge[]) => [...list].sort((a, b) => b.s - a.s);

    this.data = {
      a0, a1,
      a2: [pred.probabilities.cat, pred.probabilities.dog],
      e01, e12,
      pulses01: strongest(e01).slice(0, 28),
      pulses12: strongest(e12).filter(e => e.s > 0.12),
      win: pred.label === 'cat' ? 0 : 1,
      flatLen: acts.flat.length,
    };
  }

  draw(): void {
    const cv = this.canvas;
    const W = cv.clientWidth;
    const H = cv.clientHeight;
    if (!W || !H) return;
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
    }
    const ctx = cv.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const { surface, ink, accent, muted, sans, mono } = palette();
    const d = this.data;
    const p = this.progress;
    const nHidden = d ? d.a1.length : HIDDEN_NODES;

    const top = 34;
    const bot = H - 14;
    const mid = (top + bot) / 2;
    const xs = [14, Math.round(W * 0.46), W - (W < 460 ? 104 : 140)];
    const column = (n: number) => Array.from({ length: n }, (_, i) => top + ((bot - top) * (i + 0.5)) / n);
    const ys = [column(INPUT_NODES), column(nHidden), [mid - 44, mid + 44]];

    ctx.font = `500 10px ${mono}`;
    ctx.fillStyle = rgba(muted);
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    ctx.fillText(`ВХОД · ${d?.flatLen ?? 1024}`, xs[0] - 4, 12);
    ctx.textAlign = 'center';
    ctx.fillText(`СКРЫТЫЙ · ${nHidden}`, xs[1], 12);
    ctx.fillText('ВЫХОД · 2', xs[2], 12);

    const reach1 = clamp01(p);
    const reach2 = clamp01(p - 1);
    ctx.lineWidth = 1;
    const line = (x0: number, y0: number, x1: number, y1: number, a: number) => {
      ctx.strokeStyle = rgba(ink, a);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    };
    if (d) {
      for (const e of d.e01) line(xs[0], ys[0][e.from], xs[1], ys[1][e.to], 0.025 + 0.4 * e.s * reach1);
      for (const e of d.e12) line(xs[1], ys[1][e.from], xs[2], ys[2][e.to], 0.04 + 0.55 * e.s * reach2);
    } else {
      for (const y0 of ys[0]) for (const y1 of ys[1]) line(xs[0], y0, xs[1], y1, 0.03);
      for (const y1 of ys[1]) for (const y2 of ys[2]) line(xs[1], y1, xs[2], y2, 0.06);
    }

    // Импульсы бегут по самым сильным связям, с небольшим разбросом во времени.
    const pulses = (list: Edge[], layer: 0 | 1, t: number) => {
      ctx.fillStyle = rgba(accent);
      list.forEach((e, k) => {
        const tt = clamp01((t - (k / list.length) * 0.35) / 0.65);
        if (tt <= 0 || tt >= 1) return;
        const x = xs[layer] + (xs[layer + 1] - xs[layer]) * tt;
        const y = ys[layer][e.from] + (ys[layer + 1][e.to] - ys[layer][e.from]) * tt;
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      });
    };
    if (d && p > 0 && p < 1) pulses(d.pulses01, 0, p);
    if (d && p > 1 && p < 2) pulses(d.pulses12, 1, p - 1);

    const node = (x: number, y: number, r: number, fill: string, stroke: string) => {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.stroke();
    };
    const lit0 = d && p >= 0 ? 1 : 0;
    const lit1 = d ? clamp01((p - 0.75) / 0.25) : 0;
    const lit2 = d ? clamp01((p - 1.75) / 0.25) : 0;

    ys[0].forEach((y, i) => node(xs[0], y, 4, rgba(mix(surface, ink, (d?.a0[i] ?? 0) * lit0)), rgba(ink, 0.5)));
    ys[1].forEach((y, i) => node(xs[1], y, 6, rgba(mix(surface, ink, (d?.a1[i] ?? 0) * lit1)), rgba(ink, 0.6)));

    ys[2].forEach((y, k) => {
      const prob = d?.a2[k] ?? 0;
      const isWin = d?.win === k && lit2 > 0;
      const fill = isWin ? mix(surface, accent, lit2) : mix(surface, ink, prob * lit2);
      node(xs[2], y, 10, rgba(fill), rgba(isWin ? accent : ink, 0.8));
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = rgba(isWin ? accent : ink);
      ctx.font = `500 13px ${sans}`;
      ctx.fillText(LABELS[k], xs[2] + 20, y - 8);
      ctx.fillStyle = rgba(muted);
      ctx.font = `12px ${mono}`;
      ctx.fillText(lit2 ? `${(prob * 100).toFixed(1)}%` : '—', xs[2] + 20, y + 9);
    });
  }
}
