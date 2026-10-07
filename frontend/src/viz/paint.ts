import type { FeatureMap } from '../net/tensor';
import { mix, palette, rgba, type RGB } from './theme';

export interface PaintOptions {
  /** Настоящие оттенки серого (для входа). */
  photo?: boolean;
  /** Нормировать по |max| и показывать минус акцентным цветом. */
  signed?: boolean;
  /** Сколько первых пикселей показать (построчно); остальное фон. */
  reveal?: number;
  /** Своя шкала вместо max карты. */
  scale?: number;
  /** 0..1 — насколько погашены отрицательные значения (анимация ReLU). */
  reluFade?: number;
}

/** 0 = фон, плюс = чернила, минус = акцент. */
export function paintMap(canvas: HTMLCanvasElement, m: FeatureMap, o: PaintOptions = {}): void {
  if (canvas.width !== m.w || canvas.height !== m.h) {
    canvas.width = m.w;
    canvas.height = m.h;
  }
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(m.w, m.h);
  const px = img.data;
  const n = m.w * m.h;
  const reveal = o.reveal ?? n;
  const scale = o.scale ?? ((o.signed ? m.absMax : Math.max(m.max, -m.min)) || 1);
  const fade = o.reluFade ?? 0;
  const { mapBg, ink, accent } = palette();

  for (let i = 0; i < n; i++) {
    let c: RGB;
    if (i >= reveal) c = mapBg;
    else if (o.photo) {
      const g = m.data[i] * 255;
      c = [g, g, g];
    } else {
      let v = m.data[i] / scale;
      if (v < 0) v *= 1 - fade;
      c = mix(mapBg, v >= 0 ? ink : accent, Math.min(1, Math.abs(v)) ** 0.7);
    }
    const p = i * 4;
    px[p] = c[0];
    px[p + 1] = c[1];
    px[p + 2] = c[2];
    px[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export function clearCanvas(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = rgba(palette().mapBg);
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
