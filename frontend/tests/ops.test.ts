import { describe, expect, it } from 'vitest';
import { FILTERS, convStep, forward, mergeActivations } from '../src/net/cnn';
import { conv3x3, maxPool2, relu } from '../src/net/ops';
import { fromNested, makeMap } from '../src/net/tensor';

const flat = (w: number, h: number, v: number) => makeMap(w, h, new Float32Array(w * h).fill(v));

describe('ops', () => {
  it('свёртка ребра на однотонной картинке даёт ноль', () => {
    const out = conv3x3(flat(8, 8, 0.5), FILTERS[0].kernel);
    expect(out.absMax).toBe(0);
  });

  it('вертикальный фильтр видит вертикальную границу', () => {
    const data = new Float32Array(16);
    for (let y = 0; y < 4; y++) for (let x = 2; x < 4; x++) data[y * 4 + x] = 1;
    const out = conv3x3(makeMap(4, 4, data), FILTERS[0].kernel);
    expect(out.data[1 * 4 + 1]).toBe(4);
    expect(out.data[1 * 4 + 0]).toBe(0);
  });

  it('relu обнуляет отрицательные', () => {
    const out = relu(makeMap(2, 1, Float32Array.from([-3, 2])));
    expect(Array.from(out.data)).toEqual([0, 2]);
  });

  it('max-pool берёт максимум из 2×2 и вдвое уменьшает', () => {
    const out = maxPool2(makeMap(2, 2, Float32Array.from([1, 5, 3, 2])));
    expect([out.w, out.h, out.data[0]]).toEqual([1, 1, 5]);
  });
});

describe('cnn', () => {
  it('прямой проход даёт ожидаемые размеры', () => {
    const acts = forward(flat(64, 64, 0.3));
    expect(acts.conv1).toHaveLength(4);
    expect([acts.pool1[0].w, acts.pool2[0].w]).toEqual([32, 16]);
    expect(acts.flat).toHaveLength(1024);
    expect(acts.hidden).toHaveLength(16);
  });

  it('шаг свёртки совпадает с картой первого слоя', () => {
    const data = Float32Array.from({ length: 64 * 64 }, (_, i) => ((i * 37) % 256) / 255);
    const input = makeMap(64, 64, data);
    const acts = forward(input);
    const step = convStep(input, 10, 20, 2);
    expect(step.sum / 255).toBeCloseTo(acts.conv1[2].data[20 * 64 + 10], 4);
  });

  it('активации модели подменяют локальные, битые слои пропускаются', () => {
    const local = forward(flat(64, 64, 0.3));
    const merged = mergeActivations(local, {
      conv1: [[[1, 2], [3, 4]]],
      pool1: [[[1, 2], [3]]],
      dense: [0.5, 1.5],
    });
    expect(merged.conv1[0].w).toBe(2);
    expect(merged.pool1).toBe(local.pool1);
    expect(Array.from(merged.hidden)).toEqual([0.5, 1.5]);
    expect([...merged.fromModel].sort()).toEqual(['conv1', 'dense']);
  });

  it('fromNested отвергает непрямоугольные карты', () => {
    expect(() => fromNested([[1, 2], [3]])).toThrow();
  });
});
