/** Карта признаков: одноканальное изображение из чисел float. */
export interface FeatureMap {
  readonly w: number;
  readonly h: number;
  readonly data: Float32Array;
  readonly max: number;
  readonly min: number;
  readonly absMax: number;
}

export function makeMap(w: number, h: number, data: Float32Array): FeatureMap {
  let max = data.length ? -Infinity : 0;
  let min = data.length ? Infinity : 0;
  for (const v of data) {
    if (v > max) max = v;
    if (v < min) min = v;
  }
  return { w, h, data, max, min, absMax: Math.max(Math.abs(max), Math.abs(min)) };
}

/** Из массива [H][W], который пришёл в JSON с сервера. */
export function fromNested(rows: number[][]): FeatureMap {
  const h = rows.length;
  const w = rows[0]?.length ?? 0;
  if (!h || !w) throw new Error('пустая карта');
  const data = new Float32Array(w * h);
  rows.forEach((row, y) => {
    if (!Array.isArray(row) || row.length !== w) throw new Error('карта не прямоугольная');
    row.forEach((v, x) => {
      if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error('в карте не число');
      data[y * w + x] = v;
    });
  });
  return makeMap(w, h, data);
}
