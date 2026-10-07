import type { FeatureMap } from '../net/tensor';
import type { Prediction } from './types';

/** Демо-ответ. Вероятности выводятся из хеша пикселей: это не модель. */
export async function predictMock(input: FeatureMap): Promise<Prediction> {
  await new Promise(r => setTimeout(r, 450));
  let h = 0;
  for (let i = 0; i < input.data.length; i += 7) h = (h * 31 + Math.round(input.data[i] * 255)) % 1_000_003;
  const r = (h % 1000) / 1000;
  const conf = 0.58 + 0.4 * (((h >> 3) % 1000) / 1000);
  const cat = r < 0.5 ? conf : 1 - conf;
  return {
    label: cat >= 0.5 ? 'cat' : 'dog',
    probabilities: { cat, dog: 1 - cat },
    model: null,
    activations: null,
    mode: 'mock',
  };
}
