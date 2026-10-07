import { describe, expect, it } from 'vitest';
import { ApiError, normalizePrediction } from '../src/api/client';

describe('normalizePrediction', () => {
  it('принимает полный ответ', () => {
    const p = normalizePrediction({ label: 'dog', probabilities: { cat: 0.1, dog: 0.9 }, model: 'resnet18' });
    expect(p).toEqual({ label: 'dog', probabilities: { cat: 0.1, dog: 0.9 }, model: 'resnet18', activations: null });
  });

  it('дополняет вторую вероятность и выводит label', () => {
    const p = normalizePrediction({ probabilities: { cat: 0.75 } });
    expect(p.label).toBe('cat');
    expect(p.probabilities.dog).toBeCloseTo(0.25);
  });

  it('отвергает ответ без вероятностей', () => {
    expect(() => normalizePrediction({ label: 'cat' })).toThrow(ApiError);
    expect(() => normalizePrediction({ probabilities: { cat: 7 } })).toThrow(ApiError);
  });
});
