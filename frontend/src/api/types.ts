// Контракт с бэкендом. Источник правды: backend/app/schemas.py (и /docs на сервере).

export type Label = 'cat' | 'dog';

export interface Probabilities {
  cat: number;
  dog: number;
}

/** Одна карта признаков: [H][W]. */
export type Map2D = number[][];

/** Необязательные промежуточные выходы модели, ими подменяется локальная визуализация. */
export interface RawActivations {
  conv1?: Map2D[];
  relu1?: Map2D[];
  pool1?: Map2D[];
  conv2?: Map2D[];
  pool2?: Map2D[];
  dense?: number[];
}

export interface Prediction {
  label: Label;
  probabilities: Probabilities;
  /** Имя модели на сервере; null в демо-режиме. */
  model: string | null;
  activations: RawActivations | null;
  mode: 'live' | 'mock';
}

export interface Health {
  status: 'ok';
  model: string;
}

export type Backend =
  | { kind: 'live'; health: Health }
  | { kind: 'mock'; reason: string }
  | { kind: 'offline' };
