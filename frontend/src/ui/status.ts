import type { Backend } from '../api/types';

const $ = (id: string) => document.getElementById(id)!;

/** Индикатор в шапке и строка в подвале: куда уходят фото. */
export function renderStatus(backend: Backend | null): void {
  const root = $('modelStatus');
  const text = $('modelStatusText');
  const foot = $('footMode');

  const set = (mode: string, label: string, footer: string) => {
    root.dataset.mode = mode;
    text.textContent = label;
    foot.textContent = footer;
  };

  if (!backend) return set('unknown', 'проверяю модель…', '—');
  switch (backend.kind) {
    case 'live':
      return backend.health.model === 'dummy'
        ? set('mock', 'заглушка сервера', 'Сервер работает, но вместо модели отвечает заглушка.')
        : set('live', backend.health.model, `Отвечает модель «${backend.health.model}» на сервере.`);
    case 'mock':
      return set('mock', 'демо-режим', `Демо: ${backend.reason}, ответы выдуманы.`);
    case 'offline':
      return set('offline', 'сервер недоступен', 'Сервер модели не отвечает.');
  }
}
