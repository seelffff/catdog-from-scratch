import { config } from '../config';
import type { FeatureMap } from '../net/tensor';
import { predictMock } from './mock';
import type { Backend, Health, Label, Prediction, RawActivations } from './types';

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

const endpoint = (path: string) => `${config.apiBase}${path}`;

async function fetchWithTimeout(url: string, init: RequestInit, ms: number): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw new ApiError('Сервер не ответил вовремя');
    throw new ApiError('Сервер модели недоступен');
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchHealth(): Promise<Health | null> {
  try {
    const res = await fetchWithTimeout(endpoint('/api/health'), {}, config.healthTimeoutMs);
    if (!res.ok) return null;
    const json: unknown = await res.json();
    if (isRecord(json) && json.status === 'ok' && typeof json.model === 'string') {
      return { status: 'ok', model: json.model };
    }
    return null;
  } catch {
    return null;
  }
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null;
const isProb = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;

/** Проверяет и дополняет ответ /api/predict. Экспортируется для тестов. */
export function normalizePrediction(json: unknown): Omit<Prediction, 'mode'> {
  if (!isRecord(json)) throw new ApiError('Сервер вернул не JSON-объект');
  const probs = isRecord(json.probabilities) ? json.probabilities : {};
  let cat = isProb(probs.cat) ? probs.cat : null;
  let dog = isProb(probs.dog) ? probs.dog : null;
  if (cat === null && dog !== null) cat = 1 - dog;
  if (dog === null && cat !== null) dog = 1 - cat;
  if (cat === null || dog === null) throw new ApiError('В ответе сервера нет probabilities.cat / probabilities.dog');

  const label: Label = json.label === 'cat' || json.label === 'dog' ? json.label : cat >= dog ? 'cat' : 'dog';
  return {
    label,
    probabilities: { cat, dog },
    model: typeof json.model === 'string' ? json.model : null,
    activations: isRecord(json.activations) ? (json.activations as RawActivations) : null,
  };
}

async function predictLive(blob: Blob): Promise<Prediction> {
  const form = new FormData();
  form.append('file', blob, fileName(blob));
  const res = await fetchWithTimeout(endpoint('/api/predict'), { method: 'POST', body: form }, config.predictTimeoutMs);
  if (!res.ok) {
    let detail = '';
    try {
      const body: unknown = await res.json();
      if (isRecord(body) && typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* тело не JSON */
    }
    throw new ApiError(detail || `Сервер ответил ${res.status}`, res.status);
  }
  return { ...normalizePrediction(await res.json()), mode: 'live' };
}

function fileName(blob: Blob): string {
  if (blob instanceof File && blob.name) return blob.name;
  const ext = (blob.type.split('/')[1] ?? 'png').replace('jpeg', 'jpg');
  return `image.${ext}`;
}

/**
 * Решает, куда отправлять фото. В режиме auto сервер считается живым,
 * если отвечает на /api/health; иначе включается демо. Пока сервер не найден,
 * проверка повторяется на каждом запросе, так что запущенный позже бэкенд подхватится сам.
 */
export class ModelClient {
  private current: Backend | null = null;

  get backend(): Backend | null {
    return this.current;
  }

  async detect(): Promise<Backend> {
    if (config.apiMode === 'mock') return (this.current = { kind: 'mock', reason: 'VITE_API_MODE=mock' });
    const health = await fetchHealth();
    if (health) return (this.current = { kind: 'live', health });
    if (config.apiMode === 'live') return (this.current = { kind: 'offline' });
    return (this.current = { kind: 'mock', reason: 'сервер модели не найден' });
  }

  async predict(blob: Blob, input: FeatureMap): Promise<Prediction> {
    const backend = this.current?.kind === 'live' ? this.current : await this.detect();
    switch (backend.kind) {
      case 'live':
        try {
          return await predictLive(blob);
        } catch (e) {
          if (e instanceof ApiError && e.status === undefined) this.current = null; // сервер пропал — перепроверим
          throw e;
        }
      case 'mock':
        return predictMock(input);
      case 'offline':
        throw new ApiError('Сервер модели недоступен');
    }
  }
}
