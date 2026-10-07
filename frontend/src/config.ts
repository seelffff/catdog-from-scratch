export type ApiMode = 'auto' | 'live' | 'mock';

const env = import.meta.env;
const mode = env.VITE_API_MODE;

export const config = {
  apiBase: (env.VITE_API_BASE ?? '').replace(/\/$/, ''),
  apiMode: (mode === 'live' || mode === 'mock' ? mode : 'auto') as ApiMode,
  predictTimeoutMs: 20_000,
  healthTimeoutMs: 4_000,
  maxUploadBytes: 20 * 1024 * 1024,
  historyLimit: 8,
} as const;
