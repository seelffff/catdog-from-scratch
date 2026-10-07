/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
  readonly VITE_API_MODE?: 'auto' | 'live' | 'mock';
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
