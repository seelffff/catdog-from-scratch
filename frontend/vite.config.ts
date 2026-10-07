import { defineConfig } from 'vite';

// В dev-режиме запросы /api проксируются на бэкенд (uvicorn на :8000).
export default defineConfig({
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } },
  },
  build: { outDir: 'dist', sourcemap: true },
});
