import { defineConfig } from 'vite';

// Relative base so the built bundle in dist/ can be served from any path
// (or opened from file://); the e2e suite pins explicit ports/host on the CLI.
export default defineConfig({
  base: './',
  server: { port: 5199, strictPort: true, host: '127.0.0.1' },
  preview: { port: 5201, strictPort: true, host: '127.0.0.1' },
  build: { chunkSizeWarningLimit: 1200 },
});
