import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/stats': { target: process.env.STATS_API || 'http://stats:8001', changeOrigin: true },
    },
  },
});
