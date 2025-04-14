import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy /api requests to your FastAPI backend
      '/api': {
        target: 'http://localhost:6969', // Your FastAPI server address
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''), // Remove /api prefix
      },
    },
    port: 3000, // Optional: Set a specific port for the frontend dev server
    open: true // Optional: Automatically open the browser
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})