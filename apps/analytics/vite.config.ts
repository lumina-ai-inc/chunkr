import path from "path"
import react from "@vitejs/plugin-react"
import runtimeEnv from "vite-plugin-runtime-env";
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react(), runtimeEnv()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})
