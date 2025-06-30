import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import runtimeEnv from "vite-plugin-runtime-env";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), runtimeEnv()],
  server: {
    fs: {
      allow: ["..", "node_modules/.pnpm", "../../node_modules/.pnpm",
      // Explicitly allow KaTeX fonts
      "../../node_modules/.pnpm/katex@0.16.21/node_modules/katex/dist/fonts"]
    }
  }
});