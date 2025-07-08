import { keycloakify } from "keycloakify/vite-plugin";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import runtimeEnv from "vite-plugin-runtime-env";

const isKcBuild = process.env.IS_KC_BUILD === "true";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    ...(isKcBuild ? [] : [runtimeEnv()]),
    keycloakify({
      accountThemeImplementation: "none",
    }),
  ],
  server: {
    fs: {
      allow: [
        "..",
        "node_modules/.pnpm",
        "../../node_modules/.pnpm",
        // Explicitly allow KaTeX fonts
        "../../node_modules/.pnpm/katex@0.16.21/node_modules/katex/dist/fonts",
      ],
    },
  },
});
