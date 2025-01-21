import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import runtimeEnv from "vite-plugin-runtime-env";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), runtimeEnv()],
});
