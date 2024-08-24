import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import runtimeEnv from "vite-plugin-runtime-env";

export default defineConfig({
  plugins: [solid(), runtimeEnv()],
});
