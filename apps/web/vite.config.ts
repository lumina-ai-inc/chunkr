import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current directory.
  const env = loadEnv(mode, process.cwd(), '');
  
  // Helper function to replace placeholders in HTML
  const replaceEnvVars = (html: string): string => {
    return html
      .replace(/%VITE_IS_SELF_HOST%/g, env.VITE_IS_SELF_HOST || '')
      .replace(/%VITE_DOCS_URL%/g, env.VITE_DOCS_URL || '')
      .replace(/%VITE_KEYCLOAK_URL%/g, env.VITE_KEYCLOAK_URL || 'http://localhost:8080')
      .replace(/%VITE_KEYCLOAK_REALM%/g, env.VITE_KEYCLOAK_REALM || 'orin')
      .replace(/%VITE_KEYCLOAK_CLIENT_ID%/g, env.VITE_KEYCLOAK_CLIENT_ID || 'orin')
      .replace(/%VITE_KEYCLOAK_REDIRECT_URI%/g, env.VITE_KEYCLOAK_REDIRECT_URI || 'http://localhost:5173')
      .replace(/%VITE_KEYCLOAK_POST_LOGOUT_REDIRECT_URI%/g, env.VITE_KEYCLOAK_POST_LOGOUT_REDIRECT_URI || 'http://localhost:5173')
      .replace(/%VITE_API_URL%/g, env.VITE_API_URL || 'http://localhost:8000')
      .replace(/%VITE_CONTENTFUL_SPACE_ID%/g, env.VITE_CONTENTFUL_SPACE_ID || '')
      .replace(/%VITE_CONTENTFUL_ACCESS_TOKEN%/g, env.VITE_CONTENTFUL_ACCESS_TOKEN || '')
      .replace(/%VITE_FEATURE_FLAG_PIPELINE%/g, env.VITE_FEATURE_FLAG_PIPELINE || 'true');
  };
  
  return {
    plugins: [
      react(),
      {
        name: 'html-transform',
        transformIndexHtml: {
          enforce: 'pre',
          transform(html: string) {
            return replaceEnvVars(html);
          },
        },
      },
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
  };
});
