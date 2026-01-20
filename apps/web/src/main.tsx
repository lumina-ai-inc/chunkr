import ReactDOM from "react-dom/client";
import { Provider } from "react-redux";
import { Toaster } from "react-hot-toast";
import { AuthProvider, AuthProviderProps } from "react-oidc-context";
import {
  createBrowserRouter,
  RouterProvider,
  Outlet,
  RouteObject,
} from "react-router-dom";
import { Theme } from "@radix-ui/themes";
import { QueryClient, QueryClientProvider } from "react-query";
import "@radix-ui/themes/styles.css";
import "./index.css";
import Auth from "./auth/Auth.tsx";
import AuthGuard from "./auth/AuthGuard.tsx";
import store from "./store/store";
import Dashboard from "./pages/Dashboard/Dashboard.tsx"; // Old dashboard (keep for reference)
import DashboardThreePane from "./pages/Dashboard/DashboardThreePane.tsx";
import Landing from "./pages/Landing/Landing.tsx"; // Old landing (keep for reference)
import LandingChat from "./pages/Landing/LandingChat.tsx";
import Checkout from "./pages/Checkout/Checkout";
import { PublicSharePage } from "./pages/Share/PublicSharePage";
import { env } from "./config/env";

// Debug logging
console.log('Environment variables:', {
  VITE_KEYCLOAK_URL: env.keycloakUrl,
  VITE_KEYCLOAK_REALM: env.keycloakRealm,
  VITE_KEYCLOAK_CLIENT_ID: env.keycloakClientId,
  VITE_KEYCLOAK_REDIRECT_URI: env.keycloakRedirectUri,
  VITE_KEYCLOAK_POST_LOGOUT_REDIRECT_URI: env.keycloakPostLogoutRedirectUri,
});

const oidcConfig: AuthProviderProps = {
  authority: `${env.keycloakUrl}/realms/${env.keycloakRealm}`,
  client_id: env.keycloakClientId,
  redirect_uri: env.keycloakRedirectUri,
  post_logout_redirect_uri: env.keycloakPostLogoutRedirectUri,
  automaticSilentRenew: true,
  loadUserInfo: true,
  onSigninCallback: (user) => {
    const state = user?.state as { returnTo?: string };
    if (state?.returnTo) {
      window.location.href = state.returnTo;
    } else {
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  },
};

const router = createBrowserRouter([
  {
    path: "/share/:shareId",
    element: <PublicSharePage />, // No AuthGuard - public route
  },
  {
    path: "/",
    element: <Outlet />,
    children: [
      {
        index: true,
        element: <LandingChat />, // New chat-first landing
      },
      {
        path: "dashboard",
        element: (
          <AuthGuard>
            <DashboardThreePane /> 
          </AuthGuard>
        ),
      },
      {
        path: "dashboard-old",
        element: (
          <AuthGuard>
            <Dashboard /> // Old dashboard kept for reference
          </AuthGuard>
        ),
      },
      {
        path: "landing-old",
        element: <Landing />, // Old landing kept for reference
      },
      {
        path: "checkout/return",
        element: (
          <AuthGuard>
            <Checkout />
          </AuthGuard>
        ),
      },
    ].filter(Boolean) as RouteObject[],
  },
]);

const queryClient = new QueryClient();
ReactDOM.createRoot(document.getElementById("root")!).render(
  <Theme
    scaling="100%"
    panelBackground="solid"
    style={{
      height: "100%",
      backgroundColor: "#fff",
      color: "#111",
    }}
  >
    <QueryClientProvider client={queryClient}>
      <AuthProvider {...oidcConfig}>
        <Provider store={store}>
          <Auth>
            <RouterProvider router={router} />
          </Auth>
        </Provider>
      </AuthProvider>
    </QueryClientProvider>
    <Toaster
      position="bottom-center"
      toastOptions={{
        style: {
          background: "rgba(2, 5, 6, 0.95)",
          color: "#fff",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          backdropFilter: "blur(8px)",
          borderRadius: "8px",
          padding: "16px",
          fontSize: "14px",
          boxShadow: "0 4px 24px rgba(0, 0, 0, 0.2)",
          maxWidth: "380px",
          zIndex: 10000,
        },
        success: {
          iconTheme: {
            primary: "#27c93f", // Matching your terminal button green
            secondary: "rgba(2, 5, 6, 0.95)",
          },
        },
        error: {
          iconTheme: {
            primary: "#ff5f56", // Matching your terminal button red
            secondary: "rgba(2, 5, 6, 0.95)",
          },
        },
        loading: {
          iconTheme: {
            primary: "#67e8f9", // Matching your cyan accent color
            secondary: "rgba(2, 5, 6, 0.95)",
          },
        },
        duration: 4000,
      }}
    />
  </Theme>
);
