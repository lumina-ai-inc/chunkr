import ReactDOM from "react-dom/client";
import { Provider } from "react-redux";
import { Toaster } from "react-hot-toast";
import { AuthProvider, AuthProviderProps } from "react-oidc-context";
import { createBrowserRouter, RouterProvider, Outlet } from "react-router-dom";
import { Theme } from "@radix-ui/themes";
import { QueryClient, QueryClientProvider } from "react-query";
import "@radix-ui/themes/styles.css";
import "./index.css";
import Auth from "./auth/Auth.tsx";
import Home from "./pages/Home/Home.tsx";
import AuthGuard from "./auth/AuthGuard.tsx";
import store from "./store/store";
import Dashboard from "./pages/Dashboard/Dashboard.tsx";
import Checkout from "./pages/Checkout/Checkout";

const oidcConfig: AuthProviderProps = {
  authority:
    import.meta.env.VITE_KEYCLOAK_URL +
    "/realms/" +
    import.meta.env.VITE_KEYCLOAK_REALM,
  client_id: import.meta.env.VITE_KEYCLOAK_CLIENT_ID,
  redirect_uri: import.meta.env.VITE_KEYCLOAK_REDIRECT_URI,
  post_logout_redirect_uri: import.meta.env
    .VITE_KEYCLOAK_POST_LOGOUT_REDIRECT_URI,
  onSigninCallback: (user) => {
    const state = user?.state as { returnTo?: string };
    if (state?.returnTo) {
      window.location.href = state.returnTo;
    } else {
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  },
};

function RootLayout() {
  return (
    <Auth>
      <Outlet />
    </Auth>
  );
}

const router = createBrowserRouter([
  {
    element: <RootLayout />,
    children: [
      {
        path: "/",
        element: <Home />,
      },
      {
        path: "/dashboard",
        element: (
          <AuthGuard>
            <Dashboard />
          </AuthGuard>
        ),
      },
      {
        path: "/checkout/return",
        element: (
          <AuthGuard>
            <Checkout />
          </AuthGuard>
        ),
      },
      {
        path: "*",
        element: <Home />,
      },
    ],
  },
]);

const queryClient = new QueryClient();
ReactDOM.createRoot(document.getElementById("root")!).render(
  <Theme
    scaling="100%"
    panelBackground="solid"
    style={{
      height: "100%",
      backgroundColor: "#020809",
    }}
  >
    <QueryClientProvider client={queryClient}>
      <AuthProvider {...oidcConfig}>
        <Provider store={store}>
          <RouterProvider router={router} />
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
