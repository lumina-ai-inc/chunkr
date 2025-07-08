import ReactDOM from "react-dom/client";
import { Provider } from "react-redux";
import ThemeProvider from "./theme-provider.tsx";
import { AuthProvider, AuthProviderProps } from "react-oidc-context";
import {
  createBrowserRouter,
  RouterProvider,
  Outlet,
  RouteObject,
} from "react-router-dom";
import { QueryClient, QueryClientProvider } from "react-query";
import Auth from "./auth/Auth.tsx";
import Home from "./pages/Home/Home.tsx";
import AuthGuard from "./auth/AuthGuard.tsx";
import store from "./store/store";
import Dashboard from "./pages/Dashboard/Dashboard.tsx";
import Checkout from "./pages/Checkout/Checkout";
import Blog from "./pages/Blog/Blog.tsx";
import BlogPostPage from "./pages/BlogPostPage/BlogPostPage";
import Onboarding from "./pages/Onboarding/Onboarding.tsx";
const isSelfHost = import.meta.env.VITE_IS_SELF_HOST === "true";

const oidcConfig: AuthProviderProps = {
  authority:
    import.meta.env.VITE_KEYCLOAK_URL +
    "/realms/" +
    import.meta.env.VITE_KEYCLOAK_REALM,
  client_id: import.meta.env.VITE_KEYCLOAK_CLIENT_ID,
  redirect_uri: import.meta.env.VITE_KEYCLOAK_REDIRECT_URI,
  post_logout_redirect_uri: import.meta.env
    .VITE_KEYCLOAK_POST_LOGOUT_REDIRECT_URI,
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
    path: "/",
    element: <Outlet />,
    children: [
      {
        index: true,
        element: <Home />,
      },
      !isSelfHost && {
        path: "blog",
        element: <Blog />,
      },
      !isSelfHost && {
        path: "blog/:slug",
        element: <BlogPostPage />,
      },
      {
        path: "dashboard",
        element: (
          <AuthGuard>
            <Dashboard />
          </AuthGuard>
        ),
      },
      {
        path: "checkout/return",
        element: (
          <AuthGuard>
            <Checkout />
          </AuthGuard>
        ),
      },
      {
        path: "onboarding",
        element: (
          <AuthGuard>
            <Onboarding />
          </AuthGuard>
        ),
      },
    ].filter(Boolean) as RouteObject[],
  },
]);

const queryClient = new QueryClient();
ReactDOM.createRoot(document.getElementById("root")!).render(
  <ThemeProvider>
    <QueryClientProvider client={queryClient}>
      <AuthProvider {...oidcConfig}>
        <Provider store={store}>
          <Auth>
            <RouterProvider router={router} />
          </Auth>
        </Provider>
      </AuthProvider>
    </QueryClientProvider>
  </ThemeProvider>
);
