import ReactDOM from "react-dom/client";
import "./index.css";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Theme } from "@radix-ui/themes";
import "@radix-ui/themes/styles.css";
import { Toaster } from "react-hot-toast";
import { AuthProvider, AuthProviderProps } from "react-oidc-context";
import Auth from "./auth/Auth.tsx";
import Home from "./pages/Home/Home.tsx";
import Task from "./pages/Task/Task.tsx";
import AuthGuard from "./auth/AuthGuard.tsx";
import Pricing from "./pages/Pricing/Pricing.tsx";
import { Provider } from "react-redux";
import store from "./store/store";

const oidcConfig: AuthProviderProps = {
  authority: import.meta.env.VITE_KEYCLOAK_URL,
  client_id: import.meta.env.VITE_KEYCLOAK_CLIENT_ID,
  redirect_uri: import.meta.env.VITE_KEYCLOAK_REDIRECT_URI,
  post_logout_redirect_uri: import.meta.env
    .VITE_KEYCLOAK_POST_LOGOUT_REDIRECT_URI,
  onSigninCallback: () => {
    window.history.replaceState({}, document.title, window.location.pathname);
  },
};

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/task/:taskId/:pageCount",
    element: (
      <AuthGuard>
        <Task />
      </AuthGuard>
    ),
  },
  {
    path: "/pricing",
    element: <Pricing />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <Theme
    scaling="100%"
    accentColor="cyan"
    panelBackground="solid"
    style={{
      height: "100%",
      backgroundColor: "hsl(192, 70%, 5%)",
    }}
  >
    <AuthProvider {...oidcConfig}>
      <Auth>
        <Provider store={store}>
          <RouterProvider router={router} />
        </Provider>
      </Auth>
    </AuthProvider>
    <Toaster />
  </Theme>
);
