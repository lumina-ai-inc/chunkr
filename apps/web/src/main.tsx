import ReactDOM from "react-dom/client";
import "./index.css";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Theme } from "@radix-ui/themes";
import "@radix-ui/themes/styles.css";
import { Home } from "./pages/Home/Home.tsx";
import Task from "./pages/Task/Task.tsx";
import { initializeKeycloak, keycloak } from "./auth/KeycloakProvider";
import AuthGuard from "./auth/AuthGuard.tsx";

initializeKeycloak()
  .then()
  .catch((error) => {
    console.error("Failed to initialize Keycloak:", error);
  });

const router = createBrowserRouter([
  {
    path: "/",
    element: <>
      <button style={{ backgroundColor: 'blue' }} onClick={() => { keycloak.login(); }}>Login</button>
      <button style={{ backgroundColor: 'red' }} onClick={() => { keycloak.logout(); }}>Logout</button>
      <Home />
    </>,
  },
  {
    path: "/task/:taskId/:pageCount",
    element: <AuthGuard><Task /></AuthGuard>,
  },
  // {
  //   path: "/pricing",
  //   element: <Pricing />,
  // },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <Theme
    scaling="100%"
    accentColor="cyan"
    panelBackground="solid"
    style={{
      height: "100%",
      width: "100%",
      backgroundColor: "hsl(192, 70%, 5%)",
    }}
  >
    <RouterProvider router={router} />
  </Theme>
);
