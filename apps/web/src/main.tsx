import React from "react";
import ReactDOM from "react-dom/client";
import { Home } from "./pages/Home.tsx";
import { Viewer } from "./pages/Viewer.tsx";
import "./index.css";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Theme } from "@radix-ui/themes";
import "@radix-ui/themes/styles.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/viewer",
    element: <Viewer />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Theme
      scaling="100%"
      accentColor="cyan"
      panelBackground="solid"
      style={{ height: "100%", width: "100%", backgroundColor: "#061d23" }}
    >
      <RouterProvider router={router} />
    </Theme>
  </React.StrictMode>
);
