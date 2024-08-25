import React from "react";
import ReactDOM from "react-dom/client";
import { Home } from "./pages/Home.tsx";
import { Viewer } from "./pages/Viewer.tsx";

import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "@radix-ui/themes/styles.css";

import { Theme } from "@radix-ui/themes";

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
      accentColor="sky"
      panelBackground="solid"
      style={{ height: "100%" }}
    >
      <RouterProvider router={router} />
    </Theme>
  </React.StrictMode>
);
