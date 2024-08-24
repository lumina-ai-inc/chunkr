import React from "react";
import ReactDOM from "react-dom/client";
import { Home } from "./pages/Home.tsx";
import { Viewer } from "./pages/Viewer.tsx";
import "./index.css";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

declare global {
  interface Window {
    plausible: any;
  }
}

window.plausible = window.plausible || {};

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
    <RouterProvider router={router} />
  </React.StrictMode>
);

