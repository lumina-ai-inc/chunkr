import React from "react";
import ReactDOM from "react-dom/client";
import { Home } from "./pages/Home/Home.tsx";
import "./index.css";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Theme } from "@radix-ui/themes";
import "@radix-ui/themes/styles.css";
import Task from "./pages/Task/Task.tsx";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/task/:taskId/:pageCount",
    element: <Task />,
  },
  // {
  //   path: "/pricing",
  //   element: <Pricing />,
  // },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
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
  </React.StrictMode>
);
