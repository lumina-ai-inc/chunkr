import type { ReactNode } from "react";
import { Theme as RadixTheme } from "@radix-ui/themes";
import { Toaster } from "react-hot-toast";
import "@radix-ui/themes/styles.css";
import "./index.css";

export default function ThemeProvider(props: { children: ReactNode }) {
  const { children } = props;

  return (
    <RadixTheme
      scaling="100%"
      panelBackground="solid"
      style={{
        height: "100%",
        backgroundColor: "#020809",
      }}
    >
      {children}
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
    </RadixTheme>
  );
}