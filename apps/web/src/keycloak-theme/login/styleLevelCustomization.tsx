/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 *
 * $ npx keycloakify own --path "login/styleLevelCustomization.tsx" --revert
 */

import { Suspense, lazy, type ReactNode } from "react";
import type { ClassKey } from "@keycloakify/login-ui/useKcClsx";
import { useKcContext } from "./KcContext";

const ThemeProvider = lazy(() => import("../../theme-provider"));

type Classes = { [key in ClassKey]?: string };

type StyleLevelCustomization = {
  doUseDefaultCss: boolean;
  classes?: Classes;
  loadCustomStylesheet?: () => void;
  Provider?: (props: { children: ReactNode }) => ReactNode;
};

export function useStyleLevelCustomization(): StyleLevelCustomization {
  const { kcContext } = useKcContext();

  if (kcContext.pageId === "login.ftl") {
    return {
      doUseDefaultCss: false,
      Provider: Provider,
    };
  }

  return {
    doUseDefaultCss: true,
  };
}

// eslint-disable-next-line react-refresh/only-export-components
function Provider(props: { children: ReactNode }) {
  const { children } = props;
  return (
    <Suspense>
      <ThemeProvider>{children}</ThemeProvider>
    </Suspense>
  );
}