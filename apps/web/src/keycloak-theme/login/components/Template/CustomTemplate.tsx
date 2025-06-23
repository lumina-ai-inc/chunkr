/**Add commentMore actions
 * WARNING: Before modifying this file, run the following command:
 *
 * $ npx keycloakify own --path "login/components/Template/Template.tsx"
 *
 * This file is provided by @keycloakify/login-ui version 250004.1.0.
 * It was copied into your repository by the postinstall script: `keycloakify sync-extensions`.
 */

/* eslint-disable */

import type { ReactNode } from "react";
import { useEffect } from "react";
import { clsx } from "@keycloakify/login-ui/tools/clsx";
import { kcSanitize } from "@keycloakify/login-ui/kcSanitize";
import { useSetClassName } from "@keycloakify/login-ui/tools/useSetClassName";
import { useInitializeTemplate } from "./useInitializeTemplate";
import { useKcClsx } from "@keycloakify/login-ui/useKcClsx";
import { useI18n } from "../../i18n";
import { useKcContext } from "../../KcContext";
import {
  Box,
  Button,
  Card,
  Flex,
  Heading,
  Link,
  Text,
  TextField,
} from "@radix-ui/themes";

export function Template(props: {
  displayInfo?: boolean;
  displayMessage?: boolean;
  displayRequiredFields?: boolean;
  headerNode: ReactNode;
  socialProvidersNode?: ReactNode;
  infoNode?: ReactNode;
  documentTitle?: string;
  bodyClassName?: string;
  children: ReactNode;
}) {
  const {
    displayInfo = false,
    displayMessage = true,
    displayRequiredFields = false,
    headerNode,
    socialProvidersNode = null,
    infoNode = null,
    documentTitle,
    bodyClassName,
    children,
  } = props;

  const { kcContext } = useKcContext();

  const { msg, msgStr, currentLanguage, enabledLanguages } = useI18n();

  const { kcClsx } = useKcClsx();

  useEffect(() => {
    document.title =
      documentTitle ?? msgStr("loginTitle", kcContext.realm.displayName);
  }, []);

  useSetClassName({
    qualifiedName: "html",
    className: kcClsx("kcHtmlClass"),
  });

  useSetClassName({
    qualifiedName: "body",
    className: bodyClassName ?? kcClsx("kcBodyClass"),
  });

  const { isReadyToRender } = useInitializeTemplate();

  if (!isReadyToRender) {
    return null;
  }

  const tabIndex = undefined;

  return (
    <Flex
      height="100vh"
      width="100%" // use percentage, not viewport units
      align="center" // horizontal alignment
      justify="center" // vertical alignment
      direction="column"
      gap="6"
    >
      <Card size="4" style={{ width: 416 }}>
        <Heading as="h3" size="6" trim="start" mb="5">
          Sign up
        </Heading>

        <Box mb="5">
          <Flex mb="1">
            <Text
              as="label"
              htmlFor="example-email-field"
              size="2"
              weight="bold"
            >
              Email address
            </Text>
          </Flex>
          <TextField.Root
            tabIndex={tabIndex}
            placeholder="Enter your email"
            id="example-email-field"
          />
        </Box>

        <Box mb="5" position="relative">
          <Flex align="baseline" justify="between" mb="1">
            <Text
              as="label"
              size="2"
              weight="bold"
              htmlFor="example-password-field"
            >
              Password
            </Text>
            <Link
              href="#"
              tabIndex={tabIndex}
              size="2"
              onClick={(e) => e.preventDefault()}
            >
              Forgot password?
            </Link>
          </Flex>
          <TextField.Root
            tabIndex={tabIndex}
            placeholder="Enter your password"
            id="example-password-field"
          />
        </Box>

        <Flex mt="6" justify="end" gap="3">
          <Button tabIndex={tabIndex} variant="outline">
            Create an account
          </Button>
          <Button tabIndex={tabIndex}>Sign in</Button>
        </Flex>
      </Card>
    </Flex>
  );
}