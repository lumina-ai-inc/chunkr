/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/pages/login/Form.tsx" --revert
 */

import { useState } from "react";
import { assert } from "tsafe/assert";
import { PasswordWrapper } from "../../components/PasswordWrapper";
import { useI18n } from "../../i18n";
import { useKcContext } from "../../KcContext";
import { kcSanitize } from "@keycloakify/login-ui/kcSanitize";
import {
  Box,
  Button,
  Checkbox,
  Flex,
  Text,
  TextField,
} from "@radix-ui/themes";

export function Form() {
  const { kcContext } = useKcContext();
  assert(kcContext.pageId === "login.ftl");

  const { msg, msgStr } = useI18n();
  const [isLoginButtonDisabled, setIsLoginButtonDisabled] = useState(false);

  const showFieldError = kcContext.messagesPerField.existsError("username", "password");
  const fieldErrorMessage = kcContext.messagesPerField.getFirstError("username", "password");

  return (
    <Box id="kc-form" width="100%">
      <Box asChild id="kc-form-wrapper">
        {kcContext.realm.password && (
          <form
            id="kc-form-login"
            onSubmit={() => {
              setIsLoginButtonDisabled(true);
              return true;
            }}
            action={kcContext.url.loginAction}
            method="post"
          >
            {!kcContext.usernameHidden && (
              <Flex direction="column" gap="2" mb="4">
                <label htmlFor="username">
                  <Text as="span" size="2" weight="medium">
                    {!kcContext.realm.loginWithEmailAllowed
                      ? msg("username")
                      : !kcContext.realm.registrationEmailAsUsername
                      ? msg("usernameOrEmail")
                      : msg("email")}
                  </Text>
                </label>
                <TextField.Root
                  tabIndex={2}
                  id="username"
                  name="username"
                  defaultValue={kcContext.login.username ?? ""}
                  type="text"
                  autoFocus
                  autoComplete="username"
                  aria-invalid={showFieldError}
                />
                {showFieldError && (
                  <Text size="1" color="red" dangerouslySetInnerHTML={{
                    __html: kcSanitize(fieldErrorMessage),
                  }} />
                )}
              </Flex>
            )}

            <Flex direction="column" gap="2" mb="4">
              <label htmlFor="password">
                <Text as="span" size="2" weight="medium">
                  {msg("password")}
                </Text>
              </label>
              <PasswordWrapper passwordInputId="password">
                <TextField.Root
                  tabIndex={3}
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  aria-invalid={showFieldError}
                />
              </PasswordWrapper>
              {kcContext.usernameHidden && showFieldError && (
                <Text size="1" color="red" dangerouslySetInnerHTML={{
                  __html: kcSanitize(fieldErrorMessage),
                }} />
              )}
            </Flex>

            <Flex justify="between" align="center" mb="4">
              {kcContext.realm.rememberMe && !kcContext.usernameHidden && (
                <Flex align="center" gap="2">
                  <Checkbox
                    tabIndex={5}
                    id="rememberMe"
                    name="rememberMe"
                    defaultChecked={!!kcContext.login.rememberMe}
                  />
                  <label htmlFor="rememberMe">
                    <Text as="span" size="2">
                      {msg("rememberMe")}
                    </Text>
                  </label>
                </Flex>
              )}

              {kcContext.realm.resetPasswordAllowed && (
                <Text size="2" as="span">
                  <a tabIndex={6} href={kcContext.url.loginResetCredentialsUrl}>
                    {msg("doForgotPassword")}
                  </a>
                </Text>
              )}
            </Flex>

            <Flex direction="column" gap="3">
              <input
                type="hidden"
                id="id-hidden-input"
                name="credentialId"
                value={kcContext.auth.selectedCredential}
              />
              <Button
                tabIndex={7}
                type="submit"
                id="kc-login"
                name="login"
                disabled={isLoginButtonDisabled}
                variant="solid"
                size="3"
              >
                {msgStr("doLogIn")}
              </Button>
            </Flex>
          </form>
        )}
      </Box>
    </Box>
  );
}