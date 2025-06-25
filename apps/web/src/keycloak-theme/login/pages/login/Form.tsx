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
import { Checkbox, Text } from "@radix-ui/themes";

export function Form() {
  const { kcContext } = useKcContext();
  assert(kcContext.pageId === "login.ftl");

  const { msg, msgStr } = useI18n();
  const [isLoginButtonDisabled, setIsLoginButtonDisabled] = useState(false);

  const showFieldError = kcContext.messagesPerField.existsError("username", "password");
  const fieldErrorMessage = kcContext.messagesPerField.getFirstError("username", "password");

  return (
    <div id="kc-form" className="w-full">
      <div id="kc-form-wrapper" className="space-y-6">
        {kcContext.realm.password && (
          <form
            id="kc-form-login"
            onSubmit={() => {
              setIsLoginButtonDisabled(true);
              return true;
            }}
            action={kcContext.url.loginAction}
            method="post"
            className="space-y-6"
          >
            {!kcContext.usernameHidden && (
              <div className="space-y-2">
                <label htmlFor="username" className="block">
                  <Text as="span" weight="medium" className="text-sm">
                    {!kcContext.realm.loginWithEmailAllowed
                      ? msg("username")
                      : !kcContext.realm.registrationEmailAsUsername
                      ? msg("usernameOrEmail")
                      : msg("email")}
                  </Text>
                </label>
                <div className="relative">
                  <input
                    tabIndex={2}
                    id="username"
                    name="username"
                    defaultValue={kcContext.login.username ?? ""}
                    type="text"
                    autoFocus
                    autoComplete="off"
                    aria-invalid={showFieldError}
                    className={`flex h-10 w-full px-2 py-3 outline-none rounded-md text-sm ring-offset-background focus-visible:ring-1 focus-visible:ring-gray-300 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50 ${
                      showFieldError ? "bg-red-50 focus-visible:ring-red-500 text-red-500" : ""
                    }`}
                  />
                </div>
                {showFieldError && (
                  <Text 
                    size="1" 
                    color="red" 
                    className="text-red-600 text-sm mt-1 block"
                    dangerouslySetInnerHTML={{
                      __html: kcSanitize(fieldErrorMessage),
                    }} 
                  />
                )}
              </div>
            )}

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label htmlFor="password" className="block">
                  <Text as="span" weight="medium" className="text-sm">
                    {msg("password")}
                  </Text>
                </label>
                {kcContext.realm.resetPasswordAllowed && (
                  <Text size="2" as="span">
                    <a 
                      tabIndex={6} 
                      href={kcContext.url.loginResetCredentialsUrl}
                      className="hover:underline transition-colors focus:outline-none focus:ring-0"
                    >
                      {msg("doForgotPassword")}
                    </a>
                  </Text>
                )}
              </div>
       
                <PasswordWrapper passwordInputId="password">
                  <input
                    tabIndex={3}
                    id="password"
                    name="password"
                    type="password"
                    autoComplete="current-password"
                    aria-invalid={showFieldError}
                    className={`flex h-10 w-full px-2 py-3 outline-none rounded-md text-sm ring-offset-background focus-visible:ring-1 focus-visible:ring-gray-300 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50 ${
                      showFieldError ? "bg-red-50 focus-visible:ring-red-500 text-red-500" : ""
                    }`}
                  />
                </PasswordWrapper>
                

              {kcContext.usernameHidden && showFieldError && (
                <Text 
                  size="1" 
                  color="red" 
                  className="text-red-600 text-sm mt-1 block"
                  dangerouslySetInnerHTML={{
                    __html: kcSanitize(fieldErrorMessage),
                  }} 
                />
              )}
            </div>

            {kcContext.realm.rememberMe && !kcContext.usernameHidden && (
              <div className="flex items-center space-x-2">
                <Checkbox
                  tabIndex={5}
                  id="rememberMe"
                  name="rememberMe"
                  defaultChecked={!!kcContext.login.rememberMe}
                  color="gray"
                  className="rounded-sm focus:ring-0 focus:outline-none"
                />
                <label htmlFor="rememberMe" className="cursor-pointer">
                  <Text as="span" size="2">
                    {msg("rememberMe")}
                  </Text>
                </label>
              </div>
            )}

         
              <input
                type="hidden"
                id="id-hidden-input"
                name="credentialId"
                value={kcContext.auth.selectedCredential}
              />
              <button
                tabIndex={7}
                type="submit"
                id="kc-login"
                name="login"
                disabled={isLoginButtonDisabled}
                className={`w-full rounded-md font-medium ${
                  isLoginButtonDisabled
                    ? "opacity-50 cursor-not-allowed"
                    : ""
                }`}
              >
                {msgStr("doLogIn")}
              </button>
          
          </form>
        )}
      </div>
    </div>
  );
}