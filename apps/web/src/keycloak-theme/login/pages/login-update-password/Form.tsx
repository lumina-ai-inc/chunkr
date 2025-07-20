/**
 * Form component for the update-password page, styled like login-reset-password.
 */

import { kcSanitize } from "@keycloakify/login-ui/kcSanitize";
import { assert } from "tsafe/assert";
import { useKcContext } from "../../KcContext";
import { useI18n } from "../../i18n";
import { Text } from "@radix-ui/themes";
import { PasswordWrapper } from "../../components/PasswordWrapper";
import { LogoutOtherSessions } from "../../components/LogoutOtherSessions";

export function Form() {
  const { kcContext } = useKcContext();
  assert(kcContext.pageId === "login-update-password.ftl");

  const { msg, msgStr } = useI18n();

  const showPasswordError = kcContext.messagesPerField.existsError("password");
  const showConfirmError =
    kcContext.messagesPerField.existsError("password-confirm");

  return (
    <div id="kc-form" className="w-full">
      <div id="kc-form-wrapper" className="space-y-6">
        <form
          id="kc-passwd-update-form"
          action={kcContext.url.loginAction}
          method="post"
          className="space-y-6"
        >
          <div className="space-y-2">
            <label htmlFor="password-new" className="block">
              <Text as="span" size="2" weight="medium">
                {msg("passwordNew")}
              </Text>
            </label>
            <div className="relative">
              <PasswordWrapper passwordInputId="password-new">
                <input
                  type="password"
                  id="password-new"
                  name="password-new"
                  autoFocus
                  autoComplete="new-password"
                  className="flex h-10 w-full px-2 py-3 outline-none rounded-md text-sm text-white bg-white/5 border border-white/20 transition-all duration-200 box-border placeholder:text-white/50 focus:border-gray-400 focus:bg-white/8 focus:shadow-[0_0_0_3px_rgba(139,139,139,0.1)] hover:border-white/30 hover:bg-white/7 disabled:cursor-not-allowed disabled:opacity-50"
                  aria-invalid={showPasswordError}
                />
              </PasswordWrapper>
            </div>
            {showPasswordError && (
              <Text
                size="1"
                color="red"
                className="text-red-600 text-sm mt-1 block"
                dangerouslySetInnerHTML={{
                  __html: kcSanitize(
                    kcContext.messagesPerField.get("password")
                  ),
                }}
              />
            )}
          </div>

          <div className="space-y-2">
            <label htmlFor="password-confirm" className="block">
              <Text as="span" size="2" weight="medium">
                {msg("passwordConfirm")}
              </Text>
            </label>
            <div className="relative">
              <PasswordWrapper passwordInputId="password-confirm">
                <input
                  type="password"
                  id="password-confirm"
                  name="password-confirm"
                  autoComplete="new-password"
                  className="flex h-10 w-full px-2 py-3 outline-none rounded-md text-sm text-white bg-white/5 border border-white/20 transition-all duration-200 box-border placeholder:text-white/50 focus:border-gray-400 focus:bg-white/8 focus:shadow-[0_0_0_3px_rgba(139,139,139,0.1)] hover:border-white/30 hover:bg-white/7 disabled:cursor-not-allowed disabled:opacity-50"
                  aria-invalid={showConfirmError}
                />
              </PasswordWrapper>
            </div>
            {showConfirmError && (
              <Text
                size="1"
                color="red"
                className="text-red-600 text-sm mt-1 block"
                dangerouslySetInnerHTML={{
                  __html: kcSanitize(
                    kcContext.messagesPerField.get("password-confirm")
                  ),
                }}
              />
            )}
          </div>

          <div>
            <LogoutOtherSessions />
          </div>

          <button
            type="submit"
            className="!w-full rounded-md font-medium button-resting !h-10"
          >
            {msgStr("doSubmit")}
          </button>

          {kcContext.isAppInitiatedAction && (
            <button
              type="submit"
              name="cancel-aia"
              value="true"
              className="!w-full rounded-md font-medium button-resting !h-10 mt-2"
            >
              {msg("doCancel")}
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
