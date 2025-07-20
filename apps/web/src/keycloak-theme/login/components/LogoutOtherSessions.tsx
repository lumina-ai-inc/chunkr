/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.1.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/components/LogoutOtherSessions.tsx" --revert
 */

import { useI18n } from "../i18n";
import { useKcClsx } from "@keycloakify/login-ui/useKcClsx";
import { Checkbox, Text } from "@radix-ui/themes";

export function LogoutOtherSessions() {
  const { msg } = useI18n();

  const { kcClsx } = useKcClsx();

  return (
    <div id="kc-form-options" className={kcClsx("kcFormOptionsClass")}>
      <div className={kcClsx("kcFormOptionsWrapperClass")}>
        <div className="flex items-center space-x-2">
          <Checkbox
            id="logout-sessions"
            name="logout-sessions"
            defaultChecked
            color="gray"
            className="rounded-sm focus:ring-0 focus:outline-none"
          />
          <label htmlFor="logout-sessions" className="cursor-pointer">
            <Text as="span" size="2">
              {msg("logoutOtherSessions")}
            </Text>
          </label>
        </div>
      </div>
    </div>
  );
}

