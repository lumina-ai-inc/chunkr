/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.1.
 * To relinquish ownership and restore this file to its original content, run the following command:
 *
 * $ npx keycloakify own --path "login/pages/error/Page.tsx" --revert
 */

import { assert } from "tsafe/assert";
import { kcSanitize } from "@keycloakify/login-ui/kcSanitize";
import { useKcContext } from "../../KcContext";
import { useI18n } from "../../i18n";
import { Template } from "../../components/Template/CustomTemplate";

export function Page() {
  const { kcContext } = useKcContext();
  assert(kcContext.pageId === "error.ftl");

  const { msg } = useI18n();

  return (
    <Template displayMessage={false} headerNode={msg("errorTitle")}>
      <div id="kc-error-message" className="space-y-4">
        <p
          className="instruction text-sm"
          dangerouslySetInnerHTML={{
            __html: kcSanitize(kcContext.message.summary),
          }}
        />
        {!kcContext.skipLink &&
          kcContext.client !== undefined &&
          kcContext.client.baseUrl !== undefined && (
            <p>
              <a id="backToApplication" href={kcContext.client.baseUrl}>
                {msg("backToApplication")}
              </a>
            </p>
          )}
      </div>
    </Template>
  );
}
