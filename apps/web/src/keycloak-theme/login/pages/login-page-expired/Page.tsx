/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.1.
 * To relinquish ownership and restore this file to its original content, run the following command:
 *
 * $ npx keycloakify own --path "login/pages/login-page-expired/Page.tsx" --revert
 */

import { assert } from "tsafe/assert";
import { useKcContext } from "../../KcContext";
import { useI18n } from "../../i18n";
import { Template } from "../../components/Template/CustomTemplate";

export function Page() {
  const { kcContext } = useKcContext();
  assert(kcContext.pageId === "login-page-expired.ftl");

  const { msg } = useI18n();

  return (
    <Template headerNode={msg("pageExpiredTitle")}>
      <p id="instruction1" className="instruction">
        {msg("pageExpiredMsg1")}{" "}
        <a id="loginRestartLink" href={kcContext.url.loginRestartFlowUrl}>
          {msg("doClickHere")}
        </a>{" "}
        .<br />
        {msg("pageExpiredMsg2")}{" "}
        <a id="loginContinueLink" href={kcContext.url.loginAction}>
          {msg("doClickHere")}
        </a>{" "}
        .
      </p>
    </Template>
  );
}
