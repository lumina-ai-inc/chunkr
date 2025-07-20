/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.1.
 * To relinquish ownership and restore this file to its original content, run the following command:
 *
 * $ npx keycloakify own --path "login/pages/login-update-password/Page.tsx" --revert
 */

import { assert } from "tsafe/assert";
import { Template } from "../../components/Template/CustomTemplate";
import { Form } from "./Form";
import { useKcContext } from "../../KcContext";
import { useI18n } from "../../i18n";

export function Page() {
  const { kcContext } = useKcContext();
  assert(kcContext.pageId === "login-update-password.ftl");

  const { msg } = useI18n();
  const { messagesPerField } = kcContext;

  return (
    <Template
      displayInfo
      displayMessage={
        !messagesPerField.existsError("password", "password-confirm")
      }
      headerNode={msg("updatePasswordTitle")}
    >
      <Form />
    </Template>
  );
}
