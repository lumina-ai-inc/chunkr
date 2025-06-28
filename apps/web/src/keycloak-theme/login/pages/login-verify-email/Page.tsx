/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/pages/login-verify-email/Page.tsx" --revert
 */

import { assert } from "tsafe/assert";
import { useKcContext } from "../../KcContext";
import { useI18n } from "../../i18n";
import { Template } from "../../components/Template/CustomTemplate";

export function Page() {
    const { kcContext } = useKcContext();
    assert(kcContext.pageId === "login-verify-email.ftl");

    const { msg } = useI18n();

    const { url, user } = kcContext;

    return (
        <Template
            displayInfo
            headerNode={msg("emailVerifyTitle")}
            infoNode={
                <p className="instruction text-gray-300 text-center">
                    {msg("emailVerifyInstruction2")}
                    <br />
                    <a href={url.loginAction} className="hover:underline transition-colors focus:outline-none focus:ring-0">{msg("doClickHere")}</a>
                    &nbsp;
                    {msg("emailVerifyInstruction3")}
                </p>
            }
        >
            <p className="instruction">{msg("emailVerifyInstruction1", user?.email ?? "")}</p>
        </Template>
    );
}
