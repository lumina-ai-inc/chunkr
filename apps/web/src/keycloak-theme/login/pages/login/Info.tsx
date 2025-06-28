/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/pages/login/Info.tsx" --revert
 */

import { assert } from "tsafe/assert";
import { useI18n } from "../../i18n";
import { useKcContext } from "../../KcContext";
import { Text } from "@radix-ui/themes";

export function Info() {
    const { kcContext } = useKcContext();
    assert(kcContext.pageId === "login.ftl");

    const { url } = kcContext;

    const { msg } = useI18n();

    return (
        <div className="text-center">
            <Text as="span" className="text-gray-300 text-sm">
                {msg("noAccount")}{" "}
                <a 
                    tabIndex={8} 
                    href={url.registrationUrl}
                    className="hover:underline focus:outline-none focus:ring-0"
                >
                    {msg("doRegister")}
                </a>
            </Text>
        </div>
    );
}
