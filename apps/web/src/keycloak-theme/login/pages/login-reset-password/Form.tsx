/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/pages/login-reset-password/Form.tsx" --revert
 */

import { kcSanitize } from "@keycloakify/login-ui/kcSanitize";
import { assert } from "tsafe/assert";
import { useKcContext } from "../../KcContext";
import { useI18n } from "../../i18n";
import { Text } from "@radix-ui/themes";

export function Form() {
    const { kcContext } = useKcContext();
    assert(kcContext.pageId === "login-reset-password.ftl");

    const { msg, msgStr } = useI18n();

    const showFieldError = kcContext.messagesPerField.existsError("username");

    return (
        <div id="kc-form" className="w-full">
            <div id="kc-form-wrapper" className="space-y-6">
                <form
                    id="kc-reset-password-form"
                    action={kcContext.url.loginAction}
                    method="post"
                    className="space-y-6"
                >
                    <div className="space-y-2">
                        <label htmlFor="username" className="block">
                            <Text as="span" size="2" weight="medium">
                                {!kcContext.realm.loginWithEmailAllowed
                                    ? msg("username")
                                    : !kcContext.realm.registrationEmailAsUsername
                                      ? msg("usernameOrEmail")
                                      : msg("email")}
                            </Text>
                        </label>
                        <div className="relative">
                            <input
                                type="text"
                                id="username"
                                name="username"
                                autoFocus
                                defaultValue={kcContext.auth.attemptedUsername ?? ""}
                                aria-invalid={showFieldError}
                                className={`flex h-10 w-full px-2 py-3 outline-none rounded-md text-sm text-white bg-white/5 border border-white/20 transition-all duration-200 box-border placeholder:text-white/50 focus:border-gray-400 focus:bg-white/8 focus:shadow-[0_0_0_3px_rgba(139,139,139,0.1)] hover:border-white/30 hover:bg-white/7 disabled:cursor-not-allowed disabled:opacity-50 ${
                                    showFieldError ? "border-red-500 bg-red-50 text-red-500" : ""
                                }`}
                            />
                        </div>
                        {showFieldError && (
                            <Text 
                                size="1" 
                                color="red" 
                                className="text-red-600 text-sm mt-1 block"
                                dangerouslySetInnerHTML={{
                                    __html: kcSanitize(kcContext.messagesPerField.get("username")),
                                }} 
                            />
                        )}
                    </div>

                    <div className="flex items-center justify-between">
                        <Text size="2" as="span">
                            <a 
                                href={kcContext.url.loginUrl}
                                className="hover:underline transition-colors focus:outline-none focus:ring-0"
                            >
                                {msg("backToLogin")}
                            </a>
                        </Text>
                    </div>

                    <button
                        type="submit"
                        className="!w-full rounded-md font-medium button-resting !h-10"
                    >
                        {msgStr("doSubmit")}
                    </button>
                </form>
            </div>
        </div>
    );
}
