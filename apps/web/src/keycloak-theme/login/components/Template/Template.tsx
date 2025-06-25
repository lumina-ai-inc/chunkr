/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/components/Template/Template.tsx" --revert
 */

import type { ReactNode } from "react";
import { useEffect } from "react";
import { clsx } from "@keycloakify/login-ui/tools/clsx";
import { kcSanitize } from "@keycloakify/login-ui/kcSanitize";
import { useInitializeTemplate } from "./useInitializeTemplate";
import { useKcClsx } from "@keycloakify/login-ui/useKcClsx";
import { useI18n } from "../../i18n";
import { useKcContext } from "../../KcContext";

export function Template(props: {
    displayInfo?: boolean;
    displayMessage?: boolean;
    displayRequiredFields?: boolean;
    headerNode: ReactNode;
    socialProvidersNode?: ReactNode;
    infoNode?: ReactNode;
    documentTitle?: string;
    bodyClassName?: string;
    children: ReactNode;
}) {
    const {
        displayInfo = false,
        displayMessage = true,
        displayRequiredFields = false,
        headerNode,
        socialProvidersNode = null,
        infoNode = null,
        documentTitle,
        children
    } = props;

    const { kcContext } = useKcContext();

    const { msg, msgStr, currentLanguage, enabledLanguages } = useI18n();

    const { kcClsx } = useKcClsx();

    useEffect(() => {
        document.title = documentTitle ?? msgStr("loginTitle", kcContext.realm.displayName);
    }, [documentTitle, kcContext.realm.displayName, msgStr]);

    const { isReadyToRender } = useInitializeTemplate();

    if (!isReadyToRender) {
        return null;
    }

    return (
        <div className={kcClsx("kcLoginClass")}>
            <div id="kc-header" className={kcClsx("kcHeaderClass")}>
                <div id="kc-header-wrapper" className={kcClsx("kcHeaderWrapperClass")}>
                    {msg("loginTitleHtml", kcContext.realm.displayNameHtml)}
                </div>
            </div>
            <div className={kcClsx("kcFormCardClass")}>
                <header className={kcClsx("kcFormHeaderClass")}>
                    {enabledLanguages.length > 1 && (
                        <div className={kcClsx("kcLocaleMainClass")} id="kc-locale">
                            <div id="kc-locale-wrapper" className={kcClsx("kcLocaleWrapperClass")}>
                                <div
                                    id="kc-locale-dropdown"
                                    className={clsx(
                                        "menu-button-links",
                                        kcClsx("kcLocaleDropDownClass")
                                    )}
                                >
                                    <button
                                        tabIndex={1}
                                        id="kc-current-locale-link"
                                        aria-label={msgStr("languages")}
                                        aria-haspopup="true"
                                        aria-expanded="false"
                                        aria-controls="language-switch1"
                                    >
                                        {currentLanguage.label}
                                    </button>
                                    <ul
                                        role="menu"
                                        tabIndex={-1}
                                        aria-labelledby="kc-current-locale-link"
                                        aria-activedescendant=""
                                        id="language-switch1"
                                        className={kcClsx("kcLocaleListClass")}
                                    >
                                        {enabledLanguages.map(({ languageTag, label, href }, i) => (
                                            <li
                                                key={languageTag}
                                                className={kcClsx("kcLocaleListItemClass")}
                                                role="none"
                                            >
                                                <a
                                                    role="menuitem"
                                                    id={`language-${i + 1}`}
                                                    className={kcClsx("kcLocaleItemClass")}
                                                    href={href}
                                                >
                                                    {label}
                                                </a>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    )}
                    {(() => {
                        const node = !(
                            kcContext.auth !== undefined &&
                            kcContext.auth.showUsername &&
                            !kcContext.auth.showResetCredentials
                        ) ? (
                            <h1 id="kc-page-title">{headerNode}</h1>
                        ) : (
                            <div id="kc-username" className={kcClsx("kcFormGroupClass")}>
                                <label id="kc-attempted-username">
                                    {kcContext.auth.attemptedUsername}
                                </label>
                                <a
                                    id="reset-login"
                                    href={kcContext.url.loginRestartFlowUrl}
                                    aria-label={msgStr("restartLoginTooltip")}
                                >
                                    <div className="kc-login-tooltip">
                                        <i className={kcClsx("kcResetFlowIcon")}></i>
                                        <span className="kc-tooltip-text">
                                            {msg("restartLoginTooltip")}
                                        </span>
                                    </div>
                                </a>
                            </div>
                        );

                        if (displayRequiredFields) {
                            return (
                                <div className={kcClsx("kcContentWrapperClass")}>
                                    <div className={clsx(kcClsx("kcLabelWrapperClass"), "subtitle")}>
                                        <span className="subtitle">
                                            <span className="required">*</span>
                                            {msg("requiredFields")}
                                        </span>
                                    </div>
                                    <div className="col-md-10">{node}</div>
                                </div>
                            );
                        }

                        return node;
                    })()}
                </header>
                <div id="kc-content">
                    <div id="kc-content-wrapper">
                        {/* App-initiated actions should not see warning messages about the need to complete the action during login. */}
                        {displayMessage &&
                            kcContext.message !== undefined &&
                            (kcContext.message.type !== "warning" ||
                                !kcContext.isAppInitiatedAction) && (
                                <div
                                    className={clsx(
                                        `alert-${kcContext.message.type}`,
                                        kcClsx("kcAlertClass"),
                                        `pf-m-${kcContext.message?.type === "error" ? "danger" : kcContext.message.type}`
                                    )}
                                >
                                    <div className="pf-c-alert__icon">
                                        {kcContext.message.type === "success" && (
                                            <span className={kcClsx("kcFeedbackSuccessIcon")}></span>
                                        )}
                                        {kcContext.message.type === "warning" && (
                                            <span className={kcClsx("kcFeedbackWarningIcon")}></span>
                                        )}
                                        {kcContext.message.type === "error" && (
                                            <span className={kcClsx("kcFeedbackErrorIcon")}></span>
                                        )}
                                        {kcContext.message.type === "info" && (
                                            <span className={kcClsx("kcFeedbackInfoIcon")}></span>
                                        )}
                                    </div>
                                    <span
                                        className={kcClsx("kcAlertTitleClass")}
                                        dangerouslySetInnerHTML={{
                                            __html: kcSanitize(kcContext.message.summary)
                                        }}
                                    />
                                </div>
                            )}
                        {children}
                        {kcContext.auth !== undefined && kcContext.auth.showTryAnotherWayLink && (
                            <form
                                id="kc-select-try-another-way-form"
                                action={kcContext.url.loginAction}
                                method="post"
                            >
                                <div className={kcClsx("kcFormGroupClass")}>
                                    <input type="hidden" name="tryAnotherWay" value="on" />
                                    <a
                                        href="#"
                                        id="try-another-way"
                                        onClick={() => {
                                            document.forms[
                                                "kc-select-try-another-way-form" as never
                                            ].submit();
                                            return false;
                                        }}
                                    >
                                        {msg("doTryAnotherWay")}
                                    </a>
                                </div>
                            </form>
                        )}
                        {socialProvidersNode}
                        {displayInfo && (
                            <div id="kc-info" className={kcClsx("kcSignUpClass")}>
                                <div id="kc-info-wrapper" className={kcClsx("kcInfoAreaWrapperClass")}>
                                    {infoNode}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
