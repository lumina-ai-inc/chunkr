/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/components/PasswordWrapper.tsx" --revert
 */

import type { JSX } from "@keycloakify/login-ui/tools/JSX";
import { useIsPasswordRevealed } from "@keycloakify/login-ui/tools/useIsPasswordRevealed";
import { EyeOpenIcon, EyeClosedIcon } from "@radix-ui/react-icons";
import { useI18n } from "../i18n";

export function PasswordWrapper(props: { passwordInputId: string; children: JSX.Element }) {
    const { passwordInputId, children } = props;

    const { msgStr } = useI18n();

    const { isPasswordRevealed, toggleIsPasswordRevealed } = useIsPasswordRevealed({
        passwordInputId
    });

    return (
        <div className="flex flex-row items-center justify-center space-x-2">
            {children}
            <button
                type="button"
                className="h-10 w-10 p-1 flex items-center justify-center rounded-md focus:outline-none focus:ring-0"
                aria-label={msgStr(isPasswordRevealed ? "hidePassword" : "showPassword")}
                aria-controls={passwordInputId}
                onClick={toggleIsPasswordRevealed}
            >
                {isPasswordRevealed ? <EyeClosedIcon className="h-4 w-4" /> : <EyeOpenIcon className="h-4 w-4" />}
            </button>
        </div>
    );
}
