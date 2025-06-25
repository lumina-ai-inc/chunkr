/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/components/UserProfileFormFields/FieldErrors.tsx" --revert
 */

import { Fragment } from "react";
import type { FormFieldError } from "@keycloakify/login-ui/useUserProfileForm";
import type { Attribute } from "@keycloakify/login-ui/KcContext";
import { Text } from "@radix-ui/themes";

export function FieldErrors(props: {
    attribute: Attribute;
    displayableErrors: FormFieldError[];
    fieldIndex: number | undefined;
}) {
    const { attribute, fieldIndex } = props;

    const displayableErrors = props.displayableErrors.filter(error => error.fieldIndex === fieldIndex);

    if (displayableErrors.length === 0) {
        return null;
    }

    return (
        <Text 
            className="text-red-500 text-sm block"
            id={`input-error-${attribute.name}${fieldIndex === undefined ? "" : `-${fieldIndex}`}`}
            aria-live="polite"
        >
            {displayableErrors
                .filter(error => error.fieldIndex === fieldIndex)
                .map(({ errorMessage }, i, arr) => (
                    <Fragment key={i}>
                        {errorMessage}
                        {arr.length - 1 !== i && <br />}
                    </Fragment>
                ))}
        </Text>
    );
}
