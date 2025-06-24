/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/components/UserProfileFormFields/UserProfileFormFields.tsx" --revert
 */

import type { JSX } from "@keycloakify/login-ui/tools/JSX";
import { useEffect, Fragment } from "react";
import {
    useUserProfileForm,
    type FormAction,
    type FormFieldError
} from "@keycloakify/login-ui/useUserProfileForm";
import { GroupLabel } from "./GroupLabel";
import { FieldErrors } from "./FieldErrors";
import { InputFieldByType } from "./InputFieldByType";
import type { Attribute } from "@keycloakify/login-ui/KcContext";
import { useKcContext } from "../../KcContext";
import { useI18n } from "../../i18n";
import { DO_MAKE_USER_CONFIRM_PASSWORD } from "./DO_MAKE_USER_CONFIRM_PASSWORD";
import { assert } from "tsafe/assert";
import { Text } from "@radix-ui/themes";

export type UserProfileFormFieldsProps = {
    onIsFormSubmittableValueChange: (isFormSubmittable: boolean) => void;
    BeforeField?: (props: BeforeAfterFieldProps) => JSX.Element | null;
    AfterField?: (props: BeforeAfterFieldProps) => JSX.Element | null;
};

type BeforeAfterFieldProps = {
    attribute: Attribute;
    dispatchFormAction: React.Dispatch<FormAction>;
    displayableErrors: FormFieldError[];
    valueOrValues: string | string[];
};

export function UserProfileFormFields(props: UserProfileFormFieldsProps) {
    const { onIsFormSubmittableValueChange, BeforeField, AfterField } = props;

    const { kcContext } = useKcContext();

    assert("profile" in kcContext);

    const i18n = useI18n();

    const { advancedMsg } = i18n;

    const {
        formState: { formFieldStates, isFormSubmittable },
        dispatchFormAction
    } = useUserProfileForm({
        kcContext,
        i18n,
        doMakeUserConfirmPassword: DO_MAKE_USER_CONFIRM_PASSWORD
    });

    useEffect(() => {
        onIsFormSubmittableValueChange(isFormSubmittable);
    }, [isFormSubmittable]);

    const groupNameRef = { current: "" };

    return (
        <div className="space-y-6">
            {formFieldStates.map(({ attribute, displayableErrors, valueOrValues }) => {
                return (
                    <Fragment key={attribute.name}>
                        <GroupLabel attribute={attribute} groupNameRef={groupNameRef} />
                        {BeforeField !== undefined && (
                            <BeforeField
                                attribute={attribute}
                                dispatchFormAction={dispatchFormAction}
                                displayableErrors={displayableErrors}
                                valueOrValues={valueOrValues}
                            />
                        )}
                        <div
                            className="space-y-2"
                            style={{
                                display:
                                    attribute.annotations.inputType === "hidden" ? "none" : undefined
                            }}
                        >
                            <div>
                                <label htmlFor={attribute.name} className="block">
                                    <Text as="span" size="2" weight="medium">
                                        {advancedMsg(attribute.displayName ?? "")}
                                        {attribute.required && <span className="text-red-500 ml-1">*</span>}
                                    </Text>
                                </label>
                            </div>
                            <div>
                                {attribute.annotations.inputHelperTextBefore !== undefined && (
                                    <div
                                        className="text-sm text-gray-600 mb-2"
                                        id={`form-help-text-before-${attribute.name}`}
                                        aria-live="polite"
                                    >
                                        <Text size="1" color="gray">
                                            {advancedMsg(attribute.annotations.inputHelperTextBefore)}
                                        </Text>
                                    </div>
                                )}
                                <InputFieldByType
                                    attribute={attribute}
                                    valueOrValues={valueOrValues}
                                    displayableErrors={displayableErrors}
                                    dispatchFormAction={dispatchFormAction}
                                />
                                <FieldErrors
                                    attribute={attribute}
                                    displayableErrors={displayableErrors}
                                    fieldIndex={undefined}
                                />
                                {attribute.annotations.inputHelperTextAfter !== undefined && (
                                    <div
                                        className="text-sm text-gray-600 mt-2"
                                        id={`form-help-text-after-${attribute.name}`}
                                        aria-live="polite"
                                    >
                                        <Text size="1" color="gray">
                                            {advancedMsg(attribute.annotations.inputHelperTextAfter)}
                                        </Text>
                                    </div>
                                )}
                                {AfterField !== undefined && (
                                    <AfterField
                                        attribute={attribute}
                                        dispatchFormAction={dispatchFormAction}
                                        displayableErrors={displayableErrors}
                                        valueOrValues={valueOrValues}
                                    />
                                )}
                                {/* NOTE: Downloading of html5DataAnnotations scripts is done in the useUserProfileForm hook */}
                            </div>
                        </div>
                    </Fragment>
                );
            })}
        </div>
    );
}
