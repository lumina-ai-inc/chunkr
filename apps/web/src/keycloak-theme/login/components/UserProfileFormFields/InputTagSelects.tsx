/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/components/UserProfileFormFields/InputTagSelects.tsx" --revert
 */

import { assert } from "tsafe/assert";
import type { InputFieldByTypeProps } from "./InputFieldByType";
import { InputLabel } from "./InputLabel";
import { Text } from "@radix-ui/themes";

export function InputTagSelects(props: InputFieldByTypeProps) {
    const { attribute, dispatchFormAction, valueOrValues } = props;

    const { inputType } = (() => {
        const { inputType } = attribute.annotations;

        assert(inputType === "select-radiobuttons" || inputType === "multiselect-checkboxes");

        switch (inputType) {
            case "select-radiobuttons":
                return {
                    inputType: "radio"
                };
            case "multiselect-checkboxes":
                return {
                    inputType: "checkbox"
                };
        }
    })();

    const options = (() => {
        walk: {
            const { inputOptionsFromValidation } = attribute.annotations;

            if (inputOptionsFromValidation === undefined) {
                break walk;
            }

            const validator = (attribute.validators as Record<string, { options?: string[] }>)[
                inputOptionsFromValidation
            ];

            if (validator === undefined) {
                break walk;
            }

            if (validator.options === undefined) {
                break walk;
            }

            return validator.options;
        }

        return attribute.validators.options?.options ?? [];
    })();

    const hasError = props.displayableErrors.length !== 0;

    return (
        <div className="space-y-3">
            {options.map(option => (
                <div key={option} className="flex items-center space-x-2">
                    <input
                        type={inputType}
                        id={`${attribute.name}-${option}`}
                        name={attribute.name}
                        value={option}
                        className={`rounded-sm focus:ring-0 focus:outline-none ${
                            hasError ? "border-red-500" : ""
                        } ${attribute.readOnly ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                        aria-invalid={hasError}
                        disabled={attribute.readOnly}
                        checked={
                            valueOrValues instanceof Array
                                ? valueOrValues.includes(option)
                                : valueOrValues === option
                        }
                        onChange={event =>
                            dispatchFormAction({
                                action: "update",
                                name: attribute.name,
                                valueOrValues: (() => {
                                    const isChecked = event.target.checked;

                                    if (valueOrValues instanceof Array) {
                                        const newValues = [...valueOrValues];

                                        if (isChecked) {
                                            newValues.push(option);
                                        } else {
                                            newValues.splice(newValues.indexOf(option), 1);
                                        }

                                        return newValues;
                                    }

                                    return event.target.checked ? option : "";
                                })()
                            })
                        }
                        onBlur={() =>
                            dispatchFormAction({
                                action: "focus lost",
                                name: attribute.name,
                                fieldIndex: undefined
                            })
                        }
                    />
                    <label
                        htmlFor={`${attribute.name}-${option}`}
                        className={`cursor-pointer ${attribute.readOnly ? "opacity-50 cursor-not-allowed" : ""}`}
                    >
                        <Text as="span" size="2">
                            <InputLabel attribute={attribute} option={option} />
                        </Text>
                    </label>
                </div>
            ))}
        </div>
    );
}
