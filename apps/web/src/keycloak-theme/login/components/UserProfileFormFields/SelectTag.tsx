/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 * 
 * $ npx keycloakify own --path "login/components/UserProfileFormFields/SelectTag.tsx" --revert
 */

/* eslint-disable */

import { assert } from "tsafe/assert";
import type { InputFieldByTypeProps } from "./InputFieldByType";
import { InputLabel } from "./InputLabel";

export function SelectTag(props: InputFieldByTypeProps) {
    const { attribute, dispatchFormAction, displayableErrors, valueOrValues } = props;

    const isMultiple = attribute.annotations.inputType === "multiselect";
    const hasError = displayableErrors.length !== 0;

    return (
        <select
            id={attribute.name}
            name={attribute.name}
            className={`flex h-10 w-full px-2 py-3 outline-none rounded-md text-sm ring-offset-background focus-visible:ring-1 focus-visible:ring-gray-100 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50 ${
                hasError ? "bg-red-50 focus-visible:ring-red-500 text-red-500" : ""
            }`}
            aria-invalid={hasError}
            disabled={attribute.readOnly}
            multiple={isMultiple}
            size={
                attribute.annotations.inputTypeSize === undefined
                    ? undefined
                    : parseInt(`${attribute.annotations.inputTypeSize}`)
            }
            value={valueOrValues}
            onChange={event =>
                dispatchFormAction({
                    action: "update",
                    name: attribute.name,
                    valueOrValues: (() => {
                        if (isMultiple) {
                            return Array.from(event.target.selectedOptions).map(option => option.value);
                        }

                        return event.target.value;
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
        >
            {!isMultiple && <option value=""></option>}
            {(() => {
                const options = (() => {
                    walk: {
                        const { inputOptionsFromValidation } = attribute.annotations;

                        if (inputOptionsFromValidation === undefined) {
                            break walk;
                        }

                        assert(typeof inputOptionsFromValidation === "string");

                        const validator = (
                            attribute.validators as Record<string, { options?: string[] }>
                        )[inputOptionsFromValidation];

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

                return options.map((option, i) => (
                    <option key={option} value={option}>
                        {<InputLabel key={i} attribute={attribute} option={option} />}
                    </option>
                ));
            })()}
        </select>
    );
}
