/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 *
 * $ npx keycloakify own --path "login/components/UserProfileFormFields/InputTag.tsx" --revert
 */

import { assert } from "tsafe/assert";
import { FieldErrors } from "./FieldErrors";
import type { InputFieldByTypeProps } from "./InputFieldByType";
import { AddRemoveButtonsMultiValuedAttribute } from "./AddRemoveButtonsMultiValuedAttribute";
import { useI18n } from "../../i18n";

export function InputTag(
  props: InputFieldByTypeProps & { fieldIndex: number | undefined }
) {
  const {
    attribute,
    fieldIndex,
    dispatchFormAction,
    valueOrValues,
    displayableErrors,
  } = props;

  const { advancedMsgStr } = useI18n();

  const hasError =
    displayableErrors.find((error) => error.fieldIndex === fieldIndex) !==
    undefined;

  return (
    <>
      <input
        type={(() => {
          const { inputType } = attribute.annotations;

          if (inputType?.startsWith("html5-")) {
            return inputType.slice(6);
          }

          return inputType ?? "text";
        })()}
        id={attribute.name}
        name={attribute.name}
        value={(() => {
          if (fieldIndex !== undefined) {
            assert(valueOrValues instanceof Array);
            return valueOrValues[fieldIndex];
          }

          assert(typeof valueOrValues === "string");

          return valueOrValues;
        })()}
        className={`flex h-10 w-full px-2 py-3 outline-none rounded-md text-sm border transition-all duration-200 box-border disabled:cursor-not-allowed disabled:opacity-50 border-white/20 bg-white/5 text-white placeholder:text-white/50 focus:border-gray-400 focus:bg-white/8 focus:shadow-[0_0_0_3px_rgba(139,139,139,0.1)] hover:border-white/30 hover:bg-white/7`}
        aria-invalid={hasError}
        disabled={attribute.readOnly}
        autoComplete="off"
        placeholder={
          attribute.annotations.inputTypePlaceholder === undefined
            ? undefined
            : advancedMsgStr(attribute.annotations.inputTypePlaceholder)
        }
        pattern={attribute.annotations.inputTypePattern}
        size={
          attribute.annotations.inputTypeSize === undefined
            ? undefined
            : parseInt(`${attribute.annotations.inputTypeSize}`)
        }
        maxLength={
          attribute.annotations.inputTypeMaxlength === undefined
            ? undefined
            : parseInt(`${attribute.annotations.inputTypeMaxlength}`)
        }
        minLength={
          attribute.annotations.inputTypeMinlength === undefined
            ? undefined
            : parseInt(`${attribute.annotations.inputTypeMinlength}`)
        }
        max={attribute.annotations.inputTypeMax}
        min={attribute.annotations.inputTypeMin}
        step={attribute.annotations.inputTypeStep}
        {...Object.fromEntries(
          Object.entries(attribute.html5DataAnnotations ?? {}).map(
            ([key, value]) => [`data-${key}`, value]
          )
        )}
        onChange={(event) =>
          dispatchFormAction({
            action: "update",
            name: attribute.name,
            valueOrValues: (() => {
              if (fieldIndex !== undefined) {
                assert(valueOrValues instanceof Array);

                return valueOrValues.map((value, i) => {
                  if (i === fieldIndex) {
                    return event.target.value;
                  }

                  return value;
                });
              }

              return event.target.value;
            })(),
          })
        }
        onBlur={() =>
          dispatchFormAction({
            action: "focus lost",
            name: attribute.name,
            fieldIndex: fieldIndex,
          })
        }
      />
      {(() => {
        if (fieldIndex === undefined) {
          return null;
        }

        assert(valueOrValues instanceof Array);

        const values = valueOrValues;

        return (
          <>
            <FieldErrors
              attribute={attribute}
              displayableErrors={displayableErrors}
              fieldIndex={fieldIndex}
            />
            <AddRemoveButtonsMultiValuedAttribute
              attribute={attribute}
              values={values}
              fieldIndex={fieldIndex}
              dispatchFormAction={dispatchFormAction}
            />
          </>
        );
      })()}
    </>
  );
}
