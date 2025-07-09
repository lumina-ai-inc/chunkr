/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 *
 * $ npx keycloakify own --path "login/components/UserProfileFormFields/TextareaTag.tsx" --revert
 */

import { assert } from "tsafe/assert";
import type { InputFieldByTypeProps } from "./InputFieldByType";

export function TextareaTag(props: InputFieldByTypeProps) {
  const { attribute, dispatchFormAction, displayableErrors, valueOrValues } =
    props;

  const hasError = displayableErrors.length !== 0;

  assert(typeof valueOrValues === "string");

  const value = valueOrValues;

  return (
    <textarea
      id={attribute.name}
      name={attribute.name}
      className={`flex min-h-[80px] w-full px-2 py-3 outline-none rounded-md text-sm ring-offset-background focus-visible:ring-1 focus-visible:ring-gray-300 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50 resize-vertical`}
      aria-invalid={hasError}
      cols={
        attribute.annotations.inputTypeCols === undefined
          ? undefined
          : parseInt(`${attribute.annotations.inputTypeCols}`)
      }
      rows={
        attribute.annotations.inputTypeRows === undefined
          ? undefined
          : parseInt(`${attribute.annotations.inputTypeRows}`)
      }
      maxLength={
        attribute.annotations.inputTypeMaxlength === undefined
          ? undefined
          : parseInt(`${attribute.annotations.inputTypeMaxlength}`)
      }
      value={value}
      onChange={(event) =>
        dispatchFormAction({
          action: "update",
          name: attribute.name,
          valueOrValues: event.target.value,
        })
      }
      onBlur={() =>
        dispatchFormAction({
          action: "focus lost",
          name: attribute.name,
          fieldIndex: undefined,
        })
      }
    />
  );
}
