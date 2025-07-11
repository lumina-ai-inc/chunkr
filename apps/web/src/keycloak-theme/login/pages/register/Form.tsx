/**
 * This file has been claimed for ownership from @keycloakify/login-ui version 250004.1.0.
 * To relinquish ownership and restore this file to its original content, run the following command:
 *
 * $ npx keycloakify own --path "login/pages/register/Form.tsx" --revert
 */

import { useState, useEffect, useCallback } from "react";
import { assert } from "tsafe/assert";
import { clsx } from "@keycloakify/login-ui/tools/clsx";
import { useKcContext } from "../../KcContext";
import { useI18n } from "../../i18n";
import { UserProfileFormFields } from "../../components/UserProfileFormFields";
import { TermsAcceptance } from "./TermsAcceptance";
import { Text } from "@radix-ui/themes";

export function Form() {
  const { kcContext } = useKcContext();
  assert(kcContext.pageId === "register.ftl");
  const { msg, msgStr } = useI18n();

  const [isFormSubmittable, setIsFormSubmittable] = useState(false);
  const [areTermsAccepted, setAreTermsAccepted] = useState(false);
  const [isRegisterButtonDisabled, setIsRegisterButtonDisabled] =
    useState(false);
  const [emailValidationError, setEmailValidationError] = useState<
    string | null
  >(null);
  const [currentEmailValue, setCurrentEmailValue] = useState<string>("");

  const validateEmail = useCallback((email: string) => {
    if (!email) {
      setEmailValidationError(null);
      return true;
    }

    // Regex pattern to match blocked email domains
    const blockedEmailPattern =
      /^(?!.*@gmail\.com$|.*@yahoo\.com$|.*@hotmail\.com$|.*@outlook\.com$|.*@aol\.com$|.*@icloud\.com$|.*@live\.com$|.*@msn\.com$|.*@qq\.com$|.*@[A-Za-z]\..*$).*$/;

    if (blockedEmailPattern.test(email)) {
      setEmailValidationError(null);
      return true;
    } else {
      setEmailValidationError("Please use a company email address.");
      return false;
    }
  }, []);

  // Validate email whenever currentEmailValue changes
  useEffect(() => {
    if (currentEmailValue) {
      validateEmail(currentEmailValue);
    }
  }, [currentEmailValue, validateEmail]);

  const handleFormSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    // Get email value from form
    const formData = new FormData(event.currentTarget);
    const email = formData.get("email") as string;

    if (!validateEmail(email)) {
      event.preventDefault();
      return false;
    }

    setIsRegisterButtonDisabled(true);
    return true;
  };

  // Check email on page load if there's a pre-filled value
  useEffect(() => {
    const emailInput = document.querySelector(
      'input[name="email"]'
    ) as HTMLInputElement;
    if (emailInput && emailInput.value) {
      validateEmail(emailInput.value);
    }
  }, [validateEmail]);

  return (
    <div id="kc-form" className="w-full">
      <div id="kc-form-wrapper" className="space-y-6">
        <form
          id="kc-register-form"
          className="space-y-6"
          action={kcContext.url.registrationAction}
          method="post"
          onSubmit={handleFormSubmit}
        >
          <div className="space-y-6">
            <UserProfileFormFields
              onIsFormSubmittableValueChange={setIsFormSubmittable}
              AfterField={({ attribute, valueOrValues }) => {
                if (attribute.name === "email") {
                  const emailValue =
                    typeof valueOrValues === "string"
                      ? valueOrValues
                      : valueOrValues[0] || "";

                  // Update current email value for validation
                  if (emailValue !== currentEmailValue) {
                    setCurrentEmailValue(emailValue);
                  }

                  // Show custom validation error
                  if (emailValidationError) {
                    return (
                      <Text
                        className="text-red-500 text-sm block mt-1"
                        aria-live="polite"
                      >
                        {emailValidationError}
                      </Text>
                    );
                  }
                }
                return null;
              }}
            />
          </div>

          {kcContext.termsAcceptanceRequired && (
            <TermsAcceptance
              areTermsAccepted={areTermsAccepted}
              onAreTermsAcceptedValueChange={setAreTermsAccepted}
            />
          )}

          {kcContext.recaptchaRequired &&
            (kcContext.recaptchaVisible ||
              kcContext.recaptchaAction === undefined) && (
              <div className="space-y-2">
                <div
                  className="g-recaptcha"
                  data-size="compact"
                  data-sitekey={kcContext.recaptchaSiteKey}
                  data-action={kcContext.recaptchaAction}
                ></div>
              </div>
            )}

          <div className="space-y-4">
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

            {kcContext.recaptchaRequired &&
            !kcContext.recaptchaVisible &&
            kcContext.recaptchaAction !== undefined ? (
              <button
                className={clsx(
                  "!w-full rounded-md font-medium button-resting !h-10 g-recaptcha",
                  isRegisterButtonDisabled
                    ? "opacity-50 cursor-not-allowed"
                    : ""
                )}
                data-sitekey={kcContext.recaptchaSiteKey}
                data-callback={() => {
                  (
                    document.getElementById(
                      "kc-register-form"
                    ) as HTMLFormElement
                  ).submit();
                }}
                data-action={kcContext.recaptchaAction}
                type="submit"
                disabled={isRegisterButtonDisabled}
              >
                {msg("doRegister")}
              </button>
            ) : (
              <button
                disabled={
                  !isFormSubmittable ||
                  (kcContext.termsAcceptanceRequired && !areTermsAccepted) ||
                  isRegisterButtonDisabled ||
                  !!emailValidationError
                }
                className={`!w-full rounded-md font-medium button-resting !h-10 ${
                  !isFormSubmittable ||
                  (kcContext.termsAcceptanceRequired && !areTermsAccepted) ||
                  isRegisterButtonDisabled ||
                  !!emailValidationError
                    ? "opacity-50 cursor-not-allowed"
                    : ""
                }`}
                type="submit"
              >
                {msgStr("doRegister")}
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
