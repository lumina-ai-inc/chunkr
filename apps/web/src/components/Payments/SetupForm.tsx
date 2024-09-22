import React, { useState, useEffect } from "react";
import {
  PaymentElement,
  useStripe,
  useElements,
} from "@stripe/react-stripe-js";
import { Flex, Text } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";

export default function SetupForm({
  clientSecret,
  customerSessionSecret,
}: {
  clientSecret: string;
  customerSessionSecret: string;
}) {
  const stripe = useStripe();
  const elements = useElements();
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasPaymentMethod, setHasPaymentMethod] = useState(false);

  useEffect(() => {
    if (stripe && elements) {
      const elementsOptions = {
        clientSecret,
        customerSessionClientSecret: customerSessionSecret,
        appearance: {
          variables: {
            colorPrimary: "#FFFFFF",
            colorBackground: "#020809",
            colorText: "#FFFFFF",
            colorDanger: "#ff4444",
            fontFamily: "Roboto, sans-serif",
          },
        },
      };

      elements.update(elementsOptions);
    }
  }, [stripe, elements, clientSecret, customerSessionSecret]);

  useEffect(() => {
    if (elements) {
      const paymentElement = elements.getElement("payment");
      if (paymentElement) {
        paymentElement.on("change", (event) => {
          setHasPaymentMethod(!!event.value?.payment_method);
        });
      }
    }
  }, [elements]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!stripe || !elements) {
      return;
    }

    setIsLoading(true);

    const { error } = await stripe.confirmSetup({
      elements,
      confirmParams: {
        return_url: `${window.location.origin}/dashboard`,
      },
    });

    if (error) {
      setErrorMessage(error.message ?? "An unknown error occurred");
    }

    setIsLoading(false);
  };

  return (
    <form onSubmit={handleSubmit}>
      <Flex direction="column" gap="4">
        <PaymentElement id="payment-element" />
        {!hasPaymentMethod && (
          <BetterButton padding="4px 10px">
            <Text
              size="1"
              weight="medium"
              style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
            >
              {isLoading ? "Processing..." : "Add Payment Method"}
            </Text>
          </BetterButton>
        )}
        {errorMessage && (
          <Text size="2" style={{ color: "var(--red-9)" }}>
            {errorMessage}
          </Text>
        )}
      </Flex>
    </form>
  );
}
