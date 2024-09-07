import React, { useState } from "react";
import {
  PaymentElement,
  useStripe,
  useElements,
} from "@stripe/react-stripe-js";
import { Flex, Text } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";

export default function SetupForm() {
  const stripe = useStripe();
  const elements = useElements();
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

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
        <PaymentElement />
        <BetterButton active={!(!stripe || isLoading)} padding="4px 10px">
          <Text size="2" weight="medium">
            {isLoading ? "Processing..." : "Set up payment method"}
          </Text>
        </BetterButton>
        {errorMessage && (
          <Text size="2" style={{ color: "var(--red-9)" }}>
            {errorMessage}
          </Text>
        )}
      </Flex>
    </form>
  );
}
