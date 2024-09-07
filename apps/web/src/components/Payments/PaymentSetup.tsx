import { Elements } from "@stripe/react-stripe-js";
import { loadStripe, StripeElementsOptions } from "@stripe/stripe-js";
import SetupForm from "./SetupForm";
import { Text } from "@radix-ui/themes";

const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_API_KEY);

interface PaymentSetupProps {
  clientSecret: string;
}

export default function PaymentSetup({ clientSecret }: PaymentSetupProps) {
  const options = {
    clientSecret,
    appearance: {
      theme: "night",
      variables: {
        colorPrimary: "#9DDDE7",
        colorBackground: "#061d22",
        colorText: "#9DDDE7",
        colorDanger: "#ff4444",
        fontFamily: "Roboto, sans-serif",
      },
    },
  };

  return (
    <Elements stripe={stripePromise} options={options as StripeElementsOptions}>
      {clientSecret ? (
        <SetupForm />
      ) : (
        <Text size="2" style={{ color: "var(--red-9)" }}>
          Error: Unable to load payment form. Please try again.
        </Text>
      )}
    </Elements>
  );
}
