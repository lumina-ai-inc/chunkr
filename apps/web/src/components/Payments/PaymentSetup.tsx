import { Elements } from "@stripe/react-stripe-js";
import { loadStripe, StripeElementsOptionsMode } from "@stripe/stripe-js";
import SetupForm from "./SetupForm";

const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_API_KEY);

interface PaymentSetupProps {
  customerId: string;
  ephemeralKey: string;
  currency: string;
}

export default function PaymentSetup({
  customerId,
  ephemeralKey,
  currency = "usd",
}: PaymentSetupProps) {
  const options: StripeElementsOptionsMode = {
    mode: "setup",
    currency: currency,
    customerOptions: {
      customer: customerId,
      ephemeralKey: ephemeralKey,
    },
    // Add this configuration to display saved payment methods
    paymentMethodCreation: "manual",
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

  console.log(options);

  return (
    <Elements stripe={stripePromise} options={options}>
      <SetupForm />
    </Elements>
  );
}
