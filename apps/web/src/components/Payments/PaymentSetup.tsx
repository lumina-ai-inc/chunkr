import { Elements } from "@stripe/react-stripe-js";
import { loadStripe, StripeElementsOptions } from "@stripe/stripe-js";
import SetupForm from "./SetupForm";

const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_API_KEY);

interface PaymentSetupProps {
  customerSessionSecret: string;
  clientSecret: string;
}

export default function PaymentSetup({
  customerSessionSecret,
  clientSecret,
}: PaymentSetupProps) {
  const options: StripeElementsOptions = {
    clientSecret,
    customerSessionClientSecret: customerSessionSecret,
    appearance: {
      theme: "flat",
      variables: {
        colorPrimary: "#FFFFFF",
        colorBackground: "#020809",
        colorText: "#FFFFFF",
        colorDanger: "#ff4444",
        fontFamily: "Roboto, sans-serif",
      },
    },
  };

  return (
    <Elements stripe={stripePromise} options={options}>
      <SetupForm
        clientSecret={clientSecret}
        customerSessionSecret={customerSessionSecret}
      />
    </Elements>
  );
}
