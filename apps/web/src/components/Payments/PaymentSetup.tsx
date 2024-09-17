import { Elements } from "@stripe/react-stripe-js";
import { loadStripe, StripeElementsOptionsMode } from "@stripe/stripe-js";
import SetupForm from "./SetupForm";

const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_API_KEY);

interface PaymentSetupProps {
  customerSessionClientSecret: string | { customerSessionClientSecret: string };
}

export default function PaymentSetup({
  customerSessionClientSecret,
}: PaymentSetupProps) {
  const clientSecret =
    typeof customerSessionClientSecret === "string"
      ? customerSessionClientSecret
      : customerSessionClientSecret.customerSessionClientSecret;

  const options: StripeElementsOptionsMode = {
    mode: "setup",
    currency: "usd",
    customerSessionClientSecret: clientSecret,
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

  console.log("Stripe Elements options:", options);

  return (
    <Elements stripe={stripePromise} options={options}>
      <SetupForm />
    </Elements>
  );
}
