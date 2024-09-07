import { Elements } from "@stripe/react-stripe-js";
import { loadStripe, StripeElementsOptions } from "@stripe/stripe-js";
import SetupForm from "./SetupForm";

// Move this outside of the component to avoid recreating on every render
const stripePromise = loadStripe(
  "pk_test_51OQduZJxCgJtwOshy9VBZBQs0E8fv0GkZyko9G7CMNAyS08l671VTpcKZef15msCMiPatakpyG4gWRYAObbmAKNL00eSXG1jg8"
);

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
      <SetupForm />
    </Elements>
  );
}
