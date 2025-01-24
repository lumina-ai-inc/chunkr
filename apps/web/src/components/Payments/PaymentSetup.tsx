import { useEffect, useState } from "react";
import { loadStripe } from "@stripe/stripe-js";
import {
  Elements,
  PaymentElement,
  useStripe,
  useElements,
} from "@stripe/react-stripe-js";
import { Flex, Text, Button } from "@radix-ui/themes";
import useUser from "../../hooks/useUser";
import {
  createSetupIntent,
  createCustomerSession,
} from "../../services/stripeService";

// Initialize Stripe
const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_API_KEY);

const PaymentForm = () => {
  const stripe = useStripe();
  const elements = useElements();
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!stripe || !elements) return;

    setIsLoading(true);
    setError(null);

    try {
      const { error: submitError } = await stripe.confirmSetup({
        elements,
        confirmParams: {
          return_url: `${window.location.origin}/settings/billing`,
        },
      });

      if (submitError) {
        setError(submitError.message ?? "An unexpected error occurred");
      }
    } catch {
      setError("Failed to process payment setup");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <Flex direction="column" gap="4">
        <PaymentElement />
        {error && (
          <Text color="red" size="2">
            {error}
          </Text>
        )}
        <Button
          disabled={!stripe || isLoading}
          style={{ background: "#4CAF50", color: "#FFF" }}
        >
          {isLoading ? "Setting up..." : "Save Payment Method"}
        </Button>
      </Flex>
    </form>
  );
};

export default function PaymentSetup() {
  const { data: user } = useUser();
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  console.log(user);

  useEffect(() => {
    const initializePayment = async () => {
      if (!user) {
        console.log("No user found");
        return;
      }

      try {
        console.log("Starting payment setup with user:", user);

        // Check if we have a user_id
        if (!user.user_id) {
          console.log("No user_id found, creating customer session...");
          const customerSecret = await createCustomerSession(user.user_id); // Note: changed to user.id if that's the correct field
          console.log("Customer session created:", customerSecret);
          if (!customerSecret) {
            setError("Failed to create customer");
            return;
          }
        }

        console.log("Creating setup intent with user_id:", user.user_id);
        const setupSecret = await createSetupIntent(user.user_id);
        console.log("Setup intent created:", setupSecret);
        if (!setupSecret) {
          setError("Failed to create setup intent");
          return;
        }
        setClientSecret(setupSecret);
      } catch (error) {
        console.error("Detailed error:", error);
        setError("Failed to initialize payment setup. Please try again later.");
      }
    };

    // Add logging for the useEffect trigger
    console.log("UseEffect triggered with user:", user);
    if (user) {
      initializePayment();
    }
  }, [user]);

  if (error) {
    return <Text color="red">{error}</Text>;
  }

  if (!clientSecret) {
    return <Text>Loading payment setup...</Text>;
  }

  return (
    <Elements
      stripe={stripePromise}
      options={{
        clientSecret,
        appearance: {
          theme: "night",
          variables: {
            colorPrimary: "#4CAF50",
            colorBackground: "#1A1D1E",
            colorText: "#FFFFFF",
          },
        },
      }}
    >
      <PaymentForm />
    </Elements>
  );
}
