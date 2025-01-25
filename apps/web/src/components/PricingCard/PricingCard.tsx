import { Flex, Text } from "@radix-ui/themes";
import "./PricingCard.css";
import { useState } from "react";
import { Stripe } from "@stripe/stripe-js";
import CheckoutOverlay from "../CheckoutOverlay/CheckoutOverlay";
import { useAuth } from "react-oidc-context";
import { getBillingPortalSession } from "../../services/stripeService";

interface PricingCardProps {
  title: string;
  credits: number;
  price: number | "Custom";
  period: string;
  annualPrice?: number;
  features: string[];
  buttonText: string;
  highlighted?: boolean;
  isPopular?: boolean;
  tier: string;
  onCheckout?: (tier: string) => Promise<void>;
  stripePromise?: Promise<Stripe | null>;
  clientSecret?: string;
  currentTier?: string;
  isAuthenticated?: boolean;
  customerId?: string;
}

const PricingCard = ({
  title,
  credits,
  price,
  period,
  annualPrice,
  features,
  buttonText,
  highlighted = false,
  isPopular = false,
  tier,
  onCheckout,
  stripePromise,
  clientSecret,
  currentTier,
  isAuthenticated = false,
  customerId,
}: PricingCardProps) => {
  const [showCheckout, setShowCheckout] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const auth = useAuth();

  const isCurrentPlan = currentTier === tier;

  const getButtonText = () => {
    if (isLoading) return "Loading...";
    if (!isAuthenticated) {
      return tier === "Free" ? "Get Started" : "Subscribe";
    }
    if (isCurrentPlan) return "Current Plan";
    if (currentTier === "Free" && tier !== "Free") return "Upgrade";
    if (tier !== "Free") return "Manage Plan";
    return buttonText;
  };

  const shouldShowCheckout = () => {
    if (!isAuthenticated) return false;
    if (tier === "Free") return false;
    return true;
  };

  const handleClick = async () => {
    if (!isAuthenticated) {
      auth.signinRedirect();
      return;
    }

    if (tier !== "Free") {
      try {
        setIsLoading(true);
        setError(null);

        // If user is on free plan, create checkout session instead of billing portal
        if (currentTier === "Free") {
          if (onCheckout) {
            await onCheckout(tier);
            setShowCheckout(true);
          }
        } else {
          // Otherwise use billing portal for plan management
          const { url } = await getBillingPortalSession(
            auth.user?.access_token || "",
            customerId || ""
          );
          window.location.href = url;
        }
      } catch (err) {
        setError("Failed to process request. Please try again.");
        console.error("Billing error:", err);
      } finally {
        setIsLoading(false);
      }
      return;
    }
  };

  return (
    <Flex
      direction="column"
      className={`pricing-card ${highlighted ? "pricing-card-highlighted" : ""}`}
    >
      {isPopular && (
        <Text className="popular-tag" size="2">
          Most Popular
        </Text>
      )}

      <Text className="pricing-tag" size="6" weight="medium">
        {title}
      </Text>

      <Text size="3" color="gray">
        {credits.toLocaleString()} credits {period}
      </Text>

      <Flex align="baseline" gap="1" className="pricing-amount">
        <Text className="price" size="8" weight="bold">
          {typeof price === "number" ? `$${price}` : price}
        </Text>
        {period && (
          <Text className="period" size="2">
            /{period}
          </Text>
        )}
      </Flex>

      {annualPrice && (
        <Text className="annual-price" size="2">
          ${annualPrice}/yr (Billed annually)
        </Text>
      )}

      <Flex direction="column" gap="4" className="features-list">
        {features.map((feature, index) => (
          <Flex key={index} align="center" gap="2" className="feature-item">
            <Flex
              align="center"
              justify="center"
              className="feature-checkmark-container"
            >
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                <path
                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                  fill="#000000"
                  stroke="#000000"
                />
              </svg>
            </Flex>

            <Text size="2" style={{ color: "#ffffffe4" }}>
              {feature}
            </Text>
          </Flex>
        ))}
      </Flex>

      {!showCheckout ? (
        <button
          className={`pricing-button ${isCurrentPlan ? "current-plan" : ""}`}
          onClick={handleClick}
          disabled={isLoading}
        >
          <Text weight="medium" className="pricing-button-text">
            {getButtonText()}
          </Text>
        </button>
      ) : (
        shouldShowCheckout() &&
        showCheckout &&
        clientSecret &&
        stripePromise && (
          <CheckoutOverlay
            onClose={() => setShowCheckout(false)}
            stripePromise={stripePromise}
            clientSecret={clientSecret}
          />
        )
      )}

      {error && (
        <Text size="2" style={{ color: "red", marginTop: "8px" }}>
          {error}
        </Text>
      )}
    </Flex>
  );
};

export default PricingCard;
