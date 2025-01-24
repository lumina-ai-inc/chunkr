import { Flex, Text } from "@radix-ui/themes";
import "./PricingCard.css";

interface PricingCardProps {
  title: string;
  credits: number;
  price: number | "Custom";
  period: string;
  annualPrice?: number;
  features: string[];
  buttonText: string;
  onButtonClick?: () => void;
  highlighted?: boolean;
  isPopular?: boolean;
}

const PricingCard = ({
  title,
  credits,
  price,
  period,
  annualPrice,
  features,
  buttonText,
  onButtonClick,
  highlighted = false,
  isPopular = false,
}: PricingCardProps) => {
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

      <button className="pricing-button" onClick={onButtonClick}>
        {buttonText}
      </button>
    </Flex>
  );
};

export default PricingCard;
