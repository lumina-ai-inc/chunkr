import { Flex, Text } from "@radix-ui/themes";
import "./PricingCard.css";

interface PricingCardProps {
  title: string;
  description: string;
  price: string;
  priceDetail?: string;
  imageSrc: string;
  buttonText: string;
  onButtonClick?: () => void;
  highlighted?: boolean;
}

const PricingCard = ({
  title,
  description,
  price,
  priceDetail,
  imageSrc,
  buttonText,
  onButtonClick,
  highlighted = false,
}: PricingCardProps) => {
  return (
    <Flex
      direction="column"
      className={`pricing-card ${highlighted ? "pricing-card-highlighted" : ""}`}
    >
      <div
        className="pricing-card-image"
        style={{ backgroundImage: `url(${imageSrc})` }}
      />

      <Flex direction="column" p="4px">
        <Text size="3" className="pricing-tag" style={{ color: "#ffffff" }}>
          {title}
        </Text>

        <Text
          size="8"
          weight="bold"
          className="pricing-amount"
          mt="24px"
          style={{ color: "#ffffff" }}
        >
          {price}
          {priceDetail && (
            <Text
              size="3"
              weight="regular"
              className="pricing-period"
              style={{ color: "#ffffff" }}
            >
              {priceDetail}
            </Text>
          )}
        </Text>

        <Text
          size="3"
          className="pricing-subtitle"
          style={{ color: "#ffffff" }}
        >
          {description}
        </Text>

        <button
          className={`pricing-button ${highlighted ? "pricing-button-highlighted" : ""}`}
          onClick={onButtonClick}
        >
          {buttonText}
        </button>
      </Flex>
    </Flex>
  );
};

export default PricingCard;
