import { Flex, Text, Badge, Separator } from "@radix-ui/themes";
import { keyframes } from "@emotion/react";
import styled from "@emotion/styled";
import "./Pricing.css";

// Add this component at the end of the file
interface PricingCardProps {
  tier: string;
  price: number | string;
  features: string[];
  active: boolean;
  enterprise: boolean;
  auth: boolean;
}

const drawLine = keyframes`
  from {
    width: 0;
  }
  to {
    width: 100%;
  }
`;

const AnimatedSeparator = styled(Separator)`
  animation: ${drawLine} 1s ease-out forwards;
`;

export default function PricingCard({
  tier,
  price,
  features,
  active,
  enterprise,
  auth,
}: PricingCardProps) {
  const isActive = auth && active;

  return (
    <Flex
      direction="column"
      className={isActive ? "card-container-selected" : "card-container"}
    >
      <Text size="6" weight="bold" className="cyan-4">
        {tier}
      </Text>
      <Text
        size="9"
        weight="bold"
        className="cyan-2"
        style={{ marginTop: "32px" }}
      >
        {enterprise ? price : `$${price}`}
      </Text>
      <AnimatedSeparator
        size="2"
        style={{
          backgroundColor: "var(--cyan-12)",
          width: "100%",
          height: "2px",
          marginTop: "24px",
          marginBottom: "24px",
        }}
      />
      <Flex direction="column" gap="20px">
        {features.map((feature, index) => (
          <Text key={index} size="4" weight="medium" className="cyan-2">
            ✓ {feature}
          </Text>
        ))}
      </Flex>
      <Flex mt="32px">
        <Badge
          className={isActive ? "active-badge" : "inactive-badge"}
          style={{ padding: "8px 16px" }}
        >
          <Text size="4" weight="medium">
            {isActive
              ? "Current Plan"
              : enterprise
                ? "Book a Call"
                : auth
                  ? "Upgrade"
                  : "Login"}
          </Text>
        </Badge>
      </Flex>
    </Flex>
  );
}
