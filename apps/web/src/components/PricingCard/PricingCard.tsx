import { Flex, Text } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";
import "./PricingCard.css";

// Add this component at the end of the file
interface PricingCardProps {
  tier: string;
  title: string;
  text: string;
}

export default function PricingCard({ tier, title, text }: PricingCardProps) {
  return (
    <Flex direction="column" width="100%" className="pricing-card-container">
      <Text
        size="4"
        weight="medium"
        style={{ color: "hsla(180, 100%, 100%, 0.92)" }}
        trim="start"
      >
        {tier}
      </Text>
      <Text
        size="8"
        weight="bold"
        style={{ marginTop: "16px", color: "white" }}
      >
        {title}
      </Text>

      <Flex direction="column" gap="20px" mt="4">
        <Text
          size="4"
          weight="regular"
          style={{ color: "hsla(180, 100%, 100%, 0.95)" }}
        >
          {text}
        </Text>
      </Flex>
      <Flex mt="32px">
        <BetterButton>
          <Text size="2" weight="medium" className="white">
            Book a Call
          </Text>
        </BetterButton>
      </Flex>
    </Flex>
  );
}
