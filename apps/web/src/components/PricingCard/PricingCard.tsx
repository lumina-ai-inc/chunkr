import { Flex, Text } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";
import "./PricingCard.css";

// Add this component at the end of the file
interface PricingCardProps {
  title: string;
  checkpoints?: string[];
  icon: React.ReactNode;
}

export default function PricingCard({
  title,
  checkpoints,
  icon,
}: PricingCardProps) {
  return (
    <Flex direction="column" width="100%" className="pricing-card-container">
      <Flex
        direction="row"
        gap="4"
        style={{
          padding: "10px 12px",
          borderRadius: "8px",
          backgroundColor: "hsla(180, 100%, 100%, 0.1)",
          width: "fit-content",
        }}
      >
        {icon}
        <Text size="4" weight="bold" style={{ color: "white" }}>
          {title}
        </Text>
      </Flex>

      <Flex direction="column" gap="20px" mt="5">
        {checkpoints && (
          <Flex className="checkpoints-container">
            {checkpoints.map((checkpoint, index) => (
              <Flex
                key={index}
                align="center"
                gap="8px"
                className="checkpoint-item"
              >
                <Text size="3" style={{ color: "hsla(180, 100%, 100%, 0.95)" }}>
                  ✓
                </Text>
                <Text
                  size="3"
                  weight="medium"
                  style={{ color: "hsla(180, 100%, 100%, 0.95)" }}
                >
                  {checkpoint}
                </Text>
              </Flex>
            ))}
          </Flex>
        )}
      </Flex>
      <Flex mt="44px">
        <BetterButton>
          <Text size="3" weight="medium" className="white">
            Book a Call
          </Text>
        </BetterButton>
      </Flex>
    </Flex>
  );
}
