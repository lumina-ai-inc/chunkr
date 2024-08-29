import { Flex, Text, Separator } from "@radix-ui/themes";
import { keyframes } from "@emotion/react";
import styled from "@emotion/styled";

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

export default function Pricing() {
  return (
    <Flex direction="column">
      <Flex direction="column" align="center" justify="center" mt="72px">
        <Text size="8" weight="bold" className="cyan-8">
          Pricing
        </Text>
        <AnimatedSeparator
          size="2"
          style={{
            backgroundColor: "var(--cyan-1)",
            width: "100%",
            maxWidth: "50%",
            marginTop: "24px",
          }}
        />
        <Text
          size="4"
          weight="medium"
          className="cyan-1"
          style={{ marginTop: "16px" }}
        >
          From solo devs to enterprise teams - we've got you covered.
        </Text>
      </Flex>
    </Flex>
  );
}
