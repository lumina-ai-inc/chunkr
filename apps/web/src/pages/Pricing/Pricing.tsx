import { Flex, Text, Separator, ScrollArea } from "@radix-ui/themes";
import { keyframes } from "@emotion/react";

import "./Pricing.css";

import styled from "@emotion/styled";
import Header from "../../components/Header/Header";
import Calculator from "../../components/PriceCalculator/Calculator";
import PricingCard from "../../components/PricingCard";
import Footer from "../../components/Footer/Footer";

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
    <Flex
      direction="column"
      style={{
        position: "fixed",
        height: "100%",
        width: "100%",
        backgroundColor: "hsl(192, 71%, 4%)",
      }}
    >
      <ScrollArea>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            maxWidth: "1564px",
            margin: "0 auto",
            marginBottom: "232px",
          }}
        >
          <Header />
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              width: "100%",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Flex
              direction="column"
              align="center"
              justify="center"
              px="4"
              className="pricing-container"
            >
              <Text size="9" weight="medium" className="cyan-3">
                Pricing
              </Text>
              <AnimatedSeparator
                size="2"
                style={{
                  backgroundColor: "var(--cyan-12)",
                  width: "100%",
                  marginTop: "24px",
                  height: "3px",
                }}
              />
              <Text
                size="6"
                weight="medium"
                className="cyan-3"
                style={{
                  marginTop: "16px",
                  padding: "0 12px",
                  textAlign: "center",
                  textWrap: "balance",
                }}
              >
                Flexible pricing for every stage of your journey - get started
                for free
              </Text>
              <Text
                size="4"
                weight="medium"
                className="cyan-5"
                mt="16px"
                style={{ textAlign: "center", fontStyle: "italic" }}
              >
                We offer pay-as-you-go, self-hosted, and custom high-volume
                plans.
              </Text>
            </Flex>
            <Flex
              direction="column"
              align="center"
              justify="center"
              px="80px"
              mt="88px"
              gap="88px"
              wrap="wrap"
              width="100%"
            >
              <Calculator />
              <Flex width="100%" justify="between">
                <PricingCard
                  tier="Self-hosted"
                  price="License"
                  text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
                  active={false}
                  enterprise={true}
                  auth={true}
                />
                <PricingCard
                  tier="Managed Instance"
                  price="High Volume"
                  text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
                  active={false}
                  enterprise={true}
                  auth={true}
                />
              </Flex>
            </Flex>
          </div>
        </div>
        <Footer />
      </ScrollArea>
    </Flex>
  );
}
