import { Flex, Text, Separator, ScrollArea } from "@radix-ui/themes";
import { keyframes } from "@emotion/react";

import "./Pricing.css";

import styled from "@emotion/styled";
import Header from "../../components/Header/Header";
import Calculator from "./Calculator";
import PricingCard from "./PricingCard";

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
        backgroundColor: "hsl(192, 70%, 5%)",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <ScrollArea>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            width: "100%",
          }}
        >
          <Header />
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              width: "100%",
              padding: "0 80px",
              marginBottom: "72px",
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
                size="5"
                weight="medium"
                className="cyan-4"
                style={{
                  marginTop: "16px",
                  padding: "0 12px",
                  textAlign: "center",
                  textWrap: "balance",
                }}
              >
                From solo devs to enterprise teams - we've got you covered
              </Text>
            </Flex>
            <Flex
              direction="column"
              align="center"
              justify="center"
              mt="64px"
              gap="64px"
              wrap="wrap"
              width="100%"
            >
              <Calculator />
              <Flex width="100%" justify="center" gap="64px">
                <PricingCard
                  tier="Managed Instance"
                  price="High Volume"
                  features={[
                    "Up to 5 projects",
                    "10GB storage",
                    "Basic support",
                    "API access",
                  ]}
                  active={false}
                  enterprise={true}
                  auth={true}
                />
                <PricingCard
                  tier="Self-hosted"
                  price="License"
                  features={[
                    "Unlimited projects",
                    "50GB storage",
                    "Priority support",
                    "Advanced API access",
                    "Team collaboration",
                  ]}
                  active={false}
                  enterprise={true}
                  auth={true}
                />
              </Flex>
            </Flex>
          </div>
        </div>
      </ScrollArea>
    </Flex>
  );
}
