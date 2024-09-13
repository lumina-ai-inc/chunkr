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
      }}
      className="pulsing-background"
    >
      <ScrollArea>
        <div className="pricing-main-container">
          <div className="pricing-image-container">
            <img
              src="src/assets/pricing-image.png"
              alt="pricing hero"
              className="pricing-hero-image"
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
              }}
            />
            <div className="pricing-gradient-overlay"></div>
          </div>
          <div className="pricing-content-container">
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
                <Text size="9" weight="bold" className="cyan-1">
                  Pricing
                </Text>
                <AnimatedSeparator
                  size="2"
                  style={{
                    backgroundColor: "var(--cyan-4)",
                    width: "100%",
                    marginTop: "24px",
                    height: "3px",
                    paddingLeft: "24px",
                    paddingRight: "24px",
                  }}
                />
                <Text
                  size="6"
                  weight="medium"
                  className="cyan-2"
                  style={{
                    marginTop: "16px",
                    padding: "0 12px",
                    textAlign: "center",
                    textWrap: "balance",
                  }}
                >
                  Flexible pricing for every stage of your journey
                </Text>
                <Flex direction="row" gap="4" py="4px" align="center" mt="1">
                  <Text
                    size="4"
                    weight="medium"
                    className="cyan-4"
                    mt="16px"
                    style={{ textAlign: "center" }}
                  >
                    Metered API
                  </Text>
                  <Separator
                    size="2"
                    orientation="vertical"
                    style={{
                      backgroundColor: "var(--cyan-5)",
                      marginTop: "16px",
                    }}
                  />
                  <Text
                    size="4"
                    weight="medium"
                    className="cyan-4"
                    mt="16px"
                    style={{ textAlign: "center" }}
                  >
                    Managed Instance
                  </Text>
                  <Separator
                    size="2"
                    orientation="vertical"
                    style={{
                      backgroundColor: "var(--cyan-5)",
                      marginTop: "16px",
                    }}
                  />
                  <Text
                    size="4"
                    weight="medium"
                    className="cyan-4"
                    mt="16px"
                    style={{ textAlign: "center" }}
                  >
                    Self-hosted
                  </Text>
                </Flex>
              </Flex>
              <Flex
                direction="row"
                px="80px"
                mt="72px"
                gap="64px"
                width="100%"
                wrap="wrap"
                className="pricing-card-container"
              >
                <Flex
                  direction="column"
                  gap="8"
                  style={{ flex: 1, width: "100%" }}
                >
                  <Calculator />
                </Flex>
                <Flex
                  direction="column"
                  gap="9"
                  style={{ flex: 1, width: "100%" }}
                >
                  <PricingCard
                    tier="Managed Instance"
                    price="High Volume"
                    text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
                    active={false}
                    enterprise={true}
                    auth={true}
                  />
                  <PricingCard
                    tier="Self-hosted"
                    price="License"
                    text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
                    active={false}
                    enterprise={true}
                    auth={true}
                  />
                </Flex>
              </Flex>
              {/* <Flex
                width="100%"
                mt="128px"
                px="80px"
                className="pricing-table-container"
              >
                <PricingTable />
              </Flex> */}
            </div>
          </div>
        </div>
        <Footer />
      </ScrollArea>
    </Flex>
  );
}
