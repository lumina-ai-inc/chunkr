import { Flex, Text, Separator, ScrollArea } from "@radix-ui/themes";
import { keyframes } from "@emotion/react";
import { useAuth } from "react-oidc-context";

import "./Pricing.css";

import styled from "@emotion/styled";
import Badge from "../../components/Badge";
import Header from "../../components/Header/Header";

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

interface PricingCardProps {
  tier: string;
  price: string;
  features: string[];
  active: boolean;
  enterprise: boolean;
  auth?: boolean;
}

const PricingCard = ({
  tier,
  price,
  features,
  active,
  enterprise,
  auth,
}: PricingCardProps) => {
  // Determine if the card should be active
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
        style={{ marginTop: "16px" }}
      >
        {enterprise ? price : `$${price}`}
      </Text>
      <Text
        size="3"
        weight="medium"
        className="cyan-8"
        style={{ marginTop: "8px" }}
      >
        {enterprise ? "Contact us" : "per month"}
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
      <Flex direction="column" gap="16px">
        {features.map((feature, index) => (
          <Text key={index} size="3" weight="medium" className="cyan-2">
            ✓ {feature}
          </Text>
        ))}
      </Flex>
      <Flex mt="24px">
        <Badge className={isActive ? "active-badge" : "inactive-badge"}>
          <Text size="2" weight="medium">
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
};

export default function Pricing() {
  const auth = useAuth();
  return (
    <Flex
      direction="column"
      style={{
        position: "fixed",
        height: "100%",
        width: "100%",
        backgroundColor: "hsl(192, 70%, 5%)",
        margin: "0 auto",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <ScrollArea>
        <div style={{ maxWidth: "1564px", margin: "0 auto" }}>
          <Header />
          <Flex
            direction="column"
            align="center"
            justify="center"
            px="4"
            className="pricing-container"
          >
            <Text size="8" weight="bold" className="cyan-4">
              Pricing
            </Text>
            <AnimatedSeparator
              size="2"
              style={{
                backgroundColor: "var(--cyan-12)",
                width: "100%",
                maxWidth: "50%",
                marginTop: "24px",
                height: "3px",
              }}
            />
            <Text
              size="4"
              weight="medium"
              className="cyan-8"
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
            direction="row"
            align="center"
            justify="center"
            mt="64px"
            wrap="wrap"
            px="16px"
            className="pricing-cards-container"
          >
            <PricingCard
              tier="Free"
              price="0"
              features={[
                "100 pages/month",
                "Basic chunking",
                "Standard support",
              ]}
              active={true}
              enterprise={false}
              auth={auth.isAuthenticated}
            />
            <PricingCard
              tier="Dev"
              price="30"
              features={[
                "100 pages/month",
                "Basic chunking",
                "Standard support",
              ]}
              active={false}
              enterprise={false}
              auth={auth.isAuthenticated}
            />
            <PricingCard
              tier="Enterprise"
              price="Custom"
              features={[
                "100 pages/month",
                "Basic chunking",
                "Standard support",
              ]}
              active={false}
              enterprise={true}
              auth={auth.isAuthenticated}
            />
            <PricingCard
              tier="Self-host"
              price="License"
              features={[
                "100 pages/month",
                "Basic chunking",
                "Standard support",
              ]}
              active={false}
              enterprise={true}
              auth={auth.isAuthenticated}
            />
          </Flex>
        </div>
      </ScrollArea>
    </Flex>
  );
}
