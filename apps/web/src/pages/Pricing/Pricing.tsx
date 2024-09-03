import { Flex, Text, Separator, ScrollArea, Table } from "@radix-ui/themes";
import { keyframes } from "@emotion/react";

import "./Pricing.css";

import styled from "@emotion/styled";
import Header from "../../components/Header/Header";
import Calculator from "./Calculator";

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

const PricingTable = () => {
  return (
    <Flex
      direction="column"
      width="100%"
      p="8"
      style={{
        border: "3px solid var(--cyan-5)",
        borderRadius: "8px",
        boxShadow: "0px 0px 20px 0px rgba(0, 0, 0, 0.2)",
      }}
    >
      <Text weight="bold" size="6" className="cyan-2" mb="4">
        High Volume Plans
      </Text>
      <Table.Root>
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeaderCell>Plan</Table.ColumnHeaderCell>
            <Table.ColumnHeaderCell>Price</Table.ColumnHeaderCell>
            <Table.ColumnHeaderCell>Description</Table.ColumnHeaderCell>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          <Table.Row>
            <Table.Cell>Basic</Table.Cell>
            <Table.Cell>$10/month</Table.Cell>
            <Table.Cell>100 pages/month, basic chunking</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.Cell>Pro</Table.Cell>
            <Table.Cell>$50/month</Table.Cell>
            <Table.Cell>
              500 pages/month, advanced chunking, priority support
            </Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.Cell>Enterprise</Table.Cell>
            <Table.Cell>Custom</Table.Cell>
            <Table.Cell>
              Unlimited pages, custom features, dedicated support
            </Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.Cell>Self-host</Table.Cell>
            <Table.Cell>License</Table.Cell>
            <Table.Cell>
              On-premise deployment, customizable features
            </Table.Cell>
          </Table.Row>
        </Table.Body>
      </Table.Root>
    </Flex>
  );
};

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
              <Text size="9" weight="medium" className="cyan-4">
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
                size="5"
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
              direction="column"
              align="center"
              justify="center"
              mt="64px"
              gap="64px"
              wrap="wrap"
              width="100%"
            >
              <Calculator />
              <PricingTable />
            </Flex>
          </div>
        </div>
      </ScrollArea>
    </Flex>
  );
}
