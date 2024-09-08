import { Flex, Table, Text } from "@radix-ui/themes";
import "./PricingTable.css";

export default function PricingTable() {
  return (
    <Flex direction="column" align="center" justify="center" width="100%">
      <Table.Root variant="surface" mt="24px" style={{ width: "100%" }}>
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeaderCell>
              <Text size="9">Plans</Text>
            </Table.ColumnHeaderCell>
            <Table.ColumnHeaderCell>
              <Text size="4" weight="medium" className="cyan-5">
                API
              </Text>
            </Table.ColumnHeaderCell>
            <Table.ColumnHeaderCell>
              <Text size="4" weight="medium" className="cyan-7">
                MANAGED INSTANCE
              </Text>
            </Table.ColumnHeaderCell>
            <Table.ColumnHeaderCell>
              <Text size="4" weight="medium" className="cyan-8">
                SELF-HOSTED
              </Text>
            </Table.ColumnHeaderCell>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          <Table.Row>
            <Table.RowHeaderCell className="sub-header">
              Deployment
            </Table.RowHeaderCell>
            <Table.Cell>Instant</Table.Cell>
            <Table.Cell>Quick setup</Table.Cell>
            <Table.Cell>Custom</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.RowHeaderCell className="sub-header">
              Scalability
            </Table.RowHeaderCell>
            <Table.Cell>Automatic</Table.Cell>
            <Table.Cell>Configurable</Table.Cell>
            <Table.Cell>Manual</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.RowHeaderCell className="sub-header">
              Customization
            </Table.RowHeaderCell>
            <Table.Cell>Limited</Table.Cell>
            <Table.Cell>Moderate</Table.Cell>
            <Table.Cell>Full control</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.RowHeaderCell className="sub-header">
              Maintenance
            </Table.RowHeaderCell>
            <Table.Cell>None</Table.Cell>
            <Table.Cell>Minimal</Table.Cell>
            <Table.Cell>Self-managed</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.RowHeaderCell className="sub-header">
              Data privacy
            </Table.RowHeaderCell>
            <Table.Cell>Standard</Table.Cell>
            <Table.Cell>Enhanced</Table.Cell>
            <Table.Cell>Full control</Table.Cell>
          </Table.Row>
        </Table.Body>
      </Table.Root>
    </Flex>
  );
}
