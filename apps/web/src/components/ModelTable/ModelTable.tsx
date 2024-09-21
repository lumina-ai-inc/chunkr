import { Flex, Table, Text } from "@radix-ui/themes";
import "./ModelTable.css";

export default function ModelTable() {
  return (
    <Flex direction="column" align="center" justify="center" width="100%">
      <Table.Root variant="surface" mt="24px" style={{ width: "100%" }}>
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeaderCell>
              <Text size="4" weight="bold" className="white table-subtitle">
                FEATURES
              </Text>
            </Table.ColumnHeaderCell>
            <Table.ColumnHeaderCell>
              <Flex direction="row" gap="2" align="center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="18"
                  viewBox="0 0 16 18"
                  fill="none"
                >
                  <rect
                    width="16"
                    height="16"
                    transform="translate(0 1)"
                    fill="white"
                    fillOpacity="0.01"
                  />
                  <path
                    d="M9.31576 0.951108L9.27642 1.04304L9.31576 0.951107C9.04732 0.836241 8.73514 0.919785 8.55994 1.15337L2.15996 9.68667L2.15996 9.68667C2.01603 9.87858 1.99289 10.1353 2.10016 10.3499L2.10017 10.3499C2.20745 10.5645 2.42675 10.7 2.66663 10.7H7.25232L6.30857 16.3625C6.26056 16.6506 6.41569 16.9341 6.68413 17.0489L6.68417 17.049C6.95257 17.1637 7.26475 17.0803 7.43994 16.8467L13.8399 8.31337C13.8399 8.31336 13.8399 8.31335 13.8399 8.31334C13.9839 8.12143 14.0071 7.86468 13.8998 7.65012C13.7925 7.43554 13.5731 7.30001 13.3333 7.30001H8.74757L9.69132 1.63749C9.73933 1.34948 9.5842 1.06597 9.31576 0.951108ZM7.5167 8.34272L7.5167 8.34272C7.63704 8.48476 7.81378 8.56668 7.99994 8.56668H12.0666L7.98499 14.0088L8.62466 10.1708C8.65527 9.98716 8.60352 9.79935 8.48319 9.6573C8.36286 9.51525 8.18611 9.43334 7.99994 9.43334H3.9333L8.01491 3.99119L7.37524 7.82922C7.37524 7.82922 7.37524 7.82922 7.37524 7.82922C7.34462 8.01287 7.39637 8.20066 7.5167 8.34272Z"
                    fill="var(--cyan-6)"
                    stroke="var(--cyan-6)"
                    strokeWidth="0.2"
                  />
                </svg>
                <Text size="4" weight="bold" className="cyan-6 table-subtitle">
                  FAST MODEL
                </Text>
              </Flex>
            </Table.ColumnHeaderCell>
            <Table.ColumnHeaderCell>
              <Flex direction="row" gap="2" align="center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="none"
                >
                  <rect
                    width="16"
                    height="16"
                    fill="white"
                    fillOpacity="0.01"
                  />
                  <path
                    d="M8.00222 0.835547C4.04417 0.835547 0.835547 4.04417 0.835547 8.00221C0.835547 11.9602 4.04417 15.1689 8.00222 15.1689C11.9602 15.1689 15.1689 11.9602 15.1689 8.00221C15.1689 4.04417 11.9602 0.835547 8.00222 0.835547ZM4.80003 7.3667H2.0824C2.37738 4.58645 4.58645 2.37738 7.3667 2.0824V4.80003C7.3667 5.14981 7.65024 5.43337 8.00003 5.43337C8.34981 5.43337 8.63337 5.14981 8.63337 4.80003V2.08193C11.4157 2.37508 13.6269 4.58499 13.9221 7.3667H11.2C10.8503 7.3667 10.5667 7.65024 10.5667 8.00003C10.5667 8.34982 10.8503 8.63337 11.2 8.63337H13.9225C13.6292 11.4171 11.4171 13.6292 8.63337 13.9225V11.2C8.63337 10.8503 8.34982 10.5667 8.00003 10.5667C7.65024 10.5667 7.3667 10.8503 7.3667 11.2V13.9221C4.58499 13.6269 2.37508 11.4157 2.08193 8.63337H4.80003C5.14981 8.63337 5.43337 8.34981 5.43337 8.00003C5.43337 7.65024 5.14981 7.3667 4.80003 7.3667Z"
                    fill="var(--cyan-8)"
                    stroke="var(--cyan-6)"
                    strokeWidth="0.2"
                  />
                </svg>
                <Text size="4" weight="bold" className="cyan-6 table-subtitle">
                  HIGH QUALITY MODEL
                </Text>
              </Flex>
            </Table.ColumnHeaderCell>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          <Table.Row>
            <Table.RowHeaderCell className="sub-header">
              Table extraction
            </Table.RowHeaderCell>
            <Table.Cell>✓</Table.Cell>
            <Table.Cell>✓</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.RowHeaderCell className="sub-header">
              Image processing
            </Table.RowHeaderCell>
            <Table.Cell>Basic</Table.Cell>
            <Table.Cell>Advanced</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.RowHeaderCell className="sub-header">
              Formula OCR
            </Table.RowHeaderCell>
            <Table.Cell>-</Table.Cell>
            <Table.Cell>✓</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.RowHeaderCell className="sub-header">
              Processing speed
            </Table.RowHeaderCell>
            <Table.Cell>Very fast</Table.Cell>
            <Table.Cell>Moderate</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.RowHeaderCell className="sub-header">
              Accuracy
            </Table.RowHeaderCell>
            <Table.Cell>Good</Table.Cell>
            <Table.Cell>Excellent</Table.Cell>
          </Table.Row>
        </Table.Body>
      </Table.Root>
    </Flex>
  );
}
