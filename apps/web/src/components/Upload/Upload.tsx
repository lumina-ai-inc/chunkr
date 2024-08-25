import { Flex, Text } from "@radix-ui/themes";

export default function Upload() {
  return (
    <Flex
      direction="row"
      width="100%"
      height="302px"
      align="center"
      justify="center"
      style={{
        backgroundColor: "#061D22",
        borderRadius: "8px",
        border: "4px solid var(--cyan-5)",
        boxShadow: "0px 0px 16px 0px rgba(12, 12, 12, 0.25)",
      }}
    >
      <Flex
        direction="column"
        py="10px"
        px="12px"
        style={{ border: "1px dashed var(--Colors-Cyan-6, #9DDDE7)" }}
      >
        <Text size="6" weight="bold" className="cyan-1">
          Upload Document
        </Text>
      </Flex>
    </Flex>
  );
}
