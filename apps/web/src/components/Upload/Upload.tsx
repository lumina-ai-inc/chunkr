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
      <Text>Upload</Text>
    </Flex>
  );
}
