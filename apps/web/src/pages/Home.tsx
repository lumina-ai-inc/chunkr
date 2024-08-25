import { Text, Button, Flex } from "@radix-ui/themes";

export const Home = () => {
  return (
    <Flex
      direction="column"
      gap="2"
      style={{ backgroundColor: "var(--sky-3)" }}
    >
      <Text size="9" weight="medium" style={{ color: "var(--sky-12)" }}>
        Hello from Radix Themes :)
      </Text>
      <Button>Let's go</Button>
    </Flex>
  );
};
