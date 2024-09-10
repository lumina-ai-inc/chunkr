import { Flex, Text } from "@radix-ui/themes";
import "./Taskcard.css";
export default function TaskCard() {
  return (
    <Flex
      direction="row"
      align="center"
      justify="between"
      className="task-card"
    >
      <Flex direction="row" align="center" justify="between">
        <Text size="2" weight="medium" style={{ color: "var(--cyan-3)" }}>
          08/12/2024 12:21
        </Text>
      </Flex>
    </Flex>
  );
}
