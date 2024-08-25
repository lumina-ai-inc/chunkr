import { Text } from "@radix-ui/themes";
import BetterButton from "../components/BetterButton";

export const Home = () => {
  return (
    <Text size="9" weight="medium" style={{ color: "var(--cyan-3)" }}>
      <BetterButton>Hello</BetterButton>
    </Text>
  );
};
