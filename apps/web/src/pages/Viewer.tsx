import BetterButton from "../components/BetterButton";
import { Text } from "@radix-ui/themes";

export const Viewer = () => {
  return (
    <BetterButton>
      <Text size="9" weight="medium">
        Hello Viewer
      </Text>
    </BetterButton>
  );
};
