import { Flex, Text } from "@radix-ui/themes";
import Upload from "./Upload";
import "./UploadMain.css";
import BetterButton from "../BetterButton/BetterButton";

export default function UploadMain() {
  return (
    <Flex direction="column" width="100%">
      <Upload />
      <Flex
        direction="row"
        height="64px"
        width="100%"
        mt="40px"
        className="toggle-container"
      >
        <Flex
          direction="column"
          height="100%"
          justify="center"
          className="toggle-active"
        >
          <Text size="4" weight="medium">
            Fast
          </Text>
        </Flex>
        <Flex
          direction="column"
          height="100%"
          justify="center"
          className="toggle"
        >
          <Text size="4" weight="medium">
            High Quality
          </Text>
        </Flex>
      </Flex>
      <Flex direction="row" width="100%" mt="32px">
        <BetterButton padding="16px 64px">
          <Text size="4" weight="medium">
            Run
          </Text>
        </BetterButton>
      </Flex>
    </Flex>
  );
}
