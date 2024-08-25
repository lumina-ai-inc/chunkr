import { Flex, Text } from "@radix-ui/themes";
import Upload from "./Upload";
import "./UploadMain.css";

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
        <Flex direction="column" height="100%" className="toggle-active">
          <Text size="4" weight="bold">
            Fast
          </Text>
        </Flex>
        <Flex direction="column" height="100%" className="toggle">
          <Text size="4" weight="bold">
            High Quality
          </Text>
        </Flex>
      </Flex>
    </Flex>
  );
}
