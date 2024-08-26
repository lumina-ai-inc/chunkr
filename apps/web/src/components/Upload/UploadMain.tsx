import { Flex, Text } from "@radix-ui/themes";
import Upload from "./Upload";
import "./UploadMain.css";
import BetterButton from "../BetterButton/BetterButton";

export default function UploadMain() {
  const handleFileUpload = (file: File) => {
    // Handle the uploaded file here
    console.log("Uploaded file:", file);
    // You can now send this file to a server or process it as needed
  };

  return (
    <Flex direction="column" width="100%">
      <Upload onFileUpload={handleFileUpload} />
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
          style={{ borderTopLeftRadius: "4px", borderBottomLeftRadius: "4px" }}
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
          style={{
            borderTopRightRadius: "4px",
            borderBottomRightRadius: "4px",
          }}
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
