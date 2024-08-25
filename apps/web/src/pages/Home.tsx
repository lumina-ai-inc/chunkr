import { Flex, Text } from "@radix-ui/themes";
import "./Home.css";
import Header from "../components/Header/Header";
import UploadMain from "../components/Upload/UploadMain";

export const Home = () => {
  return (
    <Flex
      direction="column"
      style={{
        position: "fixed",
        height: "100%",
        width: "100%",
      }}
      className="pulsing-background"
    >
      <Header />
      <Flex className="hero-container" direction="row">
        <Flex className="text-container" direction="column" gap="4">
          <Text
            className="cyan-1"
            size="9"
            weight="bold"
            trim="both"
            style={{ maxWidth: "542px", fontSize: "72px", lineHeight: "96px" }}
          >
            Open Source Data Ingestion for LLMs & RAG
          </Text>
        </Flex>
        <Flex className="module-container" direction="column" gap="4">
          <UploadMain />
        </Flex>
      </Flex>
    </Flex>
  );
};
