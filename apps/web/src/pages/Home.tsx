import { Flex, ScrollArea, Text } from "@radix-ui/themes";
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
      <ScrollArea>
        <Header />
        <Flex className="hero-container">
          <Flex className="text-container" direction="column">
            <Text
              size="9"
              weight="bold"
              trim="both"
              className="cyan-2 hero-title"
            >
              Open Source Data Ingestion for LLMs & RAG
            </Text>
            <Text
              className="cyan-2"
              size="5"
              weight="medium"
              style={{
                maxWidth: "542px",
                lineHeight: "32px",
              }}
            >
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam
              euismod, nisi vel consectetur interdum, nisl nunc egestas nunc,
              vitae tincidunt nisl nunc eget nunc.
            </Text>
          </Flex>
          <Flex className="module-container" direction="column" gap="4">
            <UploadMain />
          </Flex>
        </Flex>
      </ScrollArea>
    </Flex>
  );
};
