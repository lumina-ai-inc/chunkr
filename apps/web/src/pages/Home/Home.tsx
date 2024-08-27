import { Flex, ScrollArea, Text } from "@radix-ui/themes";
import "./Home.css";
import Header from "../../components/Header/Header";
import UploadMain from "../../components/Upload/UploadMain";

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
        <Header py="40px" home={true} />
        <Flex className="hero-container">
          <Flex className="text-container" direction="column">
            <Text
              size="9"
              weight="bold"
              trim="both"
              className="cyan-2 hero-title"
            >
              Source Available Data Ingestion for LLMs & RAG
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
              Small tagline. Epxlain what demo does and what all can actually be
              done with API/self-hosting.
            </Text>
            <Flex
              className="credit-container"
              direction="column"
              gap="16px"
              mt="40px"
            >
              <Text size="6" weight="bold" className="cyan-2">
                Powered by
              </Text>
              <Flex direction="row" gap="32px" align="center">
                <a
                  href="https://www.lumina.sh/"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Flex
                    gap="12px"
                    align="center"
                    justify="center"
                    className="credit-button"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="32"
                      height="32"
                      viewBox="0 0 32 32"
                      fill="none"
                    >
                      <circle cx="16" cy="16" r="16" fill="#D1F0FA" />
                      <path
                        d="M9.57141 8.28572H22.4286M10.8571 8.28572V22C10.8571 22.4547 11.0377 22.8907 11.3592 23.2122C11.6807 23.5337 12.1168 23.7143 12.5714 23.7143H19.4286C19.8832 23.7143 20.3192 23.5337 20.6407 23.2122C20.9622 22.8907 21.1428 22.4547 21.1428 22V8.28572M10.8571 17.7143H21.1428"
                        stroke="#60B3D7"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                      />
                    </svg>
                    <Text size="7" weight="bold" className="cyan-1">
                      Lumina
                    </Text>
                  </Flex>
                </a>
                <Flex direction="row" gap="24px">
                  <a
                    href="https://github.com/skeptrunedev"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Flex height="40px" width="40px" className="credit-dp">
                      <img
                        src="src/assets/Nick.jpeg"
                        alt="Nick from Trieve"
                        style={{ borderRadius: "50%" }}
                      />
                    </Flex>
                  </a>
                  <a
                    href="https://github.com/cdxker"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Flex height="40px" width="40px" className="credit-dp">
                      <img
                        src="src/assets/Denzell.png"
                        alt="Denzell from Trieve"
                        style={{ borderRadius: "50%" }}
                      />
                    </Flex>
                  </a>
                </Flex>
              </Flex>
            </Flex>
          </Flex>
          <Flex className="module-container" direction="column" gap="4">
            <UploadMain />
          </Flex>
        </Flex>
      </ScrollArea>
    </Flex>
  );
};
