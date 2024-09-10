import { Code, Flex, ScrollArea, Text } from "@radix-ui/themes";
import { useAuth } from "react-oidc-context";
import "./Home.css";
import Header from "../../components/Header/Header";
import UploadMain from "../../components/Upload/UploadMain";
import ModelTable from "../../components/ModelTable/ModelTable";
import Footer from "../../components/Footer/Footer";

const Home = () => {
  const auth = useAuth();
  const isAuthenticated = auth.isAuthenticated;

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
        <div>
          <div className="hero-main-container">
            <div className="hero-image-container">
              <img
                src="src/assets/hero-image.png"
                alt="hero"
                className="hero-image"
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                }}
              />
              <div className="hero-gradient-overlay"></div>
            </div>
            <div className="hero-content-container">
              <Header px="0px" home={true} />
              <Flex className="hero-container">
                <Flex className="text-container" direction="column">
                  <Text
                    size="9"
                    weight="bold"
                    trim="both"
                    className="hero-title"
                  >
                    Open Source Data Ingestion for LLMs & RAG
                  </Text>
                  <Text
                    className="white"
                    size="5"
                    weight="medium"
                    style={{
                      maxWidth: "542px",
                      lineHeight: "32px",
                    }}
                  >
                    Small tagline. Epxlain what demo does and what all can
                    actually be done with API/self-hosting.
                  </Text>
                  <Flex
                    className="signup-container"
                    direction="column"
                    gap="16px"
                  >
                    <Flex
                      direction="row"
                      align="center"
                      className="signup-button"
                    >
                      <Text size="3" weight="bold" className="cyan-12">
                        Get started for free
                      </Text>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                      >
                        <rect
                          width="24"
                          height="24"
                          fill="white"
                          fill-opacity="0.01"
                        />
                        <path
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                          d="M13.0343 5.03431C13.3467 4.72188 13.8532 4.72188 14.1656 5.03431L20.5657 11.4343C20.878 11.7467 20.878 12.2533 20.5657 12.5657L14.1656 18.9658C13.8532 19.2781 13.3467 19.2781 13.0343 18.9658C12.7218 18.6533 12.7218 18.1467 13.0343 17.8342L18.0686 12.8H3.99995C3.55813 12.8 3.19995 12.4418 3.19995 12C3.19995 11.5582 3.55813 11.2 3.99995 11.2H18.0686L13.0343 6.16567C12.7218 5.85326 12.7218 5.34673 13.0343 5.03431Z"
                          fill="#0D3C48"
                        />
                      </svg>
                    </Flex>
                  </Flex>
                  <Flex
                    direction="column"
                    align="start"
                    className="credit-main-container"
                  >
                    <Flex
                      className="credit-container"
                      direction="column"
                      gap="16px"
                    >
                      <Text size="6" weight="medium" className="white">
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
                      </Flex>
                    </Flex>
                  </Flex>
                </Flex>
                <Flex className="module-container" direction="column" gap="4">
                  <UploadMain isAuthenticated={isAuthenticated} />
                </Flex>
              </Flex>
            </div>
          </div>
          <Flex p="80px" mt="40px" className="model-table-container">
            <ModelTable />
          </Flex>
          <Flex
            direction="column"
            align="center"
            justify="center"
            className="curl-block-container"
          >
            <Code
              size="5"
              weight="medium"
              style={{
                padding: "24px",
                color: "var(--cyan-6)",
                boxShadow: "0px 0px 20px 0px rgba(13, 60, 72, 0.3)",
                backgroundColor: "unset",
                borderRadius: "8px",
              }}
            >
              curl -X POST https://api.chunkmydocs.com/v1/chunk \ -H<br></br>
              "Content-Type: application/json" \ -H "Authorization: Bearer
              <br></br>
              YOUR_API_KEY" \ -d '"url": "https://example.com/document.pdf"'
              <br></br>
            </Code>
          </Flex>
          <Flex direction="column" align="center" justify="center" gap="16px">
            <Flex
              direction="row"
              align="center"
              className="signup-button button-bottom"
            >
              <Text size="3" weight="bold" className="cyan-12">
                Get started for free
              </Text>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
              >
                <rect width="24" height="24" fill="white" fill-opacity="0.01" />
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M13.0343 5.03431C13.3467 4.72188 13.8532 4.72188 14.1656 5.03431L20.5657 11.4343C20.878 11.7467 20.878 12.2533 20.5657 12.5657L14.1656 18.9658C13.8532 19.2781 13.3467 19.2781 13.0343 18.9658C12.7218 18.6533 12.7218 18.1467 13.0343 17.8342L18.0686 12.8H3.99995C3.55813 12.8 3.19995 12.4418 3.19995 12C3.19995 11.5582 3.55813 11.2 3.99995 11.2H18.0686L13.0343 6.16567C12.7218 5.85326 12.7218 5.34673 13.0343 5.03431Z"
                  fill="#0D3C48"
                />
              </svg>
            </Flex>
          </Flex>
        </div>

        <Footer />
      </ScrollArea>
    </Flex>
  );
};

export default Home;
