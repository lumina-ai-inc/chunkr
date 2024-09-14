import { Flex, ScrollArea, Text } from "@radix-ui/themes";
import { useAuth } from "react-oidc-context";
import { useNavigate } from "react-router-dom";
import "./Home.css";
import Header from "../../components/Header/Header";
import UploadMain from "../../components/Upload/UploadMain";
import Footer from "../../components/Footer/Footer";
import heroImageWebp from "../../assets/hero/hero-image.webp";
import heroImageJpg from "../../assets/hero/hero-image-85-p.jpg";

const Home = () => {
  const auth = useAuth();
  const isAuthenticated = auth.isAuthenticated;
  const navigate = useNavigate();

  const handleGetStarted = () => {
    if (auth.isAuthenticated) {
      navigate("/dashboard");
    } else {
      auth.signinRedirect();
    }
  };

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
      <ScrollArea type="scroll">
        <div>
          <div className="hero-main-container">
            <div className="hero-image-container">
              <picture>
                <source srcSet={heroImageWebp} type="image/webp" />
                <img
                  src={heroImageJpg}
                  alt="hero"
                  className="hero-image"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                  }}
                />
              </picture>
              <div className="hero-gradient-overlay"></div>
            </div>
            <div className="hero-content-container">
              <Flex className="header-container">
                <div style={{ maxWidth: "1312px", width: "100%" }}>
                  <Header px="0px" home={true} />
                </div>
              </Flex>
              <Flex className="hero-container">
                <Flex className="text-container" direction="column">
                  <Text
                    size="9"
                    weight="bold"
                    trim="both"
                    className="hero-title"
                    mb="24px"
                  >
                    Open Source Data Ingestion
                  </Text>
                  <Text
                    className="white"
                    size="5"
                    weight="medium"
                    mb="24px"
                    style={{
                      maxWidth: "542px",
                      lineHeight: "32px",
                    }}
                  >
                    API service for document layout analysis and chunking to
                    convert PDFs into RAG/LLM-ready data.
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
                      onClick={handleGetStarted}
                    >
                      <Text size="3" weight="bold">
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
                          fill="#000000"
                        />
                      </svg>
                    </Flex>
                  </Flex>
                </Flex>
                <Flex className="module-container" direction="column" gap="4">
                  <UploadMain isAuthenticated={isAuthenticated} />
                </Flex>
              </Flex>
            </div>
          </div>
        </div>

        <Footer />
      </ScrollArea>
    </Flex>
  );
};

export default Home;

// curl -X POST https://api.chunkmydocs.com/api/task \ <br></br>
// -H "Content-Type: application/json" \ <br></br>
// -H "Authorization:{"{your_api_key}"}" \ <br></br>
// -F "file=@/path/to/your/file.pdf" \ <br></br>
// -F "model=Fast" \ <br></br>
// -F "target_chunk_length=512"
