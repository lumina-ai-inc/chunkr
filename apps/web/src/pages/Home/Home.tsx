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
            <div className="hero-image-container fade-in">
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
                  {/* <Flex direction="row" gap="16px">
                    <Text
                      size="3"
                      weight="bold"
                      style={{ color: "hsl(0, 0%, 100%, 0.98)" }}
                    >
                      Backed by Y Combinator
                    </Text>
                  </Flex> */}
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
                    weight="medium"
                    size="5"
                    className="white hero-description"
                  >
                    API service for document layout analysis, OCR and chunking
                    to convert PDFs into RAG/Training-ready Data.
                  </Text>

                  <Flex direction="row" gap="24px" mb="40px">
                    <Flex
                      direction="row"
                      gap="8px"
                      align="center"
                      style={{
                        backgroundColor: "hsla(0, 0%, 100%, 0.1)",
                        borderRadius: "8px",
                        padding: "8px",
                        boxShadow:
                          "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)",
                        backdropFilter: "blur(24px)",
                        border: "1px solid hsla(0, 0%, 100%, 0.1)",
                      }}
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                      >
                        <rect
                          width="16"
                          height="16"
                          fill="white"
                          fillOpacity="0.01"
                        />
                        <path
                          fillRule="evenodd"
                          clipRule="evenodd"
                          d="M4.21317 3.14669V4.79998C4.21317 5.06508 3.99826 5.27998 3.73317 5.27998C3.46807 5.27998 3.25317 5.06508 3.25317 4.79998V2.66671C3.25317 2.61596 3.26105 2.56705 3.27564 2.52114C3.33728 2.3272 3.51882 2.18669 3.73317 2.18669H12.2665C12.4322 2.18669 12.5783 2.27063 12.6646 2.39831C12.7163 2.47492 12.7465 2.56727 12.7465 2.66669V4.79998C12.7465 5.06508 12.5316 5.27998 12.2665 5.27998C12.0014 5.27998 11.7865 5.06508 11.7865 4.79998V3.14669H8.58651V12.8533H9.87115C10.1362 12.8533 10.3511 13.0683 10.3511 13.3333C10.3511 13.5985 10.1362 13.8133 9.87115 13.8133H6.13781C5.87272 13.8133 5.65781 13.5985 5.65781 13.3333C5.65781 13.0683 5.87272 12.8533 6.13781 12.8533H7.41317V3.14669H4.21317Z"
                          fill="white"
                        />
                      </svg>
                      <Text size="2" weight="bold" style={{ color: "white" }}>
                        Text
                      </Text>
                    </Flex>
                    <Flex
                      direction="row"
                      gap="8px"
                      align="center"
                      style={{
                        backgroundColor: "hsla(0, 0%, 100%, 0.1)",
                        borderRadius: "8px",
                        padding: "8px",
                        boxShadow:
                          "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)",
                        backdropFilter: "blur(24px)",
                        border: "1px solid hsla(0, 0%, 100%, 0.1)",
                      }}
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                      >
                        <rect
                          width="16"
                          height="16"
                          fill="white"
                          fillOpacity="0.01"
                        />
                        <path
                          fillRule="evenodd"
                          clipRule="evenodd"
                          d="M8.53332 2.13333H13.3333C13.6278 2.13333 13.8667 2.37212 13.8667 2.66667V5.33333H8.53332V2.13333ZM7.46665 5.33333V2.13333H2.66665C2.3721 2.13333 2.13332 2.37212 2.13332 2.66667V5.33333H7.46665ZM2.13332 6.4V9.6H7.46665V6.4H2.13332ZM8.53332 6.4H13.8667V9.6H8.53332V6.4ZM8.53332 10.6667H13.8667V13.3333C13.8667 13.6278 13.6278 13.8667 13.3333 13.8667H8.53332V10.6667ZM2.13332 13.3333V10.6667H7.46665V13.8667H2.66665C2.3721 13.8667 2.13332 13.6278 2.13332 13.3333ZM1.06665 2.66667C1.06665 1.78301 1.78299 1.06667 2.66665 1.06667H13.3333C14.2169 1.06667 14.9333 1.78301 14.9333 2.66667V13.3333C14.9333 14.217 14.2169 14.9333 13.3333 14.9333H2.66665C1.78299 14.9333 1.06665 14.217 1.06665 13.3333V2.66667Z"
                          fill="white"
                        />
                      </svg>
                      <Text size="2" weight="bold" style={{ color: "white" }}>
                        Tables
                      </Text>
                    </Flex>
                    <Flex
                      direction="row"
                      gap="8px"
                      align="center"
                      style={{
                        backgroundColor: "hsla(0, 0%, 100%, 0.1)",
                        borderRadius: "8px",
                        padding: "8px",
                        boxShadow:
                          "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)",
                        backdropFilter: "blur(24px)",
                        border: "1px solid hsla(0, 0%, 100%, 0.1)",
                      }}
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                      >
                        <rect
                          width="16"
                          height="16"
                          fill="white"
                          fillOpacity="0.01"
                        />
                        <path
                          fillRule="evenodd"
                          clipRule="evenodd"
                          d="M2.66665 1.06667H13.3333C14.2169 1.06667 14.9333 1.78301 14.9333 2.66667V13.3333C14.9333 14.217 14.2169 14.9333 13.3333 14.9333H2.66665C1.78299 14.9333 1.06665 14.217 1.06665 13.3333V2.66667C1.06665 1.78301 1.78299 1.06667 2.66665 1.06667ZM2.66665 2.13333C2.3721 2.13333 2.13332 2.37212 2.13332 2.66667V8.92117L3.92724 7.12725C4.01928 7.03521 4.14475 6.9845 4.27491 6.98674C4.40505 6.98897 4.52871 7.04397 4.61753 7.13913L8.39844 11.1894L11.3939 8.19392C11.5813 8.00647 11.8853 8.00647 12.0727 8.19392L13.8667 9.98784V2.66667C13.8667 2.37212 13.6278 2.13333 13.3333 2.13333H2.66665ZM2.13332 13.3333V10.2788L4.25478 8.15737L8.03316 12.2049L9.53719 13.8667H2.66665C2.3721 13.8667 2.13332 13.6278 2.13332 13.3333ZM13.3333 13.8667H10.832L9.0489 11.8965L11.7333 9.21216L13.8667 11.3455V13.3333C13.8667 13.6278 13.6278 13.8667 13.3333 13.8667ZM7.09249 5.86667C7.09249 5.36547 7.49879 4.95917 7.99998 4.95917C8.50118 4.95917 8.90748 5.36547 8.90748 5.86667C8.90748 6.36786 8.50118 6.77417 7.99998 6.77417C7.49879 6.77417 7.09249 6.36786 7.09249 5.86667ZM7.99998 3.99917C6.96859 3.99917 6.13249 4.83527 6.13249 5.86667C6.13249 6.89806 6.96859 7.73417 7.99998 7.73417C9.03138 7.73417 9.86748 6.89806 9.86748 5.86667C9.86748 4.83527 9.03138 3.99917 7.99998 3.99917Z"
                          fill="white"
                        />
                      </svg>
                      <Text size="2" weight="bold" style={{ color: "white" }}>
                        Images
                      </Text>
                    </Flex>
                  </Flex>
                  <Flex
                    className="signup-container"
                    direction="row"
                    gap="16px"
                    align="center"
                  >
                    <Flex
                      direction="row"
                      align="center"
                      className="signup-button"
                      onClick={handleGetStarted}
                    >
                      {isAuthenticated ? (
                        <Text size="3" weight="bold">
                          Go to dashboard
                        </Text>
                      ) : (
                        <Text size="3" weight="bold">
                          Get started for free
                        </Text>
                      )}
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
                          fillOpacity="0.01"
                        />
                        <path
                          fillRule="evenodd"
                          clipRule="evenodd"
                          d="M13.0343 5.03431C13.3467 4.72188 13.8532 4.72188 14.1656 5.03431L20.5657 11.4343C20.878 11.7467 20.878 12.2533 20.5657 12.5657L14.1656 18.9658C13.8532 19.2781 13.3467 19.2781 13.0343 18.9658C12.7218 18.6533 12.7218 18.1467 13.0343 17.8342L18.0686 12.8H3.99995C3.55813 12.8 3.19995 12.4418 3.19995 12C3.19995 11.5582 3.55813 11.2 3.99995 11.2H18.0686L13.0343 6.16567C12.7218 5.85326 12.7218 5.34673 13.0343 5.03431Z"
                          fill="#000000"
                        />
                      </svg>
                    </Flex>
                    {!isAuthenticated && (
                      <Text
                        size="1"
                        weight="bold"
                        style={{ color: "hsl(0, 0%, 100%, 0.95)" }}
                      >
                        1500 pages in credits
                      </Text>
                    )}
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
