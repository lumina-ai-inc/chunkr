import { useEffect } from "react";
import { Flex, Text } from "@radix-ui/themes";
import { useAuth } from "react-oidc-context";
import { useNavigate } from "react-router-dom";
import "./Home.css";
import Header from "../../components/Header/Header";
// import UploadMain from "../../components/Upload/UploadMain";
import Footer from "../../components/Footer/Footer";
// import heroImageWebp from "../../assets/hero/hero-image.webp";
// import heroImageJpg from "../../assets/hero/hero-image-85-p.jpg";

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

  useEffect(() => {
    const button = document.querySelector(".signup-button");

    const handleMouseMove = (e: MouseEvent) => {
      const rect = (e.target as HTMLElement)?.getBoundingClientRect();
      const x = e.clientX - rect.left; // x position within the element
      const y = e.clientY - rect.top; // y position within the element

      // Convert to percentage
      const xPercent = (x / rect.width) * 100;
      const yPercent = (y / rect.height) * 100;

      (button as HTMLElement)?.style.setProperty("--mouse-x", `${xPercent}%`);
      (button as HTMLElement)?.style.setProperty("--mouse-y", `${yPercent}%`);
    };

    if (button) {
      button.addEventListener("mousemove", handleMouseMove as EventListener);
    }

    return () => {
      if (button) {
        button.removeEventListener(
          "mousemove",
          handleMouseMove as EventListener
        );
      }
    };
  }, []);

  return (
    <Flex
      direction="column"
      style={{
        position: "relative",
        height: "100%",
        width: "100%",
      }}
      className="background"
    >
      <Flex className="header-container">
        <div
          style={{
            maxWidth: "1424px",
            width: "100%",
            height: "fit-content",
          }}
        >
          {/* <a
            href="https://github.com/lumina-ai-inc/chunkr#readme"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "white", textDecoration: "none" }}
          >
            <Flex direction="row" className="temp-banner" justify="center">
              <Text className="white" size="2">
                
              </Text>
            </Flex>
          </a> */}
          <Header px="0px" home={true} />
        </div>
      </Flex>
      <div style={{ height: "100%" }}>
        <div className="hero-main-container">
          <div className="hero-content-container">
            <Flex className="hero-container" align="center" justify="center">
              <Flex
                className="text-container"
                direction="column"
                align="center"
                justify="center"
              >
                <Flex direction="row" gap="16px" className="yc-tag">
                  <Text
                    size="2"
                    weight="regular"
                    style={{ color: "hsl(0, 0%, 100%, 0.9)" }}
                  >
                    Backed by Y Combinator
                  </Text>
                </Flex>
                <Text
                  weight="bold"
                  className="hero-title"
                  mb="16px"
                  align="center"
                >
                  Open Source Document Ingestion
                </Text>
                <Text
                  weight="regular"
                  size="3"
                  className="hero-description"
                  align="center"
                >
                  API service to turn complex documents into RAG/LLM-ready data
                </Text>

                {/* <Text
                    weight="bold"
                    size="4"
                    mb="12px"
                    className="white signup-byline"
                  >
                    We support PDF, DOC, PPT, and XLS files.
                  </Text> */}

                {/* <Flex direction="row" gap="24px" mb="40px" wrap="wrap">
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
                          d="M12.9561 1.22288C13.1644 1.01459 13.5022 1.01459 13.7104 1.22288L15.8437 3.35621C16.052 3.56449 16.052 3.90217 15.8437 4.11045L11.6383 8.31592C11.5572 8.39699 11.4635 8.46446 11.361 8.51573L7.1718 10.6104C6.96646 10.7131 6.71849 10.6727 6.55616 10.5105C6.39383 10.3481 6.35359 10.1001 6.45626 9.89481L8.55088 5.70557C8.60215 5.60302 8.66962 5.50941 8.75069 5.42834L12.9561 1.22288ZM13.3333 2.35425L9.50494 6.18259L8.39747 8.39751L8.6691 8.66914L10.884 7.56167L14.7124 3.73333L13.3333 2.35425ZM10.6666 2.13333L9.59995 3.2H5.22663C4.76979 3.2 4.45923 3.20041 4.21919 3.22002C3.98536 3.23912 3.86579 3.27376 3.78238 3.31625C3.58167 3.41853 3.4185 3.58171 3.31622 3.78241C3.27373 3.86582 3.2391 3.9854 3.22 4.21922C3.20038 4.45926 3.19997 4.76982 3.19997 5.22667V11.84C3.19997 12.2969 3.20038 12.6074 3.22 12.8475C3.2391 13.0813 3.27373 13.2009 3.31622 13.2843C3.4185 13.485 3.58167 13.6481 3.78238 13.7504C3.86579 13.793 3.98536 13.8275 4.21919 13.8466C4.45923 13.8662 4.76979 13.8667 5.22663 13.8667H11.84C12.2968 13.8667 12.6073 13.8662 12.8474 13.8466C13.0812 13.8275 13.2008 13.793 13.2842 13.7504C13.485 13.6481 13.6481 13.485 13.7504 13.2843C13.7929 13.2009 13.8275 13.0813 13.8466 12.8475C13.8662 12.6074 13.8666 12.2969 13.8666 11.84V7.46664L14.9333 6.39998V11.84V11.8621C14.9333 12.2913 14.9333 12.6457 14.9097 12.9343C14.8852 13.2341 14.8327 13.5097 14.7008 13.7685C14.4963 14.1699 14.1699 14.4963 13.7685 14.7008C13.5096 14.8327 13.2341 14.8852 12.9343 14.9098C12.6456 14.9333 12.2913 14.9333 11.862 14.9333H11.84H5.22663H5.2046C4.77529 14.9333 4.42096 14.9333 4.13233 14.9098C3.83254 14.8852 3.55697 14.8327 3.29812 14.7008C2.89671 14.4963 2.57035 14.1699 2.36582 13.7685C2.23393 13.5097 2.18137 13.2341 2.15687 12.9343C2.13329 12.6457 2.13329 12.2913 2.1333 11.8621V11.84V5.22667V5.20464C2.13329 4.77535 2.13329 4.42099 2.15687 4.13236C2.18137 3.83257 2.23393 3.557 2.36582 3.29815C2.57035 2.89673 2.89671 2.57038 3.29812 2.36585C3.55697 2.23396 3.83254 2.1814 4.13233 2.15691C4.42096 2.13332 4.77527 2.13332 5.20458 2.13333H5.22663H10.6666Z"
                          fill="white"
                        />
                      </svg>
                      <Text size="2" weight="bold" style={{ color: "white" }}>
                        Handwriting
                      </Text>
                    </Flex>
                  </Flex> */}
                <Flex
                  className="signup-container"
                  direction="row"
                  gap="16px"
                  align="center"
                >
                  <button className="signup-button" onClick={handleGetStarted}>
                    {isAuthenticated ? (
                      <Text size="5" weight="bold">
                        Go to dashboard
                      </Text>
                    ) : (
                      <Text size="5" weight="bold">
                        Get started for free
                      </Text>
                    )}
                  </button>
                </Flex>
              </Flex>
              {/* <Flex className="module-container" direction="column" gap="4">
                  <UploadMain isAuthenticated={isAuthenticated} />
                </Flex> */}
            </Flex>
          </div>
        </div>
      </div>

      <Footer />
    </Flex>
  );
};

export default Home;
