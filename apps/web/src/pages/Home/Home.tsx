import { useEffect, useRef } from "react";
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

  const hasAnimatedRef = useRef(false);
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          // Only start typing if it hasn't animated before
          if (entry.isIntersecting && !hasAnimatedRef.current) {
            hasAnimatedRef.current = true;
            startTyping();
          }
        });
      },
      { threshold: 0.5 }
    );

    if (terminalRef.current) {
      observer.observe(terminalRef.current);
    }

    return () => observer.disconnect();
  }, []);

  const curlCommand = `$ <span class="command">curl</span> <span class="flag">-X</span> <span class="flag">POST</span> ${import.meta.env.VITE_API_URL}/api/v1/task <span class="dim">\\</span>
  <span class="flag">-H</span> <span class="string">"Content-Type: multipart/form-data"</span> <span class="dim">\\</span>
  <span class="flag">-H</span> <span class="string">"Authorization: "</span> <span class="dim">\\</span>
  <span class="flag">-F</span> <span class="string">"file=@/path/to/your/file.pdf"</span> <span class="dim">\\</span>
  <span class="flag">-F</span> <span class="string">"model=HighQuality"</span> <span class="dim">\\</span>
  <span class="flag">-F</span> <span class="string">"target_chunk_length=512"</span> <span class="dim">\\</span>
  <span class="flag">-F</span> <span class="string">"ocr_strategy=Auto"</span>`;

  const startTyping = () => {
    const textElement = terminalRef.current?.querySelector(".typed-text");
    if (!textElement) return;

    textElement.innerHTML = "";
    let displayText = "";
    let i = 0;

    const typeChar = () => {
      if (i < curlCommand.length) {
        displayText += curlCommand.charAt(i);
        textElement.innerHTML = displayText + '<span class="cursor">▋</span>';
        i++;
        setTimeout(typeChar, 5);
      } else {
        textElement.innerHTML =
          displayText + '<span class="cursor blink">▋</span>';
      }
    };

    typeChar();
  };

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
      <div>
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
                  weight="medium"
                  size="3"
                  className="hero-description"
                  align="center"
                >
                  API service to turn complex documents into RAG/LLM-ready data
                </Text>

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
            </Flex>
          </div>
        </div>
        <div className="features-container">
          <Flex
            direction="row"
            gap="24px"
            style={{
              maxWidth: "1424px",
              height: "100%",
              margin: "0 auto",
              padding: "24px",
            }}
          >
            <Flex className="features-left-box">
              <Text
                size="9"
                weight="medium"
                className="features-left-box-title"
              >
                A scalable pipeline <br></br> for your AI infrastructure
              </Text>
              <Text
                size="7"
                weight="medium"
                className="features-left-box-subtitle"
              >
                Ingestion use-cases can vary quite a bit. <br></br>
                <br></br>
                <span style={{ color: "#ffffff9b" }}>So we built an</span>{" "}
                end-to-end system{" "}
                <span style={{ color: "#ffffff9b" }}>that can cater to </span>{" "}
                solo-devs, startups and enterprises.
              </Text>

              <div className="feature-left-box-image" ref={terminalRef}>
                <div className="terminal-header">
                  <div className="terminal-title">
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
                        fill-opacity="0.01"
                      />
                      <g className="rotating-globe">
                        <path
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                          d="M7.99898 1.92002C4.64109 1.92002 1.91898 4.64213 1.91898 8.00002C1.91898 11.3579 4.64109 14.08 7.99898 14.08C11.3569 14.08 14.079 11.3579 14.079 8.00002C14.079 4.64213 11.3569 1.92002 7.99898 1.92002ZM0.958984 8.00002C0.958984 4.11193 4.1109 0.960022 7.99898 0.960022C11.887 0.960022 15.039 4.11193 15.039 8.00002C15.039 11.8881 11.887 15.04 7.99898 15.04C4.1109 15.04 0.958984 11.8881 0.958984 8.00002Z"
                          fill="white"
                          fill-opacity="0.9"
                        />
                        <path
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                          d="M14.3996 8.42664H1.59961V7.5733H14.3996V8.42664Z"
                          fill="white"
                          fill-opacity="0.9"
                        />
                        <path
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                          d="M7.57253 14.4V1.59999H8.42586V14.4H7.57253ZM11.066 7.99997C11.066 5.68238 10.2325 3.38962 8.59857 1.87365L9.10641 1.32629C10.9215 3.01032 11.8126 5.51757 11.8126 7.99997C11.8126 10.4824 10.9215 12.9896 9.10641 14.6737L8.59857 14.1263C10.2325 12.6103 11.066 10.3176 11.066 7.99997ZM4.26562 7.99999C4.26562 5.52117 5.12767 3.01522 6.88748 1.33033L7.40384 1.86965C5.82137 3.38477 5.01229 5.67881 5.01229 7.99999C5.0123 10.3212 5.82139 12.6152 7.40387 14.1303L6.88749 14.6696C5.12769 12.9847 4.26564 10.4788 4.26562 7.99999Z"
                          fill="white"
                          fill-opacity="0.9"
                        />
                        <path
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                          d="M8.00028 4.22186C10.3142 4.22186 12.6673 4.64975 14.2623 5.54076C14.4424 5.64132 14.5068 5.86875 14.4062 6.04876C14.3057 6.22876 14.0782 6.29318 13.8982 6.19263C12.4584 5.38831 10.2488 4.96853 8.00028 4.96853C5.75171 4.96853 3.54221 5.38831 2.10234 6.19263C1.92233 6.29318 1.6949 6.22876 1.59434 6.04876C1.4938 5.86875 1.5582 5.64132 1.73821 5.54076C3.3333 4.64975 5.68633 4.22186 8.00028 4.22186ZM8.00028 11.5733C10.3142 11.5733 12.6673 11.1454 14.2623 10.2544C14.4424 10.1538 14.5068 9.9264 14.4062 9.7464C14.3057 9.56639 14.0782 9.50198 13.8982 9.60252C12.4584 10.4068 10.2488 10.8267 8.00028 10.8267C5.75171 10.8267 3.54221 10.4068 2.10234 9.60253C1.92233 9.50198 1.6949 9.56639 1.59434 9.7464C1.4938 9.9264 1.5582 10.1538 1.73821 10.2544C3.3333 11.1454 5.68633 11.5733 8.00028 11.5733Z"
                          fill="white"
                          fill-opacity="0.9"
                        />
                      </g>
                    </svg>
                    chunkr API
                  </div>
                  <div className="terminal-button-row">
                    <div className="terminal-button close"></div>
                    <div className="terminal-button minimize"></div>
                    <div className="terminal-button maximize"></div>
                  </div>
                </div>
                <div className="curl-command">
                  <span className="typed-text"></span>
                </div>
              </div>
            </Flex>

            <Flex
              direction="column"
              gap="24px"
              style={{ flex: 1, height: "100%" }}
            >
              <Flex
                direction="column"
                className="feature-right-box feature-right-box-segmentation-image "
              >
                <Flex className="tag-container">
                  <Text size="1" weight="regular" style={{ color: "#ffffff" }}>
                    Semantic Segmentation
                  </Text>
                </Flex>
                <Text
                  size="6"
                  mt="16px"
                  weight="medium"
                  className="white"
                  style={{ maxWidth: "280px" }}
                >
                  Bounding boxes + tagging{" "}
                  <span style={{ color: "#ffffff9b" }}>for 11 categories</span>
                </Text>
                <Flex className="feature-right-box-image">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="128"
                    height="128"
                    viewBox="0 0 128 128"
                    fill="none"
                  >
                    <rect
                      width="128"
                      height="128"
                      fill="white"
                      fill-opacity="0.01"
                    />
                    <path
                      fill-rule="evenodd"
                      clip-rule="evenodd"
                      d="M2.13281 8.53333C2.13281 4.99871 4.99819 2.13333 8.53281 2.13333H119.466C123.001 2.13333 125.866 4.99871 125.866 8.53333V119.467C125.866 123.001 123.001 125.867 119.466 125.867H8.53281C4.99819 125.867 2.13281 123.001 2.13281 119.467V8.53333ZM14.9328 14.9333V113.067H113.066V14.9333H14.9328Z"
                      fill="url(#paint0_linear_218_66)"
                      fill-opacity="0.85"
                    />
                    <path
                      d="M68.2661 46.9333C68.2661 44.5769 66.3559 42.6667 63.9995 42.6667C61.6431 42.6667 59.7328 44.5769 59.7328 46.9333C59.7328 49.2897 61.6431 51.2 63.9995 51.2C66.3559 51.2 68.2661 49.2897 68.2661 46.9333Z"
                      fill="url(#paint1_linear_218_66)"
                      fill-opacity="0.85"
                    />
                    <path
                      d="M68.2661 29.8667C68.2661 27.5102 66.3559 25.6 63.9995 25.6C61.6431 25.6 59.7328 27.5102 59.7328 29.8667C59.7328 32.2231 61.6431 34.1333 63.9995 34.1333C66.3559 34.1333 68.2661 32.2231 68.2661 29.8667Z"
                      fill="url(#paint2_linear_218_66)"
                      fill-opacity="0.85"
                    />
                    <path
                      d="M68.2661 64C68.2661 61.6436 66.3559 59.7333 63.9995 59.7333C61.6431 59.7333 59.7328 61.6436 59.7328 64C59.7328 66.3564 61.6431 68.2667 63.9995 68.2667C66.3559 68.2667 68.2661 66.3564 68.2661 64Z"
                      fill="url(#paint3_linear_218_66)"
                      fill-opacity="0.85"
                    />
                    <path
                      d="M51.1995 64C51.1995 61.6436 49.2892 59.7333 46.9328 59.7333C44.5764 59.7333 42.6661 61.6436 42.6661 64C42.6661 66.3564 44.5764 68.2667 46.9328 68.2667C49.2892 68.2667 51.1995 66.3564 51.1995 64Z"
                      fill="url(#paint4_linear_218_66)"
                      fill-opacity="0.85"
                    />
                    <path
                      d="M34.1328 64C34.1328 61.6436 32.2226 59.7333 29.8661 59.7333C27.5097 59.7333 25.5995 61.6436 25.5995 64C25.5995 66.3564 27.5097 68.2667 29.8661 68.2667C32.2226 68.2667 34.1328 66.3564 34.1328 64Z"
                      fill="url(#paint5_linear_218_66)"
                      fill-opacity="0.85"
                    />
                    <path
                      d="M85.3328 64C85.3328 61.6436 83.4226 59.7333 81.0661 59.7333C78.7097 59.7333 76.7995 61.6436 76.7995 64C76.7995 66.3564 78.7097 68.2667 81.0661 68.2667C83.4226 68.2667 85.3328 66.3564 85.3328 64Z"
                      fill="url(#paint6_linear_218_66)"
                      fill-opacity="0.85"
                    />
                    <path
                      d="M102.399 64C102.399 61.6436 100.489 59.7333 98.1328 59.7333C95.7764 59.7333 93.8661 61.6436 93.8661 64C93.8661 66.3564 95.7764 68.2667 98.1328 68.2667C100.489 68.2667 102.399 66.3564 102.399 64Z"
                      fill="url(#paint7_linear_218_66)"
                      fill-opacity="0.85"
                    />
                    <path
                      d="M68.2661 81.0667C68.2661 78.7102 66.3559 76.8 63.9995 76.8C61.6431 76.8 59.7328 78.7102 59.7328 81.0667C59.7328 83.4231 61.6431 85.3333 63.9995 85.3333C66.3559 85.3333 68.2661 83.4231 68.2661 81.0667Z"
                      fill="url(#paint8_linear_218_66)"
                      fill-opacity="0.85"
                    />
                    <path
                      d="M68.2661 98.1333C68.2661 95.7769 66.3559 93.8667 63.9995 93.8667C61.6431 93.8667 59.7328 95.7769 59.7328 98.1333C59.7328 100.49 61.6431 102.4 63.9995 102.4C66.3559 102.4 68.2661 100.49 68.2661 98.1333Z"
                      fill="url(#paint9_linear_218_66)"
                      fill-opacity="0.85"
                    />
                    <defs>
                      <linearGradient
                        id="paint0_linear_218_66"
                        x1="64"
                        y1="2"
                        x2="64"
                        y2="120"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.77562"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                      <linearGradient
                        id="paint1_linear_218_66"
                        x1="64"
                        y1="2"
                        x2="64"
                        y2="120"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.77562"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                      <linearGradient
                        id="paint2_linear_218_66"
                        x1="64"
                        y1="2"
                        x2="64"
                        y2="120"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.77562"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                      <linearGradient
                        id="paint3_linear_218_66"
                        x1="64"
                        y1="2"
                        x2="64"
                        y2="120"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.77562"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                      <linearGradient
                        id="paint4_linear_218_66"
                        x1="64"
                        y1="2"
                        x2="64"
                        y2="120"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.77562"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                      <linearGradient
                        id="paint5_linear_218_66"
                        x1="64"
                        y1="2"
                        x2="64"
                        y2="120"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.77562"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                      <linearGradient
                        id="paint6_linear_218_66"
                        x1="64"
                        y1="2"
                        x2="64"
                        y2="120"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.77562"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                      <linearGradient
                        id="paint7_linear_218_66"
                        x1="64"
                        y1="2"
                        x2="64"
                        y2="120"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.77562"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                      <linearGradient
                        id="paint8_linear_218_66"
                        x1="64"
                        y1="2"
                        x2="64"
                        y2="120"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.77562"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                      <linearGradient
                        id="paint9_linear_218_66"
                        x1="64"
                        y1="2"
                        x2="64"
                        y2="120"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.77562"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                    </defs>
                  </svg>
                </Flex>
              </Flex>
              <Flex
                direction="column"
                className="feature-right-box feature-right-box-ocr-image "
              >
                <Flex className="tag-container">
                  <Text size="1" weight="regular" style={{ color: "#ffffff" }}>
                    Intelligent Post-processing
                  </Text>
                </Flex>
                <Text
                  size="6"
                  mt="16px"
                  weight="medium"
                  className="white"
                  style={{ maxWidth: "250px" }}
                >
                  VLMs{" "}
                  <span style={{ color: "#ffffff9b" }}>& specialized </span>
                  OCR models
                </Text>
                <Flex className="feature-right-box-image">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="124"
                    height="124"
                    viewBox="0 0 124 124"
                    fill="none"
                  >
                    <path
                      fill-rule="evenodd"
                      clip-rule="evenodd"
                      d="M0 61.9998C0 27.7583 27.7583 0 61.9999 0C96.2409 0 124 27.7583 124 61.9998C124 96.2409 96.2409 124 61.9999 124C27.7583 124 0 96.2409 0 61.9998ZM9.09555 57.3015C11.338 31.7188 31.7188 11.338 57.3015 9.09555V33.9053C57.3015 36.4896 59.3964 38.5846 61.9807 38.5846C64.565 38.5846 66.66 36.4896 66.66 33.9053V9.09218C92.2607 11.3177 112.66 31.706 114.904 57.3015H90.0559C87.472 57.3015 85.3766 59.3964 85.3766 61.9807C85.3766 64.565 87.472 66.66 90.0559 66.66H114.907C112.681 92.2729 92.2729 112.681 66.66 114.907V90.0559C66.66 87.472 64.565 85.3766 61.9807 85.3766C59.3964 85.3766 57.3015 87.472 57.3015 90.0559V114.904C31.706 112.66 11.3177 92.2607 9.09218 66.66H33.9053C36.4896 66.66 38.5846 64.565 38.5846 61.9807C38.5846 59.3964 36.4896 57.3015 33.9053 57.3015H9.09555Z"
                      fill="url(#paint0_linear_245_740)"
                    />
                    <defs>
                      <linearGradient
                        id="paint0_linear_245_740"
                        x1="62"
                        y1="0"
                        x2="62"
                        y2="124"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.78"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                    </defs>
                  </svg>
                </Flex>
              </Flex>
              <Flex
                direction="column"
                className="feature-right-box feature-right-box-outputs-image "
              >
                <Flex className="tag-container">
                  <Text size="1" weight="regular" style={{ color: "#ffffff" }}>
                    Enriched JSON Output
                  </Text>
                </Flex>
                <Text
                  size="6"
                  mt="16px"
                  weight="medium"
                  className="white"
                  style={{ maxWidth: "250px" }}
                >
                  HTML <span style={{ color: "#ffffff9b" }}> | </span>
                  Markdown <span style={{ color: "#ffffff9b" }}> | </span>
                  OCR <span style={{ color: "#ffffff9b" }}> | </span>
                  Segment Images
                </Text>
                <Flex className="feature-right-box-image">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="128"
                    height="128"
                    viewBox="0 0 128 128"
                    fill="none"
                  >
                    <rect
                      width="128"
                      height="128"
                      fill="white"
                      fill-opacity="0.01"
                    />
                    <path
                      fill-rule="evenodd"
                      clip-rule="evenodd"
                      d="M66.1701 6.99341C64.8312 6.20223 63.1679 6.20223 61.8289 6.99341L14.8956 34.7268C13.5966 35.4943 12.7996 36.8911 12.7996 38.4C12.7996 39.909 13.5966 41.3057 14.8956 42.0733L61.8289 69.8066C63.1679 70.5978 64.8312 70.5978 66.1701 69.8066L113.103 42.0733C114.403 41.3057 115.2 39.909 115.2 38.4C115.2 36.8911 114.403 35.4943 113.103 34.7268L66.1701 6.99341ZM63.9995 61.1775L25.4531 38.4L63.9995 15.6226L102.546 38.4L63.9995 61.1775ZM13.393 63.9628C14.5917 61.9341 17.2081 61.2613 19.2368 62.4601L63.9995 88.9105L108.762 62.4601C110.791 61.2613 113.407 61.9341 114.606 63.9628C115.805 65.9914 115.132 68.6078 113.103 69.8066L66.1701 97.5403C64.8312 98.3313 63.1679 98.3313 61.8289 97.5403L14.8956 69.8066C12.867 68.6078 12.1941 65.9914 13.393 63.9628ZM13.3929 89.5625C14.5917 87.5341 17.208 86.8617 19.2368 88.0598L63.9995 114.511L108.762 88.0598C110.791 86.8617 113.407 87.5341 114.606 89.5625C115.805 91.5917 115.132 94.208 113.103 95.407L66.1701 123.14C64.8312 123.931 63.1679 123.931 61.8289 123.14L14.8956 95.407C12.8669 94.208 12.1941 91.5917 13.3929 89.5625Z"
                      fill="url(#paint0_linear_249_755)"
                    />
                    <defs>
                      <linearGradient
                        id="paint0_linear_249_755"
                        x1="63.9995"
                        y1="6.40002"
                        x2="63.9995"
                        y2="123.734"
                        gradientUnits="userSpaceOnUse"
                      >
                        <stop stop-color="white" />
                        <stop
                          offset="0.78"
                          stop-color="white"
                          stop-opacity="0.5"
                        />
                      </linearGradient>
                    </defs>
                  </svg>
                </Flex>
              </Flex>
              <Flex
                style={{
                  backgroundColor: "hsla(0, 0%, 100%, 0.1)",
                  borderRadius: "12px",
                  height: "25%",
                }}
              >
                {/* Box 4 content */}
              </Flex>
            </Flex>
          </Flex>
        </div>
      </div>

      <Footer />
    </Flex>
  );
};

export default Home;
