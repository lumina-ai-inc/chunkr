import { useEffect, useRef, useState } from "react";
import { Flex, Text } from "@radix-ui/themes";
import { useAuth } from "react-oidc-context";
import { useNavigate } from "react-router-dom";
import "./Home.css";
import Header from "../../components/Header/Header";
// import UploadMain from "../../components/Upload/UploadMain";
import Footer from "../../components/Footer/Footer";
// import heroImageWebp from "../../assets/hero/hero-image.webp";
// import heroImageJpg from "../../assets/hero/hero-image-85-p.jpg";
import Lottie from "lottie-react";
import { LottieRefCurrentProps } from "lottie-react";
import timerAnimation from "../../assets/animations/timer.json";
import fileuploadAnimation from "../../assets/animations/fileupload.json";
import bargraphAnimation from "../../assets/animations/bargraph.json";
import codeAnimation from "../../assets/animations/code.json";
import secureAnimation from "../../assets/animations/secure.json";
import rustAnimation from "../../assets/animations/rust.json";
import segmentationAnimation from "../../assets/animations/segment.json";
import ocrAnimation from "../../assets/animations/ocr.json";
import stackingAnimation from "../../assets/animations/stacking.json";
import extractAnimation from "../../assets/animations/extract.json";
import PricingCard from "../../components/PricingCard/PricingCard";
import { ChunkrClient } from "@chunkr/node";

const Home = () => {
  const auth = useAuth();
  const isAuthenticated = auth.isAuthenticated;
  const navigate = useNavigate();

  const hasAnimatedRef = useRef(false);
  const terminalRef = useRef<HTMLDivElement>(null);

  const lottieRef = useRef<LottieRefCurrentProps>(null);
  const timerLottieRef = useRef<LottieRefCurrentProps>(null);
  const fileuploadLottieRef = useRef<LottieRefCurrentProps>(null);
  const bargraphLottieRef = useRef<LottieRefCurrentProps>(null);
  const codeLottieRef = useRef<LottieRefCurrentProps>(null);
  const secureLottieRef = useRef<LottieRefCurrentProps>(null);
  const rustLottieRef = useRef<LottieRefCurrentProps>(null);
  const segmentationLottieRef = useRef<LottieRefCurrentProps>(null);
  const ocrLottieRef = useRef<LottieRefCurrentProps>(null);
  const stackingLottieRef = useRef<LottieRefCurrentProps>(null);
  const extractLottieRef = useRef<LottieRefCurrentProps>(null);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    if (lottieRef.current) {
      lottieRef.current.pause();
    }
    if (timerLottieRef.current) {
      timerLottieRef.current.pause();
    }
    if (fileuploadLottieRef.current) {
      fileuploadLottieRef.current.pause();
    }
    if (bargraphLottieRef.current) {
      bargraphLottieRef.current.pause();
    }
    if (codeLottieRef.current) {
      codeLottieRef.current.pause();
    }
    if (secureLottieRef.current) {
      secureLottieRef.current.pause();
    }
    if (rustLottieRef.current) {
      rustLottieRef.current.pause();
    }
    if (segmentationLottieRef.current) {
      segmentationLottieRef.current.pause();
    }
    if (ocrLottieRef.current) {
      ocrLottieRef.current.pause();
    }
    if (stackingLottieRef.current) {
      stackingLottieRef.current.pause();
    }
    if (extractLottieRef.current) {
      extractLottieRef.current.pause();
    }
  }, []);

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

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 0);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
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

  const handleLottieHover = (ref: React.RefObject<LottieRefCurrentProps>) => {
    if (ref.current) {
      ref.current.goToAndPlay(0);
    }
  };

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
      <Flex className={`header-container ${isScrolled ? "scrolled" : ""}`}>
        <div
          style={{
            maxWidth: "1424px",
            width: "100%",
            height: "fit-content",
          }}
        >
          <Header px="0px" home={true} />
        </div>
      </Flex>
      <Flex direction="column" align="center" justify="center">
        <div className="hero-main-container">
          <div className="hero-image"></div>
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
        <Flex px="24px" width="100%" align="center" justify="center">
          <div className="hero-content-container">
            <div className="hero-content">
              <div className="placeholder-window">
                <div className="window-header">
                  <div className="window-controls">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <div className="window-title">Document Processing</div>
                </div>
                <div className="window-content">
                  <div className="loading-animation">
                    <div className="progress-bar">
                      <div className="progress"></div>
                    </div>
                    <div className="status-text">Processing document...</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Flex>
        <div className="features-container">
          <div className="features-gradient-background" />
          <Flex
            direction="row"
            style={{
              maxWidth: "1386px",
              height: "100%",
              margin: "0 auto",
              padding: "24px",
              paddingTop: "72px",
              paddingBottom: "72px",
              position: "relative",
              zIndex: 1,
            }}
          >
            <Flex className="features-left-box">
              <Text
                size="9"
                weight="medium"
                className="features-left-box-title"
              >
                A modular pipeline <br></br> for your AI infrastructure
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
              gap="32px"
              style={{ flex: 1, height: "100%" }}
            >
              <Flex direction="column" className="code-container">
                <div className="code-editor-header">
                  <div className="tab-list">
                    <button className="tab-item active">
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M8 13.1L3.6 9.4L5 8.2L7.9 10.6L10.8 8.3L12.2 9.5L8 13.1ZM8 9.7L3.6 6L8 2.3L12.4 6L8 9.7Z"
                          fill="currentColor"
                        />
                      </svg>
                      Node.js
                    </button>
                    <button className="tab-item">
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M7.9 13.5C4.3 13.5 1.4 10.6 1.4 7C1.4 3.4 4.3 0.5 7.9 0.5C11.5 0.5 14.4 3.4 14.4 7C14.4 10.6 11.5 13.5 7.9 13.5ZM7.9 11.5C10.4 11.5 12.4 9.5 12.4 7C12.4 4.5 10.4 2.5 7.9 2.5C5.4 2.5 3.4 4.5 3.4 7C3.4 9.5 5.4 11.5 7.9 11.5Z"
                          fill="currentColor"
                        />
                      </svg>
                      Python
                    </button>
                    <button className="tab-item">
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M8 0.5L14.5 4V12L8 15.5L1.5 12V4L8 0.5Z"
                          fill="currentColor"
                        />
                      </svg>
                      cURL
                    </button>
                  </div>
                  <div className="action-buttons">
                    <button className="action-button">
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M13 4H3V3H13V4ZM13 7H3V6H13V7ZM13 10H3V9H13V10Z"
                          fill="currentColor"
                        />
                      </svg>
                    </button>
                    <button className="action-button">
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path d="M8 12L3 7H13L8 12Z" fill="currentColor" />
                      </svg>
                    </button>
                  </div>
                </div>
                <div className="code-editor-content">
                  <div className="line-numbers">
                    <span>1</span>
                    <span>2</span>
                    <span>3</span>
                    <span>4</span>
                    <span>5</span>
                  </div>
                  <pre className="code-content">
                    <code>{`import { ChunkrClient } from '@chunkr/node';

const client = new ChunkrClient({ 
  apiKey: 'YOUR_API_KEY'
});

// Process a document
const result = await client.process({
  file: './document.pdf',
  model: 'HighQuality',
  chunkLength: 512
});

console.log(result.chunks);`}</code>
                  </pre>
                </div>
              </Flex>

              <Flex
                direction="column"
                className="feature-right-box feature-right-box-segmentation-image "
                onMouseEnter={() => handleLottieHover(segmentationLottieRef)}
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
                  <Lottie
                    lottieRef={segmentationLottieRef}
                    animationData={segmentationAnimation}
                    style={{ width: "112px", height: "112px" }}
                    loop={false}
                    autoplay={false}
                  />
                </Flex>
              </Flex>
              <Flex
                direction="column"
                className="feature-right-box feature-right-box-ocr-image "
                onMouseEnter={() => handleLottieHover(ocrLottieRef)}
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
                  <Lottie
                    lottieRef={ocrLottieRef}
                    animationData={ocrAnimation}
                    style={{ width: "112px", height: "112px" }}
                    loop={false}
                    autoplay={false}
                  />
                </Flex>
              </Flex>
              <Flex
                direction="column"
                className="feature-right-box feature-right-box-outputs-image "
                onMouseEnter={() => handleLottieHover(stackingLottieRef)}
              >
                <Flex className="tag-container">
                  <Text size="1" weight="regular" style={{ color: "#ffffff" }}>
                    Ready-to-go chunks
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
                  <Lottie
                    lottieRef={stackingLottieRef}
                    animationData={stackingAnimation}
                    style={{ width: "112px", height: "112px" }}
                    loop={false}
                    autoplay={false}
                  />
                </Flex>
              </Flex>
              <Flex
                direction="column"
                className="feature-right-box feature-right-box-structuredextraction-image "
                onMouseEnter={() => handleLottieHover(extractLottieRef)}
              >
                <Flex direction="row" gap="8px">
                  <Flex className="tag-container">
                    <Text
                      size="1"
                      weight="regular"
                      style={{ color: "#ffffff" }}
                    >
                      Structured Extraction
                    </Text>
                  </Flex>
                  <Flex className="tag-container">
                    <Text
                      size="1"
                      weight="regular"
                      style={{ color: "#ffffff" }}
                    >
                      New!
                    </Text>
                  </Flex>
                </Flex>

                <Text
                  size="6"
                  mt="16px"
                  weight="medium"
                  className="white"
                  style={{ maxWidth: "250px" }}
                >
                  Custom schemas
                  <span style={{ color: "#ffffff9b" }}> to extract </span>
                  specific values
                </Text>
                <Flex className="feature-right-box-image">
                  <Lottie
                    lottieRef={extractLottieRef}
                    animationData={extractAnimation}
                    style={{ width: "112px", height: "112px" }}
                    loop={false}
                    autoplay={false}
                  />
                </Flex>
              </Flex>
            </Flex>
          </Flex>
        </div>
        <div className="features-container">
          <Flex
            direction="column"
            align="center"
            style={{
              maxWidth: "1424px",
              height: "100%",
              margin: "0 auto",
              padding: "24px",
            }}
          >
            <Flex direction="column" px="24px" align="center" justify="center">
              <Flex className="yc-tag">
                <Text
                  size="2"
                  weight="medium"
                  style={{
                    color: "#ffffff",
                    textShadow: "0 0 10px rgba(255, 255, 255, 0.45)",
                    letterSpacing: "0.02em",
                  }}
                >
                  Built for scale
                </Text>
              </Flex>
              <Text
                size="9"
                weight="medium"
                align="center"
                className="features-bottom-box-title"
              >
                Fast by default<br></br>Powerful by design
              </Text>
              <Flex className="feature-bottom-box-wrapper" direction="column">
                <Flex
                  direction="row"
                  gap="32px"
                  className="features-bottom-box-container"
                  style={{
                    width: "100%",
                    marginTop: "56px",
                  }}
                >
                  <Flex
                    direction="column"
                    className="feature-bottom-box"
                    onMouseEnter={() => handleLottieHover(rustLottieRef)}
                  >
                    <Lottie
                      lottieRef={rustLottieRef}
                      animationData={rustAnimation}
                      loop={false}
                      autoplay={false}
                      style={{
                        width: 32,
                        height: 32,
                      }}
                    />
                    <Text
                      size="6"
                      weight="medium"
                      style={{ color: "white", marginTop: "24px" }}
                    >
                      Connected via Rust
                    </Text>
                    <Text
                      size="3"
                      style={{
                        color: "rgba(255, 255, 255, 0.6)",
                        marginTop: "8px",
                      }}
                    >
                      Process multiple documents simultaneously with efficient
                      resource utilization
                    </Text>
                  </Flex>

                  <Flex
                    direction="column"
                    className="feature-bottom-box"
                    onMouseEnter={() => handleLottieHover(fileuploadLottieRef)}
                  >
                    <Lottie
                      lottieRef={fileuploadLottieRef}
                      animationData={fileuploadAnimation}
                      loop={false}
                      autoplay={false}
                      style={{
                        width: 32,
                        height: 32,
                      }}
                    />
                    <Text
                      size="6"
                      weight="medium"
                      style={{ color: "white", marginTop: "24px" }}
                    >
                      Support for multiple file types
                    </Text>
                    <Text
                      size="3"
                      style={{
                        color: "rgba(255, 255, 255, 0.6)",
                        marginTop: "8px",
                      }}
                    >
                      Minimize memory overhead with efficient data handling and
                      processing
                    </Text>
                  </Flex>

                  <Flex
                    direction="column"
                    className="feature-bottom-box"
                    onMouseEnter={() => handleLottieHover(timerLottieRef)}
                  >
                    <Lottie
                      lottieRef={timerLottieRef}
                      animationData={timerAnimation}
                      loop={false}
                      autoplay={false}
                      style={{
                        width: 32,
                        height: 32,
                      }}
                    />
                    <Text
                      size="6"
                      weight="medium"
                      style={{ color: "white", marginTop: "24px" }}
                    >
                      Optimized last-mile
                    </Text>
                    <Text
                      size="3"
                      style={{
                        color: "rgba(255, 255, 255, 0.6)",
                        marginTop: "8px",
                      }}
                    >
                      Leverage Rust's native performance for lightning-fast
                      document processing
                    </Text>
                  </Flex>
                </Flex>
                <Flex
                  direction="row"
                  gap="32px"
                  className="features-bottom-box-container"
                  style={{
                    width: "100%",
                    marginTop: "48px",
                  }}
                >
                  <Flex
                    direction="column"
                    className="feature-bottom-box"
                    onMouseEnter={() => handleLottieHover(bargraphLottieRef)}
                  >
                    <Lottie
                      lottieRef={bargraphLottieRef}
                      animationData={bargraphAnimation}
                      loop={false}
                      autoplay={false}
                      style={{
                        width: 32,
                        height: 32,
                      }}
                    />
                    <Text
                      size="6"
                      weight="medium"
                      style={{ color: "white", marginTop: "24px" }}
                    >
                      Built-in visibility
                    </Text>
                    <Text
                      size="3"
                      style={{
                        color: "rgba(255, 255, 255, 0.6)",
                        marginTop: "8px",
                      }}
                    >
                      Process multiple documents simultaneously with efficient
                      resource utilization
                    </Text>
                  </Flex>

                  <Flex
                    direction="column"
                    className="feature-bottom-box"
                    onMouseEnter={() => handleLottieHover(codeLottieRef)}
                  >
                    <Lottie
                      lottieRef={codeLottieRef}
                      animationData={codeAnimation}
                      loop={false}
                      autoplay={false}
                      style={{
                        width: 32,
                        height: 32,
                      }}
                    />
                    <Text
                      size="6"
                      weight="medium"
                      style={{ color: "white", marginTop: "24px" }}
                    >
                      Simple API / Cloud-Ready
                    </Text>
                    <Text
                      size="3"
                      style={{
                        color: "rgba(255, 255, 255, 0.6)",
                        marginTop: "8px",
                      }}
                    >
                      Start in seconds with our API, or deploy on your
                      infrastructure for complete control
                    </Text>
                  </Flex>

                  <Flex
                    direction="column"
                    className="feature-bottom-box"
                    onMouseEnter={() => handleLottieHover(secureLottieRef)}
                  >
                    <Lottie
                      lottieRef={secureLottieRef}
                      animationData={secureAnimation}
                      loop={false}
                      autoplay={false}
                      style={{
                        width: 32,
                        height: 32,
                      }}
                    />
                    <Text
                      size="6"
                      weight="medium"
                      style={{ color: "white", marginTop: "24px" }}
                    >
                      Secure in every way
                    </Text>
                    <Text
                      size="3"
                      style={{
                        color: "rgba(255, 255, 255, 0.6)",
                        marginTop: "8px",
                      }}
                    >
                      Leverage Rust's native performance for lightning-fast
                      document processing
                    </Text>
                  </Flex>
                </Flex>
              </Flex>
            </Flex>
          </Flex>
        </div>
        <div className="pricing-section">
          <Flex
            direction="column"
            align="center"
            style={{
              maxWidth: "1424px",
              height: "100%",
              margin: "0 auto",
              padding: "24px",
              position: "relative",
              zIndex: 2,
            }}
          >
            <Flex className="yc-tag">
              <Text
                size="2"
                weight="medium"
                style={{
                  color: "#ffffff",
                  textShadow: "0 0 10px rgba(255, 255, 255, 0.45)",
                  letterSpacing: "0.02em",
                }}
              >
                Plans & Pricing
              </Text>
            </Flex>
            <Text
              size="9"
              weight="medium"
              align="center"
              className="features-bottom-box-title"
            >
              Simple, transparent plans for every stage
            </Text>

            <Flex
              direction="row"
              justify="between"
              gap="48px"
              className="pricing-container"
              style={{
                width: "100%",
                marginTop: "56px",
                padding: "0 24px",
                position: "relative",
                zIndex: 2,
              }}
            >
              <PricingCard
                title="Free"
                credits={100}
                price={0}
                period="month"
                features={[
                  "100 pages per month",
                  "1 request per second",
                  "Community support",
                ]}
                buttonText="Get Started"
              />

              <PricingCard
                title="Hobby"
                credits={10000}
                price={49}
                period="month"
                features={[
                  "10,000 pages per month",
                  "10 requests per second",
                  "Email support",
                ]}
                buttonText="Get Started"
              />

              <PricingCard
                title="Dev"
                credits={150000}
                price={249}
                period="month"
                features={[
                  "150,000 pages per month",
                  "20 requests per second",
                  "Priority support",
                ]}
                buttonText="Get Started"
              />

              <PricingCard
                title="Startup"
                credits={500000}
                price={449}
                period="month"
                features={[
                  "500,000 pages per month",
                  "Enhanced support",
                  "Advanced features",
                ]}
                buttonText="Get Started"
              />
            </Flex>

            <Text
              size="6"
              weight="medium"
              style={{
                color: "white",
                marginTop: "64px",
                marginBottom: "32px",
              }}
            >
              Enterprise Solutions
            </Text>

            <Flex
              direction="row"
              justify="center"
              gap="48px"
              className="pricing-container"
              style={{
                width: "100%",
                padding: "0 24px",
                position: "relative",
                zIndex: 2,
              }}
            >
              <PricingCard
                title="Enterprise"
                credits={0}
                price={0}
                period=""
                features={[
                  "Compute cost + 20%",
                  "One-time setup fee",
                  "99.9% SLA",
                  "Dedicated support team",
                ]}
                buttonText="Contact Sales"
              />

              <PricingCard
                title="Self-hosted"
                credits={0}
                price={0}
                period=""
                features={[
                  "Full source code access",
                  "Custom license terms",
                  "Implementation support",
                  "Security reviews",
                ]}
                buttonText="Contact Sales"
              />
            </Flex>
          </Flex>
        </div>
      </Flex>

      <Footer />
    </Flex>
  );
};

export default Home;
