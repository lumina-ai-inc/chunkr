// React Imports
import { useEffect, useRef, useState } from "react";

// OIDC Imports
import { useAuth } from "react-oidc-context";
import { useNavigate } from "react-router-dom";

// Radix Imports
import { Flex, Text } from "@radix-ui/themes";

// CSS Imports
import "./Home.css";

// Component Imports
// import PricingCard from "../../components/PricingCard/PricingCard";
// import CodeBlock from "../../components/CodeBlock/CodeBlock";
// import BetterButton from "../../components/BetterButton/BetterButton";
// import {
//   curlExample,
//   nodeExample,
//   pythonExample,
//   rustExample,
// } from "../../components/CodeBlock/exampleScripts";
// import Dropdown from "../../components/Dropdown/Dropdown";
import MomentumScroll from "../../components/MomentumScroll/MomentumScroll";
import Header from "../../components/Header/Header";
// import UploadMain from "../../components/Upload/UploadMain";
// import Footer from "../../components/Footer/Footer";

// Animation Imports
// import Lottie from "lottie-react";
import { LottieRefCurrentProps } from "lottie-react";
// import timerAnimation from "../../assets/animations/timer.json";
// import fileuploadAnimation from "../../assets/animations/fileupload.json";
// import bargraphAnimation from "../../assets/animations/bargraph.json";
// import codeAnimation from "../../assets/animations/code.json";
// import secureAnimation from "../../assets/animations/secure.json";
// import rustAnimation from "../../assets/animations/rust.json";
// import ocrAnimation from "../../assets/animations/ocr.json";
// import chunkingAnimation from "../../assets/animations/chunking.json";
// import vlmAnimation from "../../assets/animations/vlm.json";
// import layoutAnimation from "../../assets/animations/layout.json";

const Home = () => {
  const auth = useAuth();
  const isAuthenticated = auth.isAuthenticated;
  const navigate = useNavigate();

  // const terminalRef = useRef<HTMLDivElement>(null);

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
  const parsingLottieRef = useRef<LottieRefCurrentProps>(null);
  const notesLottieRef = useRef<LottieRefCurrentProps>(null);
  const curlLottieRef = useRef<LottieRefCurrentProps>(null);
  const vlmLottieRef = useRef<LottieRefCurrentProps>(null);
  const chunkingLottieRef = useRef<LottieRefCurrentProps>(null);
  const layoutLottieRef = useRef<LottieRefCurrentProps>(null);
  const [isScrolled, setIsScrolled] = useState(false);
  // const [selectedScript, setSelectedScript] = useState("curl");
  // const [segmentationStrategy, setSegmentationStrategy] = useState<
  //   "LayoutAnalysis" | "Page"
  // >("LayoutAnalysis");
  // const [selectedModel, setSelectedModel] = useState<
  //   "ChunkrVGT" | "AzureDocIntelligence"
  // >("ChunkrVGT");
  // const [ocrMode, setOcrMode] = useState<"ALL" | "OFF">("ALL");
  // const [vlmMode, setVlmMode] = useState<"SegmentType" | "ALL" | "OFF">(
  //   "SegmentType"
  // );
  // const [schemaMode, setSchemaMode] = useState<"JSON" | "OFF">("JSON");
  // const [selectedSegmentTypes, setSelectedSegmentTypes] = useState<string[]>([
  //   "Formula",
  //   "Picture",
  //   "Table",
  // ]);

  // const allSegmentTypes = [
  //   "Caption",
  //   "Formula",
  //   "Footnote",
  //   "List item",
  //   "Page footer",
  //   "Page header",
  //   "Picture",
  //   "Section header",
  //   "Table",
  //   "Text",
  //   "Title",
  // ];

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
    if (curlLottieRef.current) {
      curlLottieRef.current.pause();
    }
    if (parsingLottieRef.current) {
      parsingLottieRef.current.pause();
    }
    if (notesLottieRef.current) {
      notesLottieRef.current.pause();
    }
    if (vlmLottieRef.current) {
      vlmLottieRef.current.pause();
    }
    if (layoutLottieRef.current) {
      layoutLottieRef.current.pause();
    }
    if (chunkingLottieRef.current) {
      chunkingLottieRef.current.pause();
    }
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 0);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // const handleScriptSwitch = (script: string) => {
  //   setSelectedScript(script);
  // };

  // const scripts = {
  //   curl: curlExample,
  //   node: nodeExample,
  //   python: pythonExample,
  //   rust: rustExample,
  // };

  // const languageMap = {
  //   curl: "bash",
  //   node: "javascript",
  //   python: "python",
  //   rust: "rust",
  // };

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

  // const handleLottieHover = (ref: React.RefObject<LottieRefCurrentProps>) => {
  //   if (ref.current) {
  //     ref.current.goToAndPlay(0);
  //   }
  // };

  // const FeatureBox = ({
  //   icon,
  //   title,
  //   description,
  //   onMouseEnter,
  // }: {
  //   icon: React.ReactNode;
  //   title: string;
  //   description: string;
  //   onMouseEnter?: () => void;
  // }) => {
  //   return (
  //     <Flex
  //       direction="column"
  //       className="feature-bottom-box"
  //       onMouseEnter={onMouseEnter}
  //     >
  //       <Flex
  //         align="center"
  //         justify="center"
  //         className="feature-bottom-box-icon"
  //       >
  //         {icon}
  //       </Flex>

  //       <Text
  //         size="6"
  //         weight="bold"
  //         style={{
  //           color: "white",
  //           marginTop: "32px",
  //           transition: "margin 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  //         }}
  //         className="feature-box-title"
  //       >
  //         {title}
  //       </Text>
  //       <Text
  //         size="3"
  //         weight="medium"
  //         style={{
  //           color: "rgba(255, 255, 255, 0.7)",
  //           marginTop: "12px",
  //           transition: "margin 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  //         }}
  //         className="feature-box-description"
  //       >
  //         {description}
  //       </Text>
  //     </Flex>
  //   );
  // };

  return (
    <>
      <Flex className={`header-container ${isScrolled ? "scrolled" : ""}`}>
        <div
          style={{
            maxWidth: "1386px",
            width: "100%",
            height: "fit-content",
          }}
        >
          <Header auth={auth} />
        </div>
      </Flex>
      <MomentumScroll>
        <Flex
          direction="column"
          style={{
            position: "relative",
            height: "100%",
            width: "100%",
          }}
          className="background"
        >
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
                    mb="24px"
                    align="center"
                  >
                    Open Source Document Intelligence
                  </Text>
                  <Text
                    weight="medium"
                    size="3"
                    className="hero-description"
                    align="center"
                  >
                    API service to convert complex documents into LLM/RAG-ready
                    data
                  </Text>

                  <Flex
                    className="signup-container"
                    direction="row"
                    gap="16px"
                    align="center"
                  >
                    <button
                      className="signup-button"
                      onClick={handleGetStarted}
                    >
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
            {/* <Flex px="24px" width="100%" align="center" justify="center">
              <div className="hero-content-container">
                <div className="hero-content">
                  <div className="placeholder-window">
                    <div className="window-header">
                      <BetterButton radius="8px" padding="8px 24px">
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Finance
                        </Text>
                      </BetterButton>
                      <BetterButton radius="8px" padding="8px 24px">
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Legal
                        </Text>
                      </BetterButton>
                      <BetterButton radius="8px" padding="8px 24px">
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Scientific
                        </Text>
                      </BetterButton>
                      <BetterButton radius="8px" padding="8px 24px">
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Healthcare
                        </Text>
                      </BetterButton>
                      <BetterButton radius="8px" padding="8px 24px">
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Manufacturing
                        </Text>
                      </BetterButton>
                      <BetterButton radius="8px" padding="8px 24px">
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Technical Reports
                        </Text>
                      </BetterButton>
                      <BetterButton radius="8px" padding="8px 24px">
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Government
                        </Text>
                      </BetterButton>
                      <BetterButton radius="8px" padding="8px 24px">
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Magazines
                        </Text>
                      </BetterButton>
                      <BetterButton radius="8px" padding="8px 24px">
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Textbooks
                        </Text>
                      </BetterButton>
                      <BetterButton radius="8px" padding="8px 24px">
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Newspapers
                        </Text>
                      </BetterButton>
                    </div>
                    <div className="window-content">
                      <div className="loading-animation">
                        <div className="progress-bar">
                          <div className="progress"></div>
                        </div>
                        <div className="status-text">
                          Processing document...
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Flex>
            <div className="features-container">
              <div className="features-gradient-background" />
              <Flex
                direction="column"
                align="center"
                justify="between"
                style={{
                  maxWidth: "1386px",
                  height: "100%",
                  margin: "0 auto",
                  padding: "128px 24px",
                  position: "relative",
                  zIndex: 1,
                }}
              >
                <Flex className="feature-left-box">
                  <Flex direction="column" gap="16px">
                    <Text className="feature-left-box-title">
                      Production-ready document parsing
                    </Text>
                    <Text
                      size="6"
                      weight="medium"
                      className="feature-left-box-subtitle"
                    >
                      High quality HTML, Markdown and Schema extraction
                    </Text>
                    <Text
                      size="5"
                      weight="medium"
                      className="feature-left-box-subtitle"
                    >
                      SOTA models connected and optimized end-to-end.<br></br>
                      <span
                        style={{
                          color: "#ffffffbc",
                          maxWidth: "380px",
                          display: "inline-block",
                        }}
                      >
                        Customize our enterprise-grade pipeline with simple
                        controls for your use-case.
                      </span>{" "}
                    </Text>
                  </Flex>
                  <Flex direction="row" gap="96px" justify="between">
                    <Flex
                      direction="column"
                      gap="32px"
                      flexGrow="1"
                      style={{ zIndex: 5 }}
                      onMouseEnter={() => handleLottieHover(layoutLottieRef)}
                    >
                      <Flex
                        direction="column"
                        className="feature-left-box-lottie"
                        gap="8px"
                      >
                        <Lottie
                          lottieRef={layoutLottieRef}
                          animationData={layoutAnimation}
                          style={{ width: "48px", height: "48px" }}
                          loop={false}
                          autoplay={false}
                        />
                      </Flex>
                      <Text
                        size="8"
                        mt="16px"
                        weight="bold"
                        style={{ color: "#ffffff" }}
                      >
                        Segmentation
                      </Text>
                      <Text
                        size="4"
                        weight="light"
                        style={{ color: "#ffffff" }}
                      >
                        We've built a modular pipeline that can be used as
                      </Text>
                      <Flex direction="column" gap="16px">
                        <Flex className="control-switches">
                          <button
                            className={`control-switch ${segmentationStrategy === "LayoutAnalysis" ? "active" : ""}`}
                            onClick={() =>
                              setSegmentationStrategy("LayoutAnalysis")
                            }
                          >
                            <Text size="2" weight="bold">
                              Layout Analysis
                            </Text>
                          </button>
                          <button
                            className={`control-switch ${segmentationStrategy === "Page" ? "active" : ""}`}
                            onClick={() => setSegmentationStrategy("Page")}
                          >
                            <Text size="2" weight="bold">
                              Page
                            </Text>
                          </button>
                        </Flex>
                        {segmentationStrategy === "LayoutAnalysis" && (
                          <Dropdown
                            value={selectedModel}
                            options={["ChunkrVGT", "AzureDocIntelligence"]}
                            onChange={(value) =>
                              setSelectedModel(
                                value as "AzureDocIntelligence" | "ChunkrVGT"
                              )
                            }
                          />
                        )}
                      </Flex>
                    </Flex>

                    <Flex
                      direction="column"
                      gap="32px"
                      flexGrow="1"
                      onMouseEnter={() => handleLottieHover(ocrLottieRef)}
                    >
                      <Flex
                        direction="column"
                        className="feature-left-box-lottie"
                        gap="8px"
                      >
                        <Lottie
                          lottieRef={ocrLottieRef}
                          animationData={ocrAnimation}
                          style={{ width: "48px", height: "48px" }}
                          loop={false}
                          autoplay={false}
                        />
                      </Flex>
                      <Text
                        size="8"
                        mt="16px"
                        weight="bold"
                        style={{ color: "#ffffff" }}
                      >
                        OCR
                      </Text>
                      <Text
                        size="4"
                        weight="light"
                        style={{ color: "#ffffff" }}
                      >
                        We've built a modular pipeline that can be used as
                      </Text>
                      <Flex className="control-switches">
                        <button
                          className={`control-switch ${ocrMode === "ALL" ? "active" : ""}`}
                          onClick={() => setOcrMode("ALL")}
                        >
                          <Text size="2" weight="bold">
                            ALL
                          </Text>
                        </button>
                        <button
                          className={`control-switch ${ocrMode === "OFF" ? "active" : ""}`}
                          onClick={() => setOcrMode("OFF")}
                        >
                          <Text size="2" weight="bold">
                            OFF
                          </Text>
                        </button>
                      </Flex>
                    </Flex>
                    <Flex
                      direction="column"
                      gap="32px"
                      flexGrow="1"
                      onMouseEnter={() => handleLottieHover(vlmLottieRef)}
                    >
                      <Flex
                        direction="column"
                        className="feature-left-box-lottie"
                        gap="8px"
                      >
                        <Lottie
                          lottieRef={vlmLottieRef}
                          animationData={vlmAnimation}
                          style={{ width: "48px", height: "48px" }}
                          loop={false}
                          autoplay={false}
                        />
                      </Flex>
                      <Text
                        size="8"
                        mt="16px"
                        weight="bold"
                        style={{ color: "#ffffff" }}
                      >
                        VLM Processing
                      </Text>
                      <Text
                        size="4"
                        weight="light"
                        style={{ color: "#ffffff" }}
                      >
                        We've built a modular pipeline that can be used as a
                      </Text>
                      <Flex direction="column" gap="16px">
                        <Flex className="control-switches">
                          <button
                            className={`control-switch ${vlmMode === "SegmentType" ? "active" : ""}`}
                            onClick={() => setVlmMode("SegmentType")}
                          >
                            <Text size="2" weight="bold">
                              Segment Type
                            </Text>
                          </button>
                          <button
                            className={`control-switch ${vlmMode === "ALL" ? "active" : ""}`}
                            onClick={() => setVlmMode("ALL")}
                          >
                            <Text size="2" weight="bold">
                              ALL
                            </Text>
                          </button>
                          <button
                            className={`control-switch ${vlmMode === "OFF" ? "active" : ""}`}
                            onClick={() => setVlmMode("OFF")}
                          >
                            <Text size="2" weight="bold">
                              OFF
                            </Text>
                          </button>
                        </Flex>
                        {vlmMode === "SegmentType" && (
                          <Dropdown
                            value=""
                            options={allSegmentTypes}
                            onChange={(type) => {
                              setSelectedSegmentTypes((prev) =>
                                prev.includes(type)
                                  ? prev.filter((t) => t !== type)
                                  : [...prev, type]
                              );
                            }}
                            multiple={true}
                            selectedValues={selectedSegmentTypes}
                          />
                        )}
                      </Flex>
                    </Flex>
                    <Flex
                      direction="column"
                      gap="32px"
                      flexGrow="1"
                      onMouseEnter={() => handleLottieHover(chunkingLottieRef)}
                    >
                      <Flex
                        direction="column"
                        className="feature-left-box-lottie"
                        gap="8px"
                      >
                        <Lottie
                          lottieRef={chunkingLottieRef}
                          animationData={chunkingAnimation}
                          style={{ width: "48px", height: "48px" }}
                          loop={false}
                          autoplay={false}
                        />
                      </Flex>
                      <Text
                        size="8"
                        mt="16px"
                        weight="bold"
                        style={{ color: "#ffffff" }}
                      >
                        Schema Extract
                      </Text>
                      <Text
                        size="4"
                        weight="light"
                        style={{ color: "#ffffff" }}
                      >
                        We've built a modular pipeline that can be used as a
                      </Text>
                      <Flex className="control-switches">
                        <button
                          className={`control-switch ${schemaMode === "JSON" ? "active" : ""}`}
                          onClick={() => setSchemaMode("JSON")}
                        >
                          <Text size="2" weight="bold">
                            JSON Schema
                          </Text>
                        </button>
                        <button
                          className={`control-switch ${schemaMode === "OFF" ? "active" : ""}`}
                          onClick={() => setSchemaMode("OFF")}
                        >
                          <Text size="2" weight="bold">
                            OFF
                          </Text>
                        </button>
                      </Flex>
                    </Flex>
                  </Flex>
                </Flex>

                <Flex
                  direction="column"
                  gap="32px"
                  className="feature-right-box"
                >
                  <div className="feature-right-box-image" ref={terminalRef}>
                    <Flex
                      className="terminal-header"
                      align="center"
                      justify="center"
                    >
                      <Flex
                        className="terminal-title"
                        align="center"
                        justify="center"
                      >
                        <Flex gap="16px">
                          <BetterButton
                            onClick={() => handleScriptSwitch("curl")}
                            active={selectedScript === "curl"}
                          >
                            <svg
                              width="20px"
                              height="20px"
                              viewBox="0 0 16 16"
                              xmlns="http://www.w3.org/2000/svg"
                              fill="none"
                              stroke="#ffffff"
                              stroke-linecap="round"
                              stroke-linejoin="round"
                              stroke-width="1"
                            >
                              <rect
                                height="10.5"
                                width="12.5"
                                y="2.75"
                                x="1.75"
                              />
                              <path d="m8.75 10.25h2.5m-6.5-4.5 2.5 2.25-2.5 2.25" />
                            </svg>
                            <Text
                              size="1"
                              weight="bold"
                              className="default-font"
                            >
                              curl
                            </Text>
                          </BetterButton>
                          <BetterButton
                            onClick={() => handleScriptSwitch("node")}
                            active={selectedScript === "node"}
                          >
                            <svg
                              fill="#F7DF1E"
                              version="1.1"
                              xmlns="http://www.w3.org/2000/svg"
                              width="20px"
                              height="20px"
                              viewBox="0 0 512 512"
                            >
                              <path
                                display="inline"
                                d="M482.585,147.869v216.113c0,14.025-7.546,27.084-19.672,34.143L275.665,506.241
		c-5.989,3.474-12.782,5.259-19.719,5.259c-6.838,0-13.649-1.785-19.639-5.259l-62.521-36.99c-9.326-5.207-4.775-7.059-1.692-8.128
		c12.454-4.322,14.973-5.318,28.268-12.863c1.387-0.793,3.216-0.483,4.647,0.343l48.031,28.519c1.741,0.981,4.2,0.981,5.801,0
		l187.263-108.086c1.744-0.996,2.862-2.983,2.862-5.053V147.869c0-2.117-1.118-4.094-2.906-5.163L258.874,34.716
		c-1.726-1.01-4.03-1.01-5.768,0L65.962,142.736c-1.818,1.04-2.965,3.079-2.965,5.133v216.113c0,2.069,1.146,4.009,2.954,4.99
		l51.299,29.654c27.829,13.903,44.875-2.485,44.875-18.956V166.309c0-3.017,2.423-5.396,5.439-5.396h23.747
		c2.969,0,5.429,2.378,5.429,5.396v213.362c0,37.146-20.236,58.454-55.452,58.454c-10.816,0-19.347,0-43.138-11.713l-49.098-28.287
		c-12.133-6.995-19.638-20.117-19.638-34.143V147.869c0-14.043,7.505-27.15,19.638-34.135L236.308,5.526
		c11.85-6.701,27.608-6.701,39.357,0l187.248,108.208C475.039,120.748,482.585,133.826,482.585,147.869z M321.171,343.367
		c-55.88,0-68.175-14.048-72.294-41.836c-0.477-2.966-3.018-5.175-6.063-5.175h-27.306c-3.382,0-6.096,2.703-6.096,6.104
		c0,35.56,19.354,77.971,111.759,77.971c66.906,0,105.269-26.339,105.269-72.343c0-45.623-30.827-57.76-95.709-66.35
		c-65.579-8.678-72.243-13.147-72.243-28.508c0-12.661,5.643-29.581,54.216-29.581c43.374,0,59.365,9.349,65.94,38.576
		c0.579,2.755,3.083,4.765,5.923,4.765h27.409c1.7,0,3.315-0.73,4.47-1.943c1.158-1.28,1.773-2.947,1.611-4.695
		c-4.241-50.377-37.713-73.844-105.354-73.844c-60.209,0-96.118,25.414-96.118,68.002c0,46.217,35.729,59,93.5,64.702
		c69.138,6.782,74.504,16.883,74.504,30.488C384.589,333.299,365.655,343.367,321.171,343.367z"
                              ></path>
                            </svg>
                            <Text
                              size="1"
                              weight="bold"
                              className="default-font"
                            >
                              Node
                            </Text>
                          </BetterButton>
                          <BetterButton
                            onClick={() => handleScriptSwitch("python")}
                            active={selectedScript === "python"}
                          >
                            <svg
                              width="20px"
                              height="20px"
                              viewBox="0 0 15 15"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg"
                            >
                              <path
                                d="M6 2.5H7M4.5 4V1.5C4.5 0.947715 4.94772 0.5 5.5 0.5H9.5C10.0523 0.5 10.5 0.947715 10.5 1.5V6.5C10.5 7.05228 10.0523 7.5 9.5 7.5H5.5C4.94772 7.5 4.5 7.94772 4.5 8.5V13.5C4.5 14.0523 4.94772 14.5 5.5 14.5H9.5C10.0523 14.5 10.5 14.0523 10.5 13.5V11M8 4.5H1.5C0.947715 4.5 0.5 4.94772 0.5 5.5V10.5C0.5 11.0523 0.947715 11.5 1.5 11.5H4.5M7 10.5H13.5C14.0523 10.5 14.5 10.0523 14.5 9.5V4.5C14.5 3.94772 14.0523 3.5 13.5 3.5H10.5M8 12.5H9"
                                stroke="#4B8BBE"
                              />
                            </svg>
                            <Text
                              size="1"
                              weight="bold"
                              className="default-font"
                            >
                              Python
                            </Text>
                          </BetterButton>
                          <BetterButton
                            onClick={() => handleScriptSwitch("rust")}
                            active={selectedScript === "rust"}
                          >
                            <svg
                              width="20px"
                              height="20px"
                              viewBox="0 0 15 15"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg"
                            >
                              <path
                                d="M1.99997 10.5H6.99997M2.49997 4.5H8.99997C9.8284 4.5 10.5 5.17157 10.5 6C10.5 6.82843 9.8284 7.5 8.99997 7.5H4.49997M4.49997 4.5V10.5M7.49997 7.5H8.49997C9.60454 7.5 10.5 8.39543 10.5 9.5C10.5 10.0523 10.9477 10.5 11.5 10.5H13M7.49997 0.5L8.84422 1.61046L10.5372 1.19322L11.2665 2.77696L12.9728 3.13557L12.9427 4.87891L14.3245 5.94235L13.541 7.5L14.3245 9.05765L12.9427 10.1211L12.9728 11.8644L11.2665 12.223L10.5372 13.8068L8.84422 13.3895L7.49997 14.5L6.15572 13.3895L4.46279 13.8068L3.73347 12.223L2.02715 11.8644L2.05722 10.1211L0.675476 9.05765L1.45897 7.5L0.675476 5.94235L2.05722 4.87891L2.02715 3.13557L3.73347 2.77696L4.46279 1.19322L6.15572 1.61046L7.49997 0.5Z"
                                stroke="#CE422B"
                                stroke-linejoin="round"
                              />
                            </svg>
                            <Text
                              size="1"
                              weight="bold"
                              className="default-font"
                            >
                              Rust
                            </Text>
                          </BetterButton>
                        </Flex>
                      </Flex>
                      <Flex className="terminal-button-row">
                        <div className="terminal-button minimize"></div>
                        <div className="terminal-button maximize"></div>
                        <div className="terminal-button close"></div>
                      </Flex>
                    </Flex>
                    <div className="curl-command">
                      <CodeBlock
                        code={scripts[selectedScript as keyof typeof scripts]}
                        language={
                          languageMap[
                            selectedScript as keyof typeof languageMap
                          ]
                        }
                        showLineNumbers={false}
                      />
                    </div>
                  </div>
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
                  margin: "0px auto 0px",
                  padding: "128px 24px",
                }}
              >
                <Flex
                  direction="column"
                  px="24px"
                  align="center"
                  justify="center"
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
                      Built for scale
                    </Text>
                  </Flex>
                  <Text
                    size="9"
                    weight="medium"
                    align="center"
                    className="feature-bottom-box-title"
                  >
                    Fast by default<br></br>Powerful by design
                  </Text>
                  <Flex
                    className="feature-bottom-box-wrapper"
                    direction="column"
                  >
                    <Flex
                      direction="row"
                      className="features-bottom-box-container"
                      style={{
                        width: "100%",
                        marginTop: "56px",
                      }}
                    >
                      <FeatureBox
                        icon={
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
                        }
                        title="Connected via Rust"
                        description="Process multiple documents simultaneously with efficient resource utilization"
                        onMouseEnter={() => handleLottieHover(rustLottieRef)}
                      />

                      <FeatureBox
                        icon={
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
                        }
                        title="Support for multiple file types"
                        description="Minimize memory overhead with efficient data handling and processing"
                        onMouseEnter={() =>
                          handleLottieHover(fileuploadLottieRef)
                        }
                      />

                      <FeatureBox
                        icon={
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
                        }
                        title="Optimized last-mile"
                        description="Leverage Rust's native performance for lightning-fast document processing"
                        onMouseEnter={() => handleLottieHover(timerLottieRef)}
                      />
                    </Flex>
                    <Flex
                      direction="row"
                      className="features-bottom-box-container"
                      style={{
                        width: "100%",
                      }}
                    >
                      <FeatureBox
                        icon={
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
                        }
                        title="Built-in visibility"
                        description="Process multiple documents simultaneously with efficient resource utilization"
                        onMouseEnter={() =>
                          handleLottieHover(bargraphLottieRef)
                        }
                      />

                      <FeatureBox
                        icon={
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
                        }
                        title="Simple API / Cloud-Ready"
                        description="Start in seconds with our API, or deploy on your infrastructure for complete control"
                        onMouseEnter={() => handleLottieHover(codeLottieRef)}
                      />

                      <FeatureBox
                        icon={
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
                        }
                        title="Secure in every way"
                        description="Leverage Rust's native performance for lightning-fast document processing"
                        onMouseEnter={() => handleLottieHover(secureLottieRef)}
                      />
                    </Flex>
                  </Flex>
                </Flex>
              </Flex>
            </div>
            <div className="pricing-section">
              <Flex
                direction="column"
                align="center"
                justify="center"
                style={{
                  maxWidth: "1424px",
                  height: "100%",
                  margin: "0 auto",
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
                <Text align="center" className="feature-bottom-box-title">
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
                    title="Dev"
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
                    title="Startup"
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
                    title="Enterprise"
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

                <Flex
                  direction="row"
                  justify="center"
                  mt="48px"
                  className="pricing-container"
                  style={{
                    width: "100%",
                    padding: "0 24px",
                    position: "relative",
                    zIndex: 2,
                  }}
                >
                  <Flex
                    direction="row"
                    gap="96px"
                    className="pricing-enterprise-container"
                  >
                    <Flex direction="column" justify="center">
                      <Text size="6" weight="medium" style={{ color: "white" }}>
                        On-prem
                      </Text>
                      <Text
                        size="8"
                        mb="24px"
                        weight="bold"
                        mt="6"
                        style={{ color: "white" }}
                      >
                        Custom Plans
                      </Text>

                      <button className="signup-button">
                        <Text size="5" weight="bold">
                          Book a call
                        </Text>
                      </button>
                    </Flex>
                    <Flex direction="column" justify="center" height="100%">
                      <Flex direction="row" gap="48px">
                        <Flex direction="column" gap="24px">
                          <Flex
                            align="center"
                            gap="12px"
                            className="feature-item"
                          >
                            <Flex
                              align="center"
                              justify="center"
                              className="feature-checkmark-container"
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#ffffffd0"
                                  stroke="#ffffffd0"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Self-hosted deployment
                            </Text>
                          </Flex>
                          <Flex
                            align="center"
                            gap="12px"
                            className="feature-item"
                          >
                            <Flex
                              align="center"
                              justify="center"
                              className="feature-checkmark-container"
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#ffffffd0"
                                  stroke="#ffffffd0"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Unlimited processing
                            </Text>
                          </Flex>
                          <Flex
                            align="center"
                            gap="12px"
                            className="feature-item"
                          >
                            <Flex
                              align="center"
                              justify="center"
                              className="feature-checkmark-container"
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#ffffffd0"
                                  stroke="#ffffffd0"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Custom SLAs
                            </Text>
                          </Flex>
                        </Flex>

                        <Flex direction="column" gap="24px">
                          <Flex
                            align="center"
                            gap="12px"
                            className="feature-item"
                          >
                            <Flex
                              align="center"
                              justify="center"
                              className="feature-checkmark-container"
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#ffffffd0"
                                  stroke="#ffffffd0"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Dedicated support
                            </Text>
                          </Flex>
                          <Flex
                            align="center"
                            gap="12px"
                            className="feature-item"
                          >
                            <Flex
                              align="center"
                              justify="center"
                              className="feature-checkmark-container"
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#ffffffd0"
                                  stroke="#ffffffd0"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Custom integrations
                            </Text>
                          </Flex>
                          <Flex
                            align="center"
                            gap="12px"
                            className="feature-item"
                          >
                            <Flex
                              align="center"
                              justify="center"
                              className="feature-checkmark-container"
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#ffffffd0"
                                  stroke="#ffffffd0"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Priority features
                            </Text>
                          </Flex>
                        </Flex>

                        <Flex direction="column" gap="24px">
                          <Flex
                            align="center"
                            gap="12px"
                            className="feature-item"
                          >
                            <Flex
                              align="center"
                              justify="center"
                              className="feature-checkmark-container"
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#ffffffd0"
                                  stroke="#ffffffd0"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Air-gapped deployment
                            </Text>
                          </Flex>
                          <Flex
                            align="center"
                            gap="12px"
                            className="feature-item"
                          >
                            <Flex
                              align="center"
                              justify="center"
                              className="feature-checkmark-container"
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#ffffffd0"
                                  stroke="#ffffffd0"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              24/7 support
                            </Text>
                          </Flex>
                        </Flex>
                      </Flex>
                    </Flex>
                  </Flex>
                </Flex>
              </Flex>
            </div>
            <Flex
              direction="column"
              align="center"
              gap="24px"
              className="bottom-cta-container"
            >
              <Flex
                direction="column"
                align="center"
                justify="center"
                className="bottom-cta-icon"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="48"
                  height="48"
                  viewBox="0 0 30 30"
                  fill="none"
                  style={{
                    filter: "drop-shadow(0 0 4px rgba(255, 255, 255, 0.0))",
                  }}
                >
                  <defs>
                    <linearGradient
                      id="paint0_linear_236_740"
                      x1="15"
                      y1="3.75"
                      x2="15"
                      y2="26.25"
                      gradientUnits="userSpaceOnUse"
                    >
                      <stop stop-color="white" />
                      <stop offset="1" stop-color="rgba(220, 228, 221, 0.9)" />
                    </linearGradient>
                    <filter id="glow">
                      <feGaussianBlur stdDeviation="0.4" result="coloredBlur" />
                      <feMerge>
                        <feMergeNode in="coloredBlur" />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                  </defs>
                  <path
                    d="M7.35 12.225C8.03148 12.4978 8.77803 12.5646 9.4971 12.4171C10.2162 12.2695 10.8761 11.9142 11.3952 11.3952C11.9142 10.8761 12.2695 10.2162 12.4171 9.4971C12.5646 8.77803 12.4978 8.03148 12.225 7.35C13.0179 7.13652 13.7188 6.6687 14.2201 6.01836C14.7214 5.36802 14.9954 4.57111 15 3.75C17.225 3.75 19.4001 4.4098 21.2502 5.64597C23.1002 6.88213 24.5422 8.63914 25.3936 10.6948C26.2451 12.7505 26.4679 15.0125 26.0338 17.1948C25.5998 19.3771 24.5283 21.3816 22.955 22.955C21.3816 24.5283 19.3771 25.5998 17.1948 26.0338C15.0125 26.4679 12.7505 26.2451 10.6948 25.3936C8.63914 24.5422 6.88213 23.1002 5.64597 21.2502C4.4098 19.4001 3.75 17.225 3.75 15C4.57111 14.9954 5.36802 14.7214 6.01836 14.2201C6.6687 13.7188 7.13652 13.0179 7.35 12.225Z"
                    stroke="url(#paint0_linear_236_740)"
                    stroke-width="3"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    filter="url(#glow)"
                  />
                </svg>
              </Flex>
              <Text className="bottom-cta-title">What will you build?</Text>
              <Text
                size="4"
                mb="16px"
                weight="bold"
                style={{ color: "rgba(255, 255, 255, 0.9)" }}
              >
                Leverage our document ingestion infrastructure to build SOTA AI
                applications
              </Text>
              <button className="signup-button">
                <Text size="5" weight="bold">
                  Get started for free
                </Text>
              </button>
            </Flex> */}
          </Flex>

          {/* <Footer /> */}
        </Flex>
      </MomentumScroll>
    </>
  );
};

export default Home;
