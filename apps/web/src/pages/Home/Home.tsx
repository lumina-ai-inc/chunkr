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
import PricingCard from "../../components/PricingCard/PricingCard";
import CodeBlock from "../../components/CodeBlock/CodeBlock";
import BetterButton from "../../components/BetterButton/BetterButton";
import {
  curlExample,
  // nodeExample,
  pythonExample,
} from "../../components/CodeBlock/exampleScripts";
import MomentumScroll from "../../components/MomentumScroll/MomentumScroll";
import Header from "../../components/Header/Header";
import Footer from "../../components/Footer/Footer";

// Animation Imports
import Lottie from "lottie-react";
import { LottieRefCurrentProps } from "lottie-react";
import fileuploadAnimation from "../../assets/animations/fileupload.json";
import bargraphAnimation from "../../assets/animations/bargraph.json";
import codeAnimation from "../../assets/animations/code.json";
import secureAnimation from "../../assets/animations/secure.json";
import ocrAnimation from "../../assets/animations/ocr.json";
import chunkingAnimation from "../../assets/animations/chunking.json";
import vlmAnimation from "../../assets/animations/vlm.json";
import layoutAnimation from "../../assets/animations/layout.json";
import rustAnimation from "../../assets/animations/rust.json";
import devXAnimation from "../../assets/animations/devX.json";
import checklistAnimation from "../../assets/animations/checklist.json";
import apiPriceAnimation from "../../assets/animations/apiPrice.json";
import onPremAnimation from "../../assets/animations/onPrem.json";
// Service Imports
import { createCheckoutSession } from "../../services/stripeService";
import { loadStripe } from "@stripe/stripe-js";
import useMonthlyUsage from "../../hooks/useMonthlyUsage";
import Viewer from "../../components/Viewer/Viewer";
import { TaskResponse } from "../../models/taskResponse.model";
import toast from "react-hot-toast";

const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_API_KEY, {});

// Add new type and constants
type DocumentCategory = {
  id: string;
  label: string;
  pdfName: string;
};

const DOCUMENT_CATEGORIES: DocumentCategory[] = [
  { id: "financial", label: "Financial Reports ", pdfName: "financial" },
  { id: "legal", label: "Legal Documents", pdfName: "legal" },
  { id: "scientific", label: "Research Papers", pdfName: "science" },
  { id: "technical", label: "Technical Manuals", pdfName: "specs" },
  { id: "medical", label: "Medical Files", pdfName: "medical" },
  { id: "consulting", label: "Consulting Reports", pdfName: "consulting" },
  { id: "government", label: "Government Reports", pdfName: "gov" },
  { id: "textbook", label: "Textbooks", pdfName: "textbook" },
];

const BASE_URL = "https://chunkr-web.s3.us-east-1.amazonaws.com/landing_page";

const Home = () => {
  const auth = useAuth();
  const isAuthenticated = auth.isAuthenticated;
  const navigate = useNavigate();

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
  const parsingLottieRef = useRef<LottieRefCurrentProps>(null);
  const notesLottieRef = useRef<LottieRefCurrentProps>(null);
  const curlLottieRef = useRef<LottieRefCurrentProps>(null);
  const vlmLottieRef = useRef<LottieRefCurrentProps>(null);
  const chunkingLottieRef = useRef<LottieRefCurrentProps>(null);
  const layoutLottieRef = useRef<LottieRefCurrentProps>(null);
  const sparklesLottieRef = useRef<LottieRefCurrentProps>(null);
  const devXLottieRef = useRef<LottieRefCurrentProps>(null);
  const checklistLottieRef = useRef<LottieRefCurrentProps>(null);
  const apiPriceLottieRef = useRef<LottieRefCurrentProps>(null);
  const onPremLottieRef = useRef<LottieRefCurrentProps>(null);
  const [isScrolled, setIsScrolled] = useState(false);
  const [selectedScript, setSelectedScript] = useState("python");

  const [checkoutClientSecret, setCheckoutClientSecret] = useState<
    string | null
  >(null);
  const [selectedFormat, setSelectedFormat] = useState<"HTML" | "Markdown">(
    "HTML"
  );

  const { data: usageData, isLoading: isUsageDataLoading } = useMonthlyUsage();
  const currentTier = usageData?.[0]?.tier;

  const pricingRef = useRef<HTMLDivElement>(null);

  const [selectedCategory, setSelectedCategory] = useState<string>("technical");
  const [taskResponse, setTaskResponse] = useState<TaskResponse | null>(null);

  // Function to fetch task response and update PDF URL
  const fetchTaskResponse = async (pdfName: string) => {
    try {
      // Fetch from the same base URL where PDFs are stored
      const response = await fetch(
        `${BASE_URL}/output/${pdfName}_response.json`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: TaskResponse = await response.json();

      // Update the PDF URL in the response
      if (data.output) {
        data.output.pdf_url = `${BASE_URL}/input/${pdfName}.pdf`;
      }

      setTaskResponse(data);
    } catch (error) {
      console.error("Error loading task response:", error);
    }
  };

  // Effect to fetch task response when category changes
  useEffect(() => {
    const category = DOCUMENT_CATEGORIES.find(
      (cat) => cat.id === selectedCategory
    );
    if (category) {
      fetchTaskResponse(category.pdfName);
    }
  }, [selectedCategory]);

  // Update the placeholder window content
  const renderPlaceholderWindow = () => (
    <div className="placeholder-window">
      <div className="window-header">
        <Flex
          width="100%"
          justify="between"
          align="center"
          minWidth="1247px"
          overflow="auto"
        >
          {DOCUMENT_CATEGORIES.map((category) => (
            <BetterButton
              key={category.id}
              radius="8px"
              padding="8px 24px"
              onClick={() => setSelectedCategory(category.id)}
              active={selectedCategory === category.id}
            >
              <Text size="1" weight="medium" style={{ color: "white" }}>
                {category.label}
              </Text>
            </BetterButton>
          ))}
        </Flex>
      </div>
      <div className="window-content">
        {taskResponse && (
          <Viewer
            task={taskResponse}
            externalFormat={selectedFormat}
            hideHeader={true}
          />
        )}
      </div>
    </div>
  );

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
    if (parsingLottieRef.current) {
      parsingLottieRef.current.pause();
    }
    if (sparklesLottieRef.current) {
      sparklesLottieRef.current.pause();
    }
    if (devXLottieRef.current) {
      devXLottieRef.current.pause();
    }
    if (checklistLottieRef.current) {
      checklistLottieRef.current.pause();
    }
    if (apiPriceLottieRef.current) {
      apiPriceLottieRef.current.pause();
    }
    if (onPremLottieRef.current) {
      onPremLottieRef.current.pause();
    }
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 0);

      // If we have a hash and we've scrolled to the pricing section
      if (window.location.hash === "#pricing") {
        const pricingElement = pricingRef.current;
        if (pricingElement) {
          const rect = pricingElement.getBoundingClientRect();
          // If pricing section is in view (with some buffer)
          if (rect.top >= -100 && rect.top <= 100) {
            history.replaceState(null, "", window.location.pathname);
          }
        }
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    const handleHashChange = () => {
      if (window.location.hash === "#pricing" && pricingRef.current) {
        // Let MomentumScroll handle the scrolling
        // The hash will be removed by MomentumScroll's onComplete callback
        return;
      }
    };

    handleHashChange();
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const handleScriptSwitch = (script: string) => {
    setSelectedScript(script);
  };

  const scripts = {
    curl: curlExample,
    // node: nodeExample,
    python: pythonExample,
  };

  const languageMap = {
    python: "python",
    curl: "bash",
    // node: "javascript",
  };

  // Create a unified auth redirect handler
  const handleAuthRedirect = (returnPath?: string) => {
    const currentPath = window.location.pathname + window.location.hash;
    auth.signinRedirect({
      state: { returnTo: returnPath || currentPath },
    });
  };

  // Update handleGetStarted
  const handleGetStarted = () => {
    if (auth.isAuthenticated) {
      navigate("/dashboard");
    } else {
      handleAuthRedirect("/dashboard");
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

  // Update handleCheckout
  const handleCheckout = async (tier: string) => {
    if (!auth.isAuthenticated) {
      handleAuthRedirect("/#pricing"); // Preserve pricing section
      return;
    }
    try {
      const session = await createCheckoutSession(
        auth.user?.access_token || "",
        tier
      );
      setCheckoutClientSecret(session.client_secret);
    } catch (error) {
      console.error("Failed to create checkout session:", error);
      toast.error(
        "Failed to start checkout process - refresh page and try again."
      );
    }
  };

  const handleFormatSwitch = (format: "HTML" | "Markdown") => {
    setSelectedFormat(format);
  };

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
                    chunks
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
            <Flex
              px="24px"
              width="100%"
              align="center"
              justify="center"
              direction="column"
              className="hero-content-container-main"
            >
              <div className="hero-content-container">
                <div className="hero-content">{renderPlaceholderWindow()}</div>
              </div>
              <Flex className="hero-content-container-switch-row">
                <Flex
                  align="center"
                  gap="6px"
                  className={`hero-content-switch ${
                    selectedFormat === "HTML" ? "active" : ""
                  }`}
                  onClick={() => handleFormatSwitch("HTML")}
                >
                  <svg
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M4.17456 5.15007C4.08271 4.54492 4.55117 4 5.16324 4H18.8368C19.4488 4 19.9173 4.54493 19.8254 5.15007L18.0801 16.6489C18.03 16.9786 17.8189 17.2617 17.5172 17.4037L12.4258 19.7996C12.1561 19.9265 11.8439 19.9265 11.5742 19.7996L6.4828 17.4037C6.18107 17.2617 5.96997 16.9786 5.91993 16.6489L4.17456 5.15007Z"
                      stroke={selectedFormat === "HTML" ? "#000" : "#FFFFFF"}
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M15 7.5H9.5V11H14.5V14.5L12.3714 15.3514C12.133 15.4468 11.867 15.4468 11.6286 15.3514L9.5 14.5"
                      stroke={selectedFormat === "HTML" ? "#000" : "#FFFFFF"}
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <Text size="2" weight="bold">
                    HTML
                  </Text>
                </Flex>
                <Flex
                  align="center"
                  gap="6px"
                  className={`hero-content-switch ${
                    selectedFormat === "Markdown" ? "active" : ""
                  }`}
                  onClick={() => handleFormatSwitch("Markdown")}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 15 15"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M2.5 5.5L2.85355 5.14645C2.71055 5.00345 2.4955 4.96067 2.30866 5.03806C2.12182 5.11545 2 5.29777 2 5.5H2.5ZM4.5 7.5L4.14645 7.85355L4.5 8.20711L4.85355 7.85355L4.5 7.5ZM6.5 5.5H7C7 5.29777 6.87818 5.11545 6.69134 5.03806C6.5045 4.96067 6.28945 5.00345 6.14645 5.14645L6.5 5.5ZM10.5 9.5L10.1464 9.85355L10.5 10.2071L10.8536 9.85355L10.5 9.5ZM1.5 3H13.5V2H1.5V3ZM14 3.5V11.5H15V3.5H14ZM13.5 12H1.5V13H13.5V12ZM1 11.5V3.5H0V11.5H1ZM1.5 12C1.22386 12 1 11.7761 1 11.5H0C0 12.3284 0.671574 13 1.5 13V12ZM14 11.5C14 11.7761 13.7761 12 13.5 12V13C14.3284 13 15 12.3284 15 11.5H14ZM13.5 3C13.7761 3 14 3.22386 14 3.5H15C15 2.67157 14.3284 2 13.5 2V3ZM1.5 2C0.671573 2 0 2.67157 0 3.5H1C1 3.22386 1.22386 3 1.5 3V2ZM3 10V5.5H2V10H3ZM2.14645 5.85355L4.14645 7.85355L4.85355 7.14645L2.85355 5.14645L2.14645 5.85355ZM4.85355 7.85355L6.85355 5.85355L6.14645 5.14645L4.14645 7.14645L4.85355 7.85355ZM6 5.5V10H7V5.5H6ZM10 5V9.5H11V5H10ZM8.14645 7.85355L10.1464 9.85355L10.8536 9.14645L8.85355 7.14645L8.14645 7.85355ZM10.8536 9.85355L12.8536 7.85355L12.1464 7.14645L10.1464 9.14645L10.8536 9.85355Z"
                      fill={selectedFormat === "HTML" ? "#FFF" : "#000"}
                    />
                  </svg>
                  <Text size="2" weight="bold">
                    Markdown
                  </Text>
                </Flex>
              </Flex>
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
                  padding: "256px 24px 124px 24px",
                  position: "relative",
                  zIndex: 1,
                }}
              >
                <Flex className="feature-left-box">
                  <Flex
                    direction="column"
                    gap="16px"
                    onMouseEnter={() => handleLottieHover(devXLottieRef)}
                  >
                    <Flex className="yc-tag" gap="12px">
                      <Lottie
                        lottieRef={devXLottieRef}
                        animationData={devXAnimation}
                        style={{ width: "16px", height: "16px" }}
                        loop={false}
                        autoplay={false}
                      />
                      <Text
                        size="2"
                        weight="medium"
                        style={{
                          color: "#ffffff",
                          textShadow: "0 0 10px rgba(255, 255, 255, 0.45)",
                          letterSpacing: "0.02em",
                        }}
                      >
                        Simple DevX
                      </Text>
                    </Flex>
                    <Text className="feature-left-box-title">
                      Lightning fast integration
                    </Text>
                    <Text
                      size="5"
                      weight="medium"
                      className="feature-left-box-subtitle"
                    >
                      Build stand out experiences with top-tier document
                      parsing.<br></br>
                      <span
                        style={{
                          color: "#ffffffbc",
                          maxWidth: "460px",
                          display: "inline-block",
                        }}
                      >
                        Configure your pipeline with simple controls to setup
                        the optimal balance of speed, quality, and features.
                      </span>{" "}
                    </Text>
                  </Flex>
                  <Flex
                    direction="row"
                    gap="96px"
                    justify="between"
                    className="feature-left-box-controls"
                    style={{ display: "none" }}
                  ></Flex>
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
                          {/* <BetterButton
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
                          </BetterButton> */}
                          {/* <BetterButton
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
                          </BetterButton> */}
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
                  onMouseEnter={() => handleLottieHover(checklistLottieRef)}
                >
                  <Flex direction="column" align="center">
                    <Flex className="yc-tag">
                      <Lottie
                        lottieRef={checklistLottieRef}
                        animationData={checklistAnimation}
                        style={{ width: "16px", height: "16px" }}
                        loop={false}
                        autoplay={false}
                      />
                      <Text
                        size="2"
                        weight="medium"
                        style={{
                          color: "#ffffff",
                          textShadow: "0 0 10px rgba(255, 255, 255, 0.45)",
                          letterSpacing: "0.02em",
                        }}
                      >
                        Feature Complete
                      </Text>
                    </Flex>
                    <Text
                      size="9"
                      mt="16px"
                      weight="medium"
                      align="center"
                      className="feature-bottom-box-title"
                    >
                      Your RAG app's <br></br> Secret Weapon
                    </Text>
                    <Text
                      size="5"
                      weight="medium"
                      className="feature-left-box-subtitle"
                      align="center"
                      mt="16px"
                      style={{ maxWidth: "504px" }}
                    >
                      Production-ready vision infrastructure for every use case.{" "}
                      <br></br>
                      <span style={{ color: "#ffffffbc" }}>
                        From word level bounding boxes to segment level custom
                        VLM processing - we've got you covered.
                      </span>
                    </Text>
                  </Flex>
                  <Flex
                    className="feature-bottom-box-wrapper"
                    gap="16px"
                    justify="between"
                    direction="column"
                  >
                    <Flex
                      direction="row"
                      className="features-bottom-box-container"
                      justify="between"
                      style={{
                        width: "100%",
                        marginTop: "56px",
                      }}
                    >
                      <FeatureBox
                        icon={
                          <Lottie
                            lottieRef={layoutLottieRef}
                            animationData={layoutAnimation}
                            loop={false}
                            autoplay={false}
                            style={{
                              width: 32,
                              height: 32,
                            }}
                          />
                        }
                        title="Layout Analysis"
                        description="Identify over 11 segment types like Titles, Pictures, Tables, and List-items"
                        onMouseEnter={() => handleLottieHover(layoutLottieRef)}
                        data-feature="layout"
                      />

                      <FeatureBox
                        icon={
                          <Lottie
                            lottieRef={ocrLottieRef}
                            animationData={ocrAnimation}
                            loop={false}
                            autoplay={false}
                            style={{
                              width: 32,
                              height: 32,
                            }}
                          />
                        }
                        title="Multi-lingual OCR"
                        description="Word-level OCR with multi-lingual support and auto text-layer detection"
                        onMouseEnter={() => handleLottieHover(ocrLottieRef)}
                        data-feature="ocr"
                      />

                      <FeatureBox
                        icon={
                          <Lottie
                            lottieRef={vlmLottieRef}
                            animationData={vlmAnimation}
                            loop={false}
                            autoplay={false}
                            style={{
                              width: 32,
                              height: 32,
                            }}
                          />
                        }
                        title="VLM's for complex parsing"
                        description="Powerful defaults for tables + formulas, and custom parsing prompts for any segment"
                        onMouseEnter={() => handleLottieHover(vlmLottieRef)}
                        data-feature="vlm"
                      />
                    </Flex>
                    <Flex
                      direction="row"
                      className="features-bottom-box-container"
                      justify="between"
                      style={{
                        width: "100%",
                      }}
                    >
                      <FeatureBox
                        icon={
                          <Lottie
                            lottieRef={chunkingLottieRef}
                            animationData={chunkingAnimation}
                            loop={false}
                            autoplay={false}
                            style={{
                              width: 32,
                              height: 32,
                            }}
                            data-feature="chunking"
                          />
                        }
                        title="Semantic Chunking"
                        description="Set your own chunk size, and let us handle the logic to maintain semantic integrity"
                        onMouseEnter={() =>
                          handleLottieHover(chunkingLottieRef)
                        }
                        data-feature="chunking"
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
                        title="Flexible File Handling"
                        description="Process PDFs, PPTs, Word docs & images via direct upload, URLs, or base64 "
                        onMouseEnter={() =>
                          handleLottieHover(fileuploadLottieRef)
                        }
                        data-feature="fileupload"
                      />

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
                        title="Built-in Visibility"
                        description="Dashboard to track ingest, view extraction results, and experiment with configurations"
                        onMouseEnter={() =>
                          handleLottieHover(bargraphLottieRef)
                        }
                        data-feature="visibility"
                      />
                    </Flex>
                    <Flex
                      direction="row"
                      className="features-bottom-box-container"
                      justify="between"
                      style={{
                        width: "100%",
                      }}
                    >
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
                        title="Secure & Private"
                        description="Zero data retention with custom expiration times, SOC2 + HIPPA in progress"
                        onMouseEnter={() => handleLottieHover(secureLottieRef)}
                        data-feature="secure"
                      />

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
                        title="Last-mile handled"
                        description="Built in Rust for blazing-fast operations and high reliability - under 0.05% error rate"
                        onMouseEnter={() => handleLottieHover(rustLottieRef)}
                        data-feature="rust"
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
                        title="Cloud-ready / Self-host"
                        description="Hit our API, or deploy on your own compute with our Dockers and Helm charts"
                        onMouseEnter={() => handleLottieHover(codeLottieRef)}
                        data-feature="code"
                      />
                    </Flex>
                  </Flex>
                </Flex>
              </Flex>
            </div>
            <div id="pricing" ref={pricingRef} className="pricing-section">
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
                <Flex
                  direction="column"
                  align="center"
                  onMouseEnter={() => handleLottieHover(apiPriceLottieRef)}
                >
                  <Flex className="yc-tag">
                    <Lottie
                      lottieRef={apiPriceLottieRef}
                      animationData={apiPriceAnimation}
                      style={{ width: "16px", height: "16px" }}
                      loop={false}
                      autoplay={false}
                    />
                    <Text
                      size="2"
                      weight="medium"
                      style={{
                        color: "#ffffff",
                        textShadow: "0 0 10px rgba(255, 255, 255, 0.45)",
                        letterSpacing: "0.02em",
                      }}
                    >
                      API Pricing
                    </Text>
                  </Flex>
                  <Text
                    align="center"
                    mt="16px"
                    className="feature-bottom-box-title"
                  >
                    Simple plans that scale with you
                  </Text>
                  <Text
                    size="5"
                    weight="medium"
                    align="center"
                    mt="16px"
                    className="feature-left-box-subtitle"
                  >
                    Start with included monthly pages - then pay-as-you-go
                  </Text>
                </Flex>

                <Flex
                  direction="row"
                  justify="between"
                  gap="48px"
                  wrap="wrap"
                  className="pricing-container"
                  style={{
                    width: "100%",
                    marginTop: "56px",
                    padding: "0 24px",
                    position: "relative",
                    zIndex: 2,
                  }}
                >
                  {(!auth.isAuthenticated ||
                    currentTier === "Free" ||
                    isUsageDataLoading) && (
                    <PricingCard
                      title="Free"
                      credits={100}
                      price="Free"
                      period="month"
                      features={[
                        "200 pages included",
                        "No payment info required",
                        "Discord community support",
                      ]}
                      buttonText="Get Started"
                      tier="Free"
                      onCheckout={handleCheckout}
                      stripePromise={stripePromise}
                      clientSecret={checkoutClientSecret || undefined}
                      currentTier={currentTier}
                      isAuthenticated={auth.isAuthenticated}
                    />
                  )}

                  <PricingCard
                    title="Starter"
                    credits={10000}
                    price={50}
                    period="month"
                    features={[
                      "5,000 pages included",
                      "$0.01 / page ",
                      "Community + Email support",
                    ]}
                    buttonText="Get Started"
                    tier="Starter"
                    onCheckout={handleCheckout}
                    stripePromise={stripePromise}
                    clientSecret={checkoutClientSecret || undefined}
                    currentTier={currentTier}
                    isAuthenticated={auth.isAuthenticated}
                  />

                  <PricingCard
                    title="Dev"
                    credits={150000}
                    price={200}
                    period="month"
                    features={[
                      "25,000 pages included",
                      "$0.008 / page",
                      "Priority support channel",
                    ]}
                    buttonText="Get Started"
                    tier="Dev"
                    onCheckout={handleCheckout}
                    stripePromise={stripePromise}
                    clientSecret={checkoutClientSecret || undefined}
                    currentTier={currentTier}
                    isAuthenticated={auth.isAuthenticated}
                  />

                  <PricingCard
                    title="Growth"
                    credits={500000}
                    price={500}
                    period="month"
                    features={[
                      "100,000 pages included",
                      "$0.005 / page",
                      "Dedicated founder support",
                    ]}
                    buttonText="Get Started"
                    tier="Growth"
                    onCheckout={handleCheckout}
                    stripePromise={stripePromise}
                    clientSecret={checkoutClientSecret || undefined}
                    currentTier={currentTier}
                    isAuthenticated={auth.isAuthenticated}
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
                        Enterprise
                      </Text>
                      <Text
                        size="8"
                        mb="24px"
                        weight="bold"
                        mt="6"
                        style={{ color: "white" }}
                      >
                        Custom
                      </Text>

                      <BetterButton
                        padding="12px 24px"
                        radius="8px"
                        onClick={() => {
                          window.open("https://cal.com/mehulc/30min", "_blank");
                        }}
                      >
                        <Text
                          size="4"
                          weight="medium"
                          style={{ color: "white" }}
                        >
                          Book a call
                        </Text>
                      </BetterButton>
                    </Flex>
                    <Flex direction="column" justify="center" height="100%">
                      <Flex
                        direction="row"
                        gap="48px"
                        wrap="wrap"
                        className="enterprise-feature-item-container"
                      >
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
                                width="12"
                                height="12"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#000000"
                                  stroke="#000000"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Custom deployment strategy
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
                                width="12"
                                height="12"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#000000"
                                  stroke="#000000"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              High volume discounts
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
                                width="12"
                                height="12"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#000000"
                                  stroke="#000000"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              24/7 founder-led support
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
                                width="12"
                                height="12"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#000000"
                                  stroke="#000000"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Custom SLAs & agreements
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
                                width="12"
                                height="12"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#000000"
                                  stroke="#000000"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Tuned to your data
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
                                width="12"
                                height="12"
                                viewBox="0 0 16 16"
                                fill="none"
                              >
                                <path
                                  d="M13.3 4.3L6 11.6L2.7 8.3L3.3 7.7L6 10.4L12.7 3.7L13.3 4.3Z"
                                  fill="#000000"
                                  stroke="#000000"
                                />
                              </svg>
                            </Flex>
                            <Text
                              size="2"
                              style={{ color: "rgba(255, 255, 255, 0.8)" }}
                            >
                              Dedicated migration support
                            </Text>
                          </Flex>
                        </Flex>
                      </Flex>
                    </Flex>
                  </Flex>
                </Flex>
              </Flex>
            </div>

            <div id="pricing" className="pricing-section">
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
                <Flex
                  direction="column"
                  align="center"
                  onMouseEnter={() => handleLottieHover(onPremLottieRef)}
                >
                  <Flex className="yc-tag">
                    <Lottie
                      lottieRef={onPremLottieRef}
                      animationData={onPremAnimation}
                      style={{ width: "16px", height: "16px" }}
                      loop={false}
                      autoplay={false}
                    />
                    <Text
                      size="2"
                      weight="medium"
                      style={{
                        color: "#ffffff",
                        textShadow: "0 0 10px rgba(255, 255, 255, 0.45)",
                        letterSpacing: "0.02em",
                      }}
                    >
                      On-prem
                    </Text>
                  </Flex>
                  <Text
                    align="center"
                    mt="16px"
                    className="feature-bottom-box-title"
                  >
                    Bring your own compute
                  </Text>
                </Flex>
                <Flex
                  direction="row"
                  justify="center"
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
                    title="Research"
                    credits={0}
                    price="Free"
                    period=""
                    features={[
                      "Non-commercial use",
                      "Community support",
                      "All features included",
                      "Easy to deploy",
                      "Docker images + Helm charts",
                      "Perfect for testing",
                    ]}
                    buttonText="Github"
                    tier="Free"
                    isAuthenticated={auth.isAuthenticated}
                    currentTier={currentTier}
                    isCallToAction={true}
                    callToActionUrl="https://github.com/lumina-ai-inc/chunkr"
                  />

                  <PricingCard
                    title="Commercial License"
                    credits={0}
                    price="Custom"
                    period="month"
                    features={[
                      "Managed by us in your cloud / Self-host",
                      "Unlimited pages - fixed monthly price",
                      "Tuned to your data",
                      "Compliance support",
                      "Enterprise-grade SLAs",
                      "24/7 founder-led support",
                    ]}
                    buttonText="Book a Call"
                    tier="Commercial"
                    isCallToAction={true}
                    callToActionUrl="https://cal.com/mehulc/30min"
                    isAuthenticated={auth.isAuthenticated}
                    currentTier={currentTier}
                  />
                </Flex>
              </Flex>
            </div>
          </Flex>

          <Footer />
        </Flex>
      </MomentumScroll>
    </>
  );
};

const FeatureBox = ({
  icon,
  title,
  description,
  onMouseEnter,
  "data-feature": dataFeature,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  onMouseEnter?: () => void;
  "data-feature": string;
}) => {
  return (
    <Flex
      direction="column"
      className="feature-bottom-box"
      onMouseEnter={onMouseEnter}
      data-feature={dataFeature}
    >
      <Flex align="center" justify="center" className="feature-bottom-box-icon">
        {icon}
      </Flex>

      <Text
        size="6"
        weight="bold"
        style={{
          color: "white",
          marginTop: "32px",
          transition: "margin 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
        }}
        className="feature-box-title"
      >
        {title}
      </Text>
      <Text
        size="3"
        weight="medium"
        style={{
          color: "rgba(255, 255, 255, 0.7)",
          marginTop: "12px",
          transition: "margin 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
        }}
        className="feature-box-description"
      >
        {description}
      </Text>
    </Flex>
  );
};

export default Home;
