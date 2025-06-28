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
import BetterButton from "../../components/BetterButton/BetterButton";
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
  { id: "technical", label: "Technical", pdfName: "technical" },
  { id: "billing", label: "Billing", pdfName: "billing" },
  { id: "construction", label: "Construction", pdfName: "construction" },
  { id: "consulting", label: "Consulting", pdfName: "consulting" },
  { id: "education", label: "Education", pdfName: "education" },
  { id: "financial", label: "Financial", pdfName: "financial" },
  { id: "government", label: "Government", pdfName: "government" },
  { id: "historical", label: "Historical", pdfName: "historical" },
  { id: "legal", label: "Legal", pdfName: "legal" },
  { id: "medical", label: "Medical", pdfName: "medical" },
  { id: "research", label: "Research", pdfName: "research" },
];

const BASE_URL =
  "https://chunkr-web.s3.us-east-1.amazonaws.com/landing_page_v2";

const Home = () => {
  const auth = useAuth();
  const isAuthenticated = auth.isAuthenticated;
  const navigate = useNavigate();

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
      // Updated to match new structure from frontend_pdfs.py
      const response = await fetch(
        `${BASE_URL}/output/${pdfName}/${pdfName}_response.json`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: TaskResponse = await response.json();

      // Update the PDF URL in the response to match new structure
      if (data.output) {
        data.output.pdf_url = `${BASE_URL}/input/${pdfName}/${pdfName}.pdf`;
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
          justify="start"
          align="center"
          overflow="auto"
          gap="8px"
          p="16px"
          pb="12px"
          className="category-scroll-container"
        >
          {DOCUMENT_CATEGORIES.map((category) => (
            <BetterButton
              key={category.id}
              radius="8px"
              padding="8px 24px"
              onClick={() => setSelectedCategory(category.id)}
              active={selectedCategory === category.id}
            >
              <Text
                size="1"
                weight="medium"
                style={{ color: "white", width: "max-content" }}
              >
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
                    LLM-ready inputs <br></br>for top tier products
                  </Text>
                  <Text
                    size="5"
                    weight="medium"
                    className="feature-left-box-subtitle"
                    align="center"
                    mt="16px"
                    style={{ maxWidth: "504px" }}
                  >
                    From word level bounding boxes to<br></br>custom VLM prompts
                    - we've got you covered.
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
                      title="Intelligent Chunking"
                      description="Set your own chunk size, and let us handle the logic to maintain semantic integrity"
                      onMouseEnter={() => handleLottieHover(chunkingLottieRef)}
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
                      onMouseEnter={() => handleLottieHover(bargraphLottieRef)}
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
                      <Text size="4" weight="medium" style={{ color: "white" }}>
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
