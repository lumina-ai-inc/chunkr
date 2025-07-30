import { Flex, Text } from "@radix-ui/themes";
import "./Dashboard.css";
import BetterButton from "../../components/BetterButton/BetterButton";
import TaskTable from "../../components/TaskTable/TaskTable";
import { useAuth } from "react-oidc-context";
import useUser from "../../hooks/useUser";
import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useTaskQuery } from "../../hooks/useTaskQuery";
import { Suspense, lazy } from "react";
import Loader from "../Loader/Loader";
import Usage from "../../components/Usage/Usage";
import { useLocation, useNavigate } from "react-router-dom";
import UploadDialog from "../../components/Upload/UploadDialog";
import ApiKeyDialog from "../../components/ApiDialog/ApiKeyDialog";
import { toast } from "react-hot-toast";
import { getBillingPortalSession } from "../../services/stripeService";
import ReactJson from "react-json-view";
import { useQueryClient } from "react-query";

// Lazy load components
const Viewer = lazy(() => import("../../components/Viewer/Viewer"));
const ExcelViewer = lazy(
  () => import("../../components/ExcelViewer/ExcelViewer")
);

const DOCS_URL = import.meta.env.VITE_DOCS_URL;

export default function Dashboard() {
  const auth = useAuth();
  const user = useUser();
  const navigate = useNavigate();
  const [selectedNav, setSelectedNav] = useState("Tasks");
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);
  const [isNavOpen, setIsNavOpen] = useState(true);
  const [showApiKey, setShowApiKey] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [showDownloadOptions, setShowDownloadOptions] = useState(false);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const configRef = useRef<HTMLDivElement>(null);
  const downloadRef = useRef<HTMLDivElement>(null);
  const [excelViewMode, setExcelViewMode] = useState<"ss" | "pdf">("ss");

  const location = useLocation();
  const searchParams = useMemo(
    () => new URLSearchParams(location.search),
    [location.search]
  );
  const taskId = searchParams.get("taskId");

  const { data: taskResponse, isLoading } = useTaskQuery(taskId || "");
  const queryClient = useQueryClient();
  const isExcelFile =
    taskResponse?.output?.file_name?.toLowerCase().endsWith(".xls") ||
    taskResponse?.output?.file_name?.toLowerCase().endsWith(".xlsx");

  const hasSsRanges = useMemo(() => {
    return (
      taskResponse?.output?.chunks?.some((chunk) =>
        chunk.segments?.some((segment) => segment.ss_range)
      ) ?? false
    );
  }, [taskResponse]);

  const showExcelToggle = isExcelFile && hasSsRanges;
  useEffect(() => {
    if (isExcelFile && !hasSsRanges && excelViewMode === "ss") {
      setExcelViewMode("pdf");
    }
  }, [isExcelFile, hasSsRanges, excelViewMode]);

  useEffect(() => {
    if (showExcelToggle) {
      setExcelViewMode("ss");
    }
  }, [showExcelToggle]);

  useEffect(() => {
    if (!searchParams.has("view")) {
      const params = new URLSearchParams(searchParams);
      params.set("view", "tasks");
      navigate({
        pathname: "/dashboard",
        search: params.toString(),
      });
    }
  }, [navigate, searchParams]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        profileRef.current &&
        !profileRef.current.contains(event.target as Node)
      ) {
        setIsProfileMenuOpen(false);
      }
      if (
        configRef.current &&
        !configRef.current.contains(event.target as Node)
      ) {
        setShowConfig(false);
      }
      if (
        downloadRef.current &&
        !downloadRef.current.contains(event.target as Node)
      ) {
        setShowDownloadOptions(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  useEffect(() => {
    if (location.pathname === "/dashboard") {
      setSelectedNav("Tasks");
    }
  }, [location.pathname]);

  useEffect(() => {
    if (location.state?.selectedNav) {
      setSelectedNav(location.state.selectedNav);
      // Clear the state after using it
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  const handleDocsNav = useCallback(() => {
    window.open(DOCS_URL, "_blank");
  }, []);

  const handleNavigation = useCallback(
    (item: string) => {
      // Handle special navigation items
      if (item === "API Key") {
        setShowApiKey(true);
        return;
      }

      if (item === "Docs") {
        handleDocsNav();
        return;
      }

      const params = new URLSearchParams();
      const currentParams = new URLSearchParams(searchParams);

      // Set view first (either "tasks" or "usage")
      params.set("view", item.toLowerCase());

      // Always preserve all view-specific parameters EXCEPT taskId when clicking on Tasks
      // For Tasks view
      const tablePageIndex = currentParams.get("tablePageIndex");
      const tablePageSize = currentParams.get("tablePageSize");
      // Only preserve taskId if we're not clicking on Tasks nav item
      const taskId = currentParams.get("taskId");
      if (taskId && item !== "Tasks") {
        params.set("taskId", taskId);
      }

      if (tablePageIndex) params.set("tablePageIndex", tablePageIndex);
      if (tablePageSize) params.set("tablePageSize", tablePageSize);

      // For Usage view
      const timeRange = currentParams.get("timeRange");
      if (timeRange) params.set("timeRange", timeRange);
      else if (item === "Usage" && !timeRange) {
        params.set("timeRange", "week"); // Set default only if switching to Usage and no timeRange exists
      }

      navigate({
        pathname: "/dashboard",
        search: params.toString(),
      });
      setSelectedNav(item);
    },
    [searchParams, navigate, handleDocsNav]
  );

  // Update initial selected nav based on URL params
  useEffect(() => {
    const view = searchParams.get("view");
    if (view === "usage") {
      setSelectedNav("Usage");
    } else if (view === "tasks") {
      setSelectedNav("Tasks");
    }
  }, [searchParams]);

  const handleHeaderNavigation = useCallback(() => {
    const params = new URLSearchParams();
    const currentParams = new URLSearchParams(searchParams);

    // Set view first
    params.set("view", "tasks");

    // Preserve all view-specific parameters
    for (const [key, value] of currentParams.entries()) {
      if (key !== "view" && key !== "taskId") {
        // Don't preserve taskId when clicking header
        params.set(key, value);
      }
    }

    navigate({
      pathname: "/dashboard",
      search: params.toString(),
    });
    setSelectedNav("Tasks");
  }, [searchParams, navigate]);

  const navIcons = {
    Tasks: (
      <g>
        <path
          d="M12.75 7.5H21.25"
          stroke="#FFF"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M12.75 16.5H21.25"
          stroke="#FFF"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M8.25 4.75H2.75V10.25H8.25V4.75Z"
          stroke="#FFF"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M8.25 13.75H2.75V19.25H8.25V13.75Z"
          stroke="#FFF"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
    ),
    Usage: (
      <g>
        <svg
          width="20"
          height="20"
          viewBox="0 0 22 22"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <g clip-path="url(#clip0_113_1401)">
            <path
              d="M5.25 20.25H6.75C7.30228 20.25 7.75 19.8023 7.75 19.25L7.75 13.75C7.75 13.1977 7.30228 12.75 6.75 12.75H5.25C4.69772 12.75 4.25 13.1977 4.25 13.75L4.25 19.25C4.25 19.8023 4.69772 20.25 5.25 20.25Z"
              stroke="#FFF"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M18.25 20.25H19.75C20.3023 20.25 20.75 19.8023 20.75 19.25V9.75C20.75 9.19772 20.3023 8.75 19.75 8.75H18.25C17.6977 8.75 17.25 9.19771 17.25 9.75V19.25C17.25 19.8023 17.6977 20.25 18.25 20.25Z"
              stroke="#FFF"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M11.75 20.25H13.25C13.8023 20.25 14.25 19.8023 14.25 19.25L14.25 5.75C14.25 5.19771 13.8023 4.75 13.25 4.75H11.75C11.1977 4.75 10.75 5.19771 10.75 5.75L10.75 19.25C10.75 19.8023 11.1977 20.25 11.75 20.25Z"
              stroke="#FFF"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </g>
          <defs>
            <clipPath id="clip0_113_1401">
              <rect
                width="24"
                height="24"
                fill="white"
                transform="translate(0.5)"
              />
            </clipPath>
          </defs>
        </svg>
      </g>
    ),
    "API Key": (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
      >
        <rect width="24" height="24" fill="white" fillOpacity="0.01"></rect>
        <path
          fillRule="evenodd"
          clipRule="evenodd"
          d="M15.9428 4.29713C16.1069 3.88689 15.9073 3.42132 15.4971 3.25723C15.0869 3.09315 14.6213 3.29267 14.4572 3.70291L8.05722 19.7029C7.89312 20.1131 8.09266 20.5787 8.50288 20.7427C8.91312 20.9069 9.37869 20.7074 9.54278 20.2971L15.9428 4.29713ZM6.16568 8.23433C6.47811 8.54675 6.47811 9.05327 6.16568 9.36569L3.53138 12L6.16568 14.6343C6.47811 14.9467 6.47811 15.4533 6.16568 15.7657C5.85326 16.0781 5.34674 16.0781 5.03432 15.7657L1.83432 12.5657C1.52189 12.2533 1.52189 11.7467 1.83432 11.4343L5.03432 8.23433C5.34674 7.92191 5.85326 7.92191 6.16568 8.23433ZM17.8342 8.23433C18.1467 7.92191 18.6533 7.92191 18.9658 8.23433L22.1658 11.4343C22.4781 11.7467 22.4781 12.2533 22.1658 12.5657L18.9658 15.7657C18.6533 16.0781 18.1467 16.0781 17.8342 15.7657C17.5219 15.4533 17.5219 14.9467 17.8342 14.6343L20.4686 12L17.8342 9.36569C17.5219 9.05327 17.5219 8.54675 17.8342 8.23433Z"
          fill="hsla(0, 0%, 100%, 0.9)"
        ></path>
      </svg>
    ),
    Docs: (
      <g>
        <path
          d="M9.25 6.75H15.75"
          stroke="#FFF"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M8 15.75H19.75V21.25H8C6.48 21.25 5.25 20.02 5.25 18.5C5.25 16.98 6.48 15.75 8 15.75Z"
          stroke="#FFF"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M5.25 18.5V5.75C5.25 4.09315 6.59315 2.75 8.25 2.75H18.75C19.3023 2.75 19.75 3.19772 19.75 3.75V16"
          stroke="#FFF"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
    ),
  };

  const userDisplayName =
    user?.data?.first_name && user?.data?.last_name
      ? `${user.data.first_name} ${user.data.last_name}`
      : user?.data?.email || "User";

  // Format the tier display name
  const rawTier = user?.data?.tier || "Free";
  const displayTier = rawTier === "SelfHosted" ? "Self Hosted" : rawTier;

  const showProfilePopup = user?.data && isProfileMenuOpen;

  const content = useMemo(() => {
    const view = searchParams.get("view");

    switch (view) {
      case "usage":
        return {
          title: "Usage",
          component: (
            <Usage key="usage-view" customerId={user.data?.customer_id || ""} />
          ),
        };
      case "tasks":
      default:
        if (taskId) {
          return {
            title: "Tasks",
            component: (
              <Suspense fallback={<Loader />}>
                {isLoading ? (
                  <Loader />
                ) : taskResponse?.output?.pdf_url ? (
                  <Viewer
                    key={`viewer-${taskId}`}
                    task={taskResponse!}
                    rightPanelContent={
                      isExcelFile && excelViewMode === "ss" ? (
                        <ExcelViewer taskResponse={taskResponse!} />
                      ) : undefined
                    }
                  />
                ) : null}
              </Suspense>
            ),
          };
        }
        return {
          title: "Tasks",
          component: (
            <TaskTable key={`task-table-${searchParams.toString()}`} />
          ),
        };
    }
  }, [
    searchParams,
    taskId,
    taskResponse,
    isLoading,
    user,
    excelViewMode,
    isExcelFile,
  ]);

  const toggleNav = () => {
    setIsNavOpen(!isNavOpen);
  };

  const handleContactClick = (type: "email" | "calendar") => {
    if (type === "email") {
      navigator.clipboard.writeText("mehul@chunkr.ai");
      toast.success("Email copied to clipboard!");
    } else {
      window.open("https://cal.com/mehulc/30min", "_blank");
    }
  };

  const handleBillingNavigation = async () => {
    if (user?.data?.tier === "Free") {
      navigate("/");
      setTimeout(() => {
        window.location.hash = "pricing";
      }, 100);
      return;
    }

    try {
      const { url } = await getBillingPortalSession(
        auth.user?.access_token || "",
        user?.data?.customer_id || ""
      );
      window.location.href = url;
    } catch (error) {
      console.error("Error redirecting to billing portal:", error);
    }
  };

  const handleDownloadOriginalFile = useCallback(() => {
    if (taskResponse?.configuration?.input_file_url) {
      fetch(taskResponse.configuration.input_file_url)
        .then((response) => response.blob())
        .then((blob) => {
          const blobUrl = window.URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = blobUrl;

          const originalFilename =
            taskResponse?.output?.file_name || "document.pdf";
          const extension = originalFilename.split(".").pop() || "pdf";
          const baseFilename = originalFilename.replace(`.${extension}`, "");
          link.download = `${baseFilename}_chunkr.${extension}`;

          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          window.URL.revokeObjectURL(blobUrl);
        })
        .catch((error) => {
          console.error("Error downloading file:", error);
        });
    }
  }, [taskResponse]);

  const handleDownloadPDF = useCallback(() => {
    if (taskResponse?.output?.pdf_url) {
      window.open(taskResponse.output.pdf_url, "_blank");
    }
  }, [taskResponse]);

  const handleDownloadJSON = useCallback(() => {
    if (taskResponse) {
      const jsonString = JSON.stringify(taskResponse, null, 2);
      const blob = new Blob([jsonString], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;

      const originalFilename =
        taskResponse?.output?.file_name || "document.pdf";
      const baseFilename = originalFilename.replace(/\.[^/.]+$/, "");
      a.download = `${baseFilename}_chunkr_json.json`;

      a.click();
      URL.revokeObjectURL(url);
    }
  }, [taskResponse]);

  const handleDownloadHTML = useCallback(() => {
    if (taskResponse?.output) {
      const combinedHtml = taskResponse.output.chunks
        .map((chunk) =>
          chunk.segments
            .map((segment) => {
              if (
                segment.segment_type === "Table" &&
                segment.html?.startsWith("<span class=")
              ) {
                return `<br><img src="${segment.image}" />`;
              }
              return segment.html || "";
            })
            .filter(Boolean)
            .join("")
        )
        .join("<hr>");

      const sanitizedHtml = combinedHtml;
      const blob = new Blob([sanitizedHtml], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;

      const originalFilename = taskResponse.output.file_name || "document.pdf";
      const baseFilename = originalFilename.replace(/\.[^/.]+$/, "");
      a.download = `${baseFilename}_chunkr_html.html`;

      a.click();
      URL.revokeObjectURL(url);
    }
  }, [taskResponse]);

  const handleDownloadMarkdown = useCallback(() => {
    if (taskResponse?.output) {
      const combinedMarkdown = taskResponse.output.chunks
        .map((chunk) =>
          chunk.segments
            .map((segment) => {
              const textContent = segment.content || "";
              if (
                segment.segment_type === "Table" &&
                segment.html?.startsWith("<span class=")
              ) {
                return `![Image](${segment.image})`;
              }
              return segment.markdown ? segment.markdown : textContent;
            })
            .filter(Boolean)
            .join("\n\n")
        )
        .join("\n\n---\n\n");

      const blob = new Blob([combinedMarkdown], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;

      const originalFilename = taskResponse.output.file_name || "document.pdf";
      const baseFilename = originalFilename.replace(/\.[^/.]+$/, "");
      a.download = `${baseFilename}_chunkr_markdown.md`;

      a.click();
      URL.revokeObjectURL(url);
    }
  }, [taskResponse]);

  // Truncate filename for display if it exceeds maximum length
  const MAX_FILENAME_LENGTH = 20;
  const filenameToDisplay = taskResponse?.output?.file_name || taskId || "";
  const truncatedFilename =
    filenameToDisplay.length > MAX_FILENAME_LENGTH
      ? `${filenameToDisplay.substring(0, MAX_FILENAME_LENGTH)}...`
      : filenameToDisplay;

  return (
    <Flex
      direction="row"
      width="100%"
      height="100%"
      minWidth="1156px"
      minHeight="700px"
    >
      <Flex
        className={`dashboard-nav-container ${isNavOpen ? "" : "closed"}`}
        align="start"
        direction="column"
      >
        <Flex className="dashboard-nav-header">
          <Flex
            gap="8px"
            align="center"
            justify="center"
            onClick={() => navigate("/")}
            style={{ cursor: "pointer" }}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="28"
              height="28"
              viewBox="0 0 22 22"
              fill="none"
            >
              <path
                d="M5.39 8.965C5.88975 9.16504 6.43722 9.21401 6.96454 9.10584C7.49186 8.99767 7.97584 8.73711 8.35648 8.35648C8.73711 7.97584 8.99767 7.49186 9.10584 6.96454C9.21401 6.43722 9.16504 5.88975 8.965 5.39C9.54645 5.23345 10.0604 4.89038 10.4281 4.41346C10.7957 3.93655 10.9966 3.35215 11 2.75C12.6317 2.75 14.2267 3.23385 15.5835 4.14038C16.9402 5.0469 17.9976 6.33537 18.622 7.84286C19.2464 9.35035 19.4098 11.0092 19.0915 12.6095C18.7732 14.2098 17.9874 15.6798 16.8336 16.8336C15.6798 17.9874 14.2098 18.7732 12.6095 19.0915C11.0092 19.4098 9.35035 19.2464 7.84286 18.622C6.33537 17.9976 5.0469 16.9402 4.14038 15.5835C3.23385 14.2267 2.75 12.6317 2.75 11C3.35215 10.9966 3.93655 10.7957 4.41346 10.4281C4.89038 10.0604 5.23345 9.54645 5.39 8.965Z"
                stroke="url(#paint0_linear_293_747)"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <defs>
                <linearGradient
                  id="paint0_linear_293_747"
                  x1="11"
                  y1="2.75"
                  x2="11"
                  y2="19.25"
                  gradientUnits="userSpaceOnUse"
                >
                  <stop stopColor="white" />
                  <stop offset="1" stopColor="#DCE4DD" />
                </linearGradient>
              </defs>
            </svg>
            <Text size="5" weight="bold" mb="2px" style={{ color: "#FFF" }}>
              chunkr
            </Text>
          </Flex>
          <Flex className="dashboard-toggle" onClick={toggleNav}>
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M21.97 15V9C21.97 4 19.97 2 14.97 2H8.96997C3.96997 2 1.96997 4 1.96997 9V15C1.96997 20 3.96997 22 8.96997 22H14.97C19.97 22 21.97 20 21.97 15Z"
                stroke="#FFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M7.96997 2V22"
                stroke="#FFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M14.97 9.43994L12.41 11.9999L14.97 14.5599"
                stroke="#FFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </Flex>
        </Flex>
        <Flex
          className="dashboard-nav-body"
          direction="column"
          justify="between"
        >
          <Flex direction="column">
            <Flex className="dashboard-nav-items" direction="column">
              {["Tasks", "Usage", "API Key", "Docs"].map((item) => (
                <Flex
                  key={item}
                  className={`dashboard-nav-item ${
                    selectedNav === item ? "selected" : ""
                  }`}
                  onClick={() => handleNavigation(item)}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 22 22"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    {navIcons[item as keyof typeof navIcons]}
                  </svg>
                  <Text
                    size="3"
                    weight="medium"
                    style={{
                      color: selectedNav === item ? "rgb(2, 5, 6)" : "#FFF",
                    }}
                  >
                    {item}
                  </Text>
                </Flex>
              ))}
            </Flex>
          </Flex>

          <Flex className="profile-section" direction="column">
            <Flex
              ref={profileRef}
              className="profile-info"
              onClick={() => setIsProfileMenuOpen(!isProfileMenuOpen)}
              style={{ position: "relative" }}
            >
              <Flex gap="12px" align="center">
                <Flex direction="column" gap="4px">
                  <Text size="3" weight="bold" style={{ color: "#FFF" }}>
                    {userDisplayName}
                  </Text>
                  <Text size="1" style={{ color: "rgba(255,255,255,0.8)" }}>
                    {/* Use the formatted tier name */}
                    {displayTier}
                  </Text>
                </Flex>
              </Flex>
              {showProfilePopup && (
                <Flex className="profile-popup">
                  <Flex className="profile-menu" direction="column">
                    <Flex
                      className="profile-menu-item"
                      onClick={handleBillingNavigation}
                      style={{ cursor: "pointer" }}
                    >
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M12 21.25C17.1086 21.25 21.25 17.1086 21.25 12C21.25 6.89137 17.1086 2.75 12 2.75C6.89137 2.75 2.75 6.89137 2.75 12C2.75 17.1086 6.89137 21.25 12 21.25Z"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeMiterlimit="10"
                        />
                        <path
                          d="M9.88012 14.36C9.88012 15.53 10.8301 16.25 12.0001 16.25C13.1701 16.25 14.1201 15.53 14.1201 14.36C14.1201 13.19 13.3501 12.75 11.5301 11.66C10.6701 11.15 9.87012 10.82 9.87012 9.64C9.87012 8.46 10.8201 7.75 11.9901 7.75C13.1601 7.75 14.1101 8.7 14.1101 9.87"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                        {user?.data?.tier === "Free"
                          ? "Upgrade Plan"
                          : "Manage Billing"}
                      </Text>
                    </Flex>
                    <Flex
                      className="profile-menu-item"
                      onClick={() => handleContactClick("email")}
                      style={{ cursor: "pointer" }}
                    >
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <g clip-path="url(#clip0_305_31838)">
                          <path
                            d="M20.25 4.75H3.75C3.19772 4.75 2.75 5.19771 2.75 5.75V18.25C2.75 18.8023 3.19772 19.25 3.75 19.25H20.25C20.8023 19.25 21.25 18.8023 21.25 18.25V5.75C21.25 5.19772 20.8023 4.75 20.25 4.75Z"
                            stroke="#FFF"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M21.25 7.25L13.9625 13.5527C12.8356 14.5273 11.1644 14.5273 10.0375 13.5527L2.75 7.25"
                            stroke="#FFF"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </g>
                        <defs>
                          <clipPath id="clip0_305_31838">
                            <rect width="24" height="24" fill="white" />
                          </clipPath>
                        </defs>
                      </svg>
                      <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                        Email Us
                      </Text>
                    </Flex>

                    <Flex
                      className="profile-menu-item"
                      onClick={() => handleContactClick("calendar")}
                      style={{ cursor: "pointer" }}
                    >
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <g clip-path="url(#clip0_305_31814)">
                          <path
                            d="M19.2499 14.93V18.23C19.2499 19.46 18.1599 20.4 16.9399 20.23C9.59991 19.21 3.78991 13.4 2.76991 6.06C2.59991 4.84 3.53991 3.75 4.76991 3.75H8.06991C8.55991 3.75 8.97991 4.1 9.05991 4.58L9.44991 6.77C9.58991 7.54 9.26991 8.32 8.62991 8.77L7.73991 9.4C9.16991 11.81 11.1999 13.82 13.6199 15.24L14.2299 14.37C14.6799 13.73 15.4599 13.41 16.2299 13.55L18.4199 13.94C18.8999 14.03 19.2499 14.44 19.2499 14.93V14.93Z"
                            stroke="#FFF"
                            strokeWidth="1"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M15.75 3.75H20.25V8.25"
                            stroke="#FFF"
                            strokeWidth="1"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M20.22 3.77979L14.75 9.24979"
                            stroke="#FFF"
                            strokeWidth="1"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </g>
                        <defs>
                          <clipPath id="clip0_305_31814">
                            <rect width="24" height="24" fill="white" />
                          </clipPath>
                        </defs>
                      </svg>
                      <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                        Book a Call
                      </Text>
                    </Flex>
                    <Flex
                      className="profile-menu-item"
                      onClick={() =>
                        window.open("https://discord.gg/XzKWFByKzW", "_blank")
                      }
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        fill="none"
                        viewBox="0 0 430 430"
                      >
                        <g strokeWidth="12">
                          <path
                            stroke="#FFFFFF"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M312.601 305.475c-26.273 14.715-58.717 23.408-93.825 23.408s-67.552-8.693-93.825-23.408"
                          />
                          <path
                            stroke="#FFFFFF"
                            strokeLinejoin="round"
                            d="M158.352 72.148c3.725 6.192 6.909 13.393 9.531 21.46 15.975-4.123 33.091-6.358 50.893-6.358 15.228 0 29.955 1.636 43.895 4.69 2.513-7.394 5.511-14.03 8.977-19.792 25.747 3.086 49.437 10.893 69.495 22.22 20.228 24.718 37.723 59.3 48.459 99.37 11.642 43.446 13.177 85.277 6.259 118.097-21.508 20.403-52.494 36.375-88.916 45.331-7.134-12.146-13.866-25.549-20.013-39.981-20.648 7.5-43.759 11.698-68.156 11.698-26.99 0-52.406-5.137-74.653-14.199-6.428 15.387-13.524 29.638-21.068 42.482-36.422-8.956-67.408-24.928-88.916-45.331-6.918-32.82-5.383-74.651 6.259-118.097 10.736-40.07 28.23-74.652 48.46-99.37 20.057-11.327 43.747-19.134 69.494-22.22Z"
                          />
                          <path
                            stroke="#FFFFFF"
                            d="M310 230c0 16.569-11.193 30-25 30s-25-13.431-25-30 11.193-30 25-30 25 13.431 25 30Zm-140 0c0 16.569-11.193 30-25 30s-25-13.431-25-30 11.193-30 25-30 25 13.431 25 30Z"
                          />
                        </g>
                      </svg>
                      <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                        Join the Discord
                      </Text>
                    </Flex>
                    <Flex
                      className="profile-menu-item"
                      onClick={() => auth.signoutRedirect()}
                    >
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <g clip-path="url(#clip0_305_27927)">
                          <path
                            d="M16 16.25L20.25 12L16 7.75"
                            stroke="#FFFFFF"
                            stroke-width="1.5"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                          />
                          <path
                            d="M20.25 12H8.75"
                            stroke="#FFFFFF"
                            stroke-width="1.5"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                          />
                          <path
                            d="M13.25 20.25H5.75C4.65 20.25 3.75 19.35 3.75 18.25V5.75C3.75 4.65 4.65 3.75 5.75 3.75H13.25"
                            stroke="#FFFFFF"
                            stroke-width="1.5"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                          />
                        </g>
                        <defs>
                          <clipPath id="clip0_305_27927">
                            <rect width="24" height="24" fill="white" />
                          </clipPath>
                        </defs>
                      </svg>
                      <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                        Logout
                      </Text>
                    </Flex>
                  </Flex>
                </Flex>
              )}
            </Flex>
          </Flex>
        </Flex>
      </Flex>
      <Flex direction="column" className="main-container">
        <Flex className="main-header">
          <Flex gap="8px" align="center">
            <div className="main-header-toggle" onClick={toggleNav}>
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M21.97 15V9C21.97 4 19.97 2 14.97 2H8.96997C3.96997 2 1.96997 4 1.96997 9V15C1.96997 20 3.96997 22 8.96997 22H14.97C19.97 22 21.97 20 21.97 15Z"
                  stroke="#FFF"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M14.97 2V22"
                  stroke="#FFF"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M7.96997 9.43994L10.53 11.9999L7.96997 14.5599"
                  stroke="#FFF"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <Flex
              onClick={handleHeaderNavigation}
              style={{ cursor: "pointer" }}
            >
              <Text size="5" weight="bold" className="main-header-text">
                {content.title}
              </Text>
            </Flex>
            {searchParams.get("view") === "tasks" &&
              taskId &&
              taskResponse?.output?.file_name && (
                <>
                  <svg
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M9 6L15 12L9 18"
                      stroke="#FFFFFF"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <Flex className="header-task-tag">
                    <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                      {truncatedFilename}
                    </Text>
                  </Flex>
                </>
              )}
          </Flex>
          <Flex gap="24px">
            {searchParams.get("view") === "tasks" && taskId && taskResponse && (
              <>
                {showExcelToggle && (
                  <div className="excel-view-toggle">
                    <button
                      className={`toggle-button ${
                        excelViewMode === "ss" ? "active" : ""
                      }`}
                      onClick={() => setExcelViewMode("ss")}
                    >
                      <Text size="2" weight="medium">
                        Spreadsheet
                      </Text>
                    </button>
                    <button
                      className={`toggle-button ${
                        excelViewMode === "pdf" ? "active" : ""
                      }`}
                      onClick={() => setExcelViewMode("pdf")}
                    >
                      <Text size="2" weight="medium">
                        PDF
                      </Text>
                    </button>
                  </div>
                )}

                <div
                  className="download-dropdown-container"
                  style={{ position: "relative" }}
                  ref={downloadRef}
                >
                  <BetterButton
                    onClick={() => setShowDownloadOptions(!showDownloadOptions)}
                    active={showDownloadOptions}
                  >
                    <svg
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <g clip-path="url(#clip0_304_23621)">
                        <path
                          d="M19.25 9.25V20.25C19.25 20.8 18.8 21.25 18.25 21.25H5.75C5.2 21.25 4.75 20.8 4.75 20.25V3.75C4.75 3.2 5.2 2.75 5.75 2.75H12.75"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M12.75 9.25H19.25L12.75 2.75V9.25Z"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M14.5 15.75L12 18.25L9.5 15.75"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M12 18V12.75"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </g>
                      <defs>
                        <clipPath id="clip0_304_23621">
                          <rect width="24" height="24" fill="white" />
                        </clipPath>
                      </defs>
                    </svg>
                    <Text
                      size="2"
                      weight="medium"
                      style={{ color: "rgba(255, 255, 255, 0.95)" }}
                    >
                      Download
                    </Text>
                  </BetterButton>
                  {showDownloadOptions && (
                    <div
                      className="download-options-popup"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <Flex
                        className="download-options-menu"
                        direction="column"
                      >
                        <div
                          className="download-options-menu-item"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDownloadOriginalFile();
                            setShowDownloadOptions(false);
                          }}
                        >
                          <Text
                            size="2"
                            weight="medium"
                            style={{ color: "#FFF" }}
                          >
                            Original File
                          </Text>
                        </div>
                        <div
                          className="download-options-menu-item"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDownloadPDF();
                            setShowDownloadOptions(false);
                          }}
                        >
                          <Text
                            size="2"
                            weight="medium"
                            style={{ color: "#FFF" }}
                          >
                            PDF
                          </Text>
                        </div>
                        <div
                          className="download-options-menu-item"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDownloadJSON();
                            setShowDownloadOptions(false);
                          }}
                        >
                          <Text
                            size="2"
                            weight="medium"
                            style={{ color: "#FFF" }}
                          >
                            JSON
                          </Text>
                        </div>
                        <div
                          className="download-options-menu-item"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDownloadHTML();
                            setShowDownloadOptions(false);
                          }}
                        >
                          <Text
                            size="2"
                            weight="medium"
                            style={{ color: "#FFF" }}
                          >
                            HTML
                          </Text>
                        </div>
                        <div
                          className="download-options-menu-item"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDownloadMarkdown();
                            setShowDownloadOptions(false);
                          }}
                        >
                          <Text
                            size="2"
                            weight="medium"
                            style={{ color: "#FFF" }}
                          >
                            Markdown
                          </Text>
                        </div>
                      </Flex>
                    </div>
                  )}
                </div>

                <div
                  ref={configRef}
                  style={{ position: "relative", zIndex: 1000 }}
                >
                  <BetterButton
                    onClick={() => setShowConfig(!showConfig)}
                    active={showConfig}
                  >
                    <svg
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <g clip-path="url(#clip0_304_23691)">
                        <path
                          d="M20.6519 14.43L19.0419 13.5C19.1519 13.02 19.2019 12.51 19.2019 12C19.2019 11.49 19.1519 10.98 19.0419 10.5L20.6519 9.57C20.8919 9.44 20.9719 9.13 20.8419 8.89L19.0919 5.86C18.9519 5.62 18.6419 5.54 18.4019 5.68L16.7819 6.62C16.0419 5.94 15.1719 5.42 14.2019 5.12V3.25C14.2019 2.97 13.9819 2.75 13.7019 2.75H10.2019C9.92193 2.75 9.70193 2.97 9.70193 3.25V5.12C8.73193 5.42 7.86193 5.94 7.12193 6.62L5.50193 5.68C5.26193 5.54 4.95193 5.62 4.81193 5.86L3.06193 8.89C2.93193 9.13 3.01193 9.44 3.25193 9.57L4.86193 10.5C4.75193 10.98 4.70193 11.49 4.70193 12C4.70193 12.51 4.75193 13.02 4.86193 13.5L3.25193 14.43C3.01193 14.56 2.93193 14.87 3.06193 15.11L4.81193 18.14C4.95193 18.38 5.26193 18.46 5.50193 18.32L7.12193 17.38C7.86193 18.06 8.73193 18.58 9.70193 18.88V20.75C9.70193 21.03 9.92193 21.25 10.2019 21.25H13.7019C13.9819 21.25 14.2019 21.03 14.2019 20.75V18.88C15.1719 18.58 16.0419 18.06 16.7819 17.38L18.4019 18.32C18.6419 18.46 18.9519 18.38 19.0919 18.14L20.8419 15.11C20.9719 14.87 20.8919 14.56 20.6519 14.43ZM15.0919 12.84C15.0119 13.11 14.9119 13.38 14.7619 13.62C14.4819 14.12 14.0719 14.53 13.5719 14.81C13.3319 14.96 13.0619 15.06 12.7919 15.14C12.5219 15.21 12.2419 15.25 11.9519 15.25C11.6619 15.25 11.3819 15.21 11.1119 15.14C10.8419 15.06 10.5719 14.96 10.3319 14.81C9.83193 14.53 9.42193 14.12 9.14193 13.62C8.99193 13.38 8.89193 13.11 8.81193 12.84C8.74193 12.57 8.70193 12.29 8.70193 12C8.70193 11.71 8.74193 11.43 8.81193 11.16C8.89193 10.89 8.99193 10.62 9.14193 10.38C9.42193 9.88 9.83193 9.47 10.3319 9.19C10.5719 9.04 10.8419 8.94 11.1119 8.86C11.3819 8.79 11.6619 8.75 11.9519 8.75C12.2419 8.79 12.5219 8.86 12.7919 8.94C13.0619 9.04 13.5719 9.19C14.0719 9.47 14.4819 9.88 14.7619 10.38C14.9119 10.62 15.0119 10.89 15.0919 11.16C15.1619 11.43 15.2019 11.71 15.2019 12C15.2019 12.29 15.1619 12.57 15.0919 12.84Z"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </g>
                      <defs>
                        <clipPath id="clip0_304_23691">
                          <rect width="24" height="24" fill="white" />
                        </clipPath>
                      </defs>
                    </svg>
                    <Text
                      size="2"
                      weight="medium"
                      style={{ color: "rgba(255, 255, 255, 0.95)" }}
                    >
                      Configuration
                    </Text>
                    {showConfig && (
                      <div
                        className="config-popup"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <Flex
                          className="config-popup-content"
                          direction="column"
                          height="100%"
                          width="100%"
                          overflow="auto"
                        >
                          <ReactJson
                            src={{
                              ...taskResponse.configuration,
                              input_file_url: taskResponse.configuration
                                .input_file_url
                                ? taskResponse.configuration.input_file_url
                                    .length > 10
                                  ? taskResponse.configuration.input_file_url.substring(
                                      0,
                                      10
                                    ) + "..."
                                  : taskResponse.configuration.input_file_url
                                : null,
                            }}
                            theme="monokai"
                            displayDataTypes={false}
                            enableClipboard={false}
                            style={{
                              backgroundColor: "transparent",
                              padding: "12px",
                              fontSize: "12px",
                              textAlign: "justify",
                            }}
                            collapsed={1}
                            name={false}
                            displayObjectSize={false}
                          />
                        </Flex>
                      </div>
                    )}
                  </BetterButton>
                </div>

                <BetterButton
                  onClick={() => {
                    navigator.clipboard.writeText(taskId || "");
                    toast.success("Task ID copied to clipboard");
                  }}
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      fillRule="evenodd"
                      clipRule="evenodd"
                      d="M15 1.25H10.9436C9.10583 1.24998 7.65019 1.24997 6.51098 1.40314C5.33856 1.56076 4.38961 1.89288 3.64124 2.64124C2.89288 3.38961 2.56076 4.33856 2.40314 5.51098C2.24997 6.65019 2.24998 8.10582 2.25 9.94357V16C2.25 17.8722 3.62205 19.424 5.41551 19.7047C5.55348 20.4687 5.81753 21.1208 6.34835 21.6517C6.95027 22.2536 7.70814 22.5125 8.60825 22.6335C9.47522 22.75 10.5775 22.75 11.9451 22.75H15.0549C16.4225 22.75 17.5248 22.75 18.3918 22.6335C19.2919 22.5125 20.0497 22.2536 20.6517 21.6517C21.2536 21.0497 21.5125 20.2919 21.6335 19.3918C21.75 18.5248 21.75 17.4225 21.75 16.0549V10.9451C21.75 9.57754 21.75 8.47522 21.6335 7.60825C21.5125 6.70814 21.2536 5.95027 20.6517 5.34835C20.1208 4.81753 19.4687 4.55348 18.7047 4.41551C18.424 2.62205 16.8722 1.25 15 1.25ZM17.1293 4.27117C16.8265 3.38623 15.9876 2.75 15 2.75H11C9.09318 2.75 7.73851 2.75159 6.71085 2.88976C5.70476 3.02502 5.12511 3.27869 4.70190 3.7019C4.27869 4.12511 4.02502 4.70476 3.88976 5.71085C3.75159 6.73851 3.75 8.09318 3.75 10V16C3.75 16.9876 4.38624 17.8265 5.27117 18.1293C5.24998 17.5194 5.24999 16.8297 5.25 16.0549V10.9451C5.24998 9.57754 5.24996 8.47522 5.36652 7.60825C5.48754 6.70814 5.74643 5.95027 6.34835 5.34835C6.95027 4.74643 7.70814 4.48754 8.60825 4.36652C9.47522 4.24996 10.5775 4.24998 11.9451 4.25H15.0549C15.8297 4.24999 16.5194 4.24998 17.1293 4.27117ZM7.40901 6.40901C7.68577 6.13225 8.07435 5.9518 8.80812 5.85315C9.56347 5.75159 10.5646 5.75 12 5.75H15C16.4354 5.75 17.4365 5.75159 18.1919 5.85315C18.9257 5.9518 19.3142 6.13225 19.591 6.40901C19.8678 6.68577 20.0482 7.07435 20.1469 7.80812C20.2484 8.56347 20.25 9.56458 20.25 11V16C20.25 17.4354 20.2484 18.4365 20.1469 19.1919C20.0482 19.9257 19.8678 20.3142 19.591 20.591C19.3142 20.8678 18.9257 21.0482 18.1919 21.1469C17.4365 21.2484 16.4354 21.25 15 21.25H12C10.5646 21.25 9.56347 21.2484 8.80812 21.1469C8.07435 21.0482 7.68577 20.8678 7.40901 20.591C7.13225 20.3142 6.9518 19.9257 6.85315 19.1919C6.75159 18.4365 6.75 17.4354 6.75 16V11C6.75 9.56458 6.75159 8.56347 6.85315 7.80812C6.9518 7.07435 7.13225 6.68577 7.40901 6.40901Z"
                      fill="#FFFFFF"
                    ></path>
                  </svg>
                  <Text
                    size="2"
                    weight="medium"
                    style={{ color: "rgba(255, 255, 255, 0.95)" }}
                  >
                    Copy Task ID
                  </Text>
                </BetterButton>
              </>
            )}
            <BetterButton onClick={() => setShowUploadDialog(true)}>
              <svg
                width="18"
                height="18"
                viewBox="0 0 25 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <g clip-path="url(#clip0_113_1479)">
                  <path
                    d="M19.75 9.25V20.25C19.75 20.8 19.3 21.25 18.75 21.25H6.25C5.7 21.25 5.25 20.8 5.25 20.25V3.75C5.25 3.2 5.7 2.75 6.25 2.75H13.25"
                    stroke="#FFF"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>
                  <path
                    d="M13.25 9.25H19.75L13.25 2.75V9.25Z"
                    stroke="#FFF"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>
                  <path
                    d="M10 15.25L12.5 12.75L15 15.25"
                    stroke="#FFF"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>
                  <path
                    d="M12.5 13.75V18.25"
                    stroke="#FFF"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>
                </g>
                <defs>
                  <clipPath id="clip0_113_1479">
                    <rect
                      width="24"
                      height="24"
                      fill="white"
                      transform="translate(0.5)"
                    ></rect>
                  </clipPath>
                </defs>
              </svg>
              <Text
                size="2"
                weight="medium"
                style={{ color: "rgba(255, 255, 255, 0.95)" }}
              >
                Create Task
              </Text>
            </BetterButton>
          </Flex>
        </Flex>
        <Flex className="main-body">{auth && user && content.component}</Flex>
      </Flex>

      {/* Dialogs */}
      {user.data && (
        <ApiKeyDialog
          user={user.data}
          showApiKey={showApiKey}
          setShowApiKey={setShowApiKey}
        />
      )}
      <UploadDialog
        auth={auth}
        open={showUploadDialog}
        onOpenChange={setShowUploadDialog}
        onUploadComplete={async () => {
          setShowUploadDialog(false);
          queryClient.invalidateQueries(["tasks"]);
        }}
      />
    </Flex>
  );
}
