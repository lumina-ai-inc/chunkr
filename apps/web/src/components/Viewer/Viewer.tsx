import { useRef, useState, useCallback, useMemo, useEffect, memo } from "react";
import { Flex, Text } from "@radix-ui/themes";
import { SegmentChunk } from "../SegmentChunk/SegmentChunk";
import { PDF } from "../PDF/PDF";
import "./Viewer.css";
import Loader from "../../pages/Loader/Loader";
import { TaskResponse, Chunk } from "../../models/taskResponse.model";
import ReactJson from "react-json-view";
import BetterButton from "../BetterButton/BetterButton";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";

const MemoizedPDF = memo(PDF);

const CHUNK_LOAD_SIZE = 5; // Number of chunks to load at a time
const PAGE_CHUNK_SIZE = 5; // Number of pages to load at a time

export default function Viewer({ task }: { task: TaskResponse }) {
  const output = task.output;
  const inputFileUrl = task.configuration.input_file_url;
  const memoizedOutput = useMemo(() => output, [output]);
  const [showConfig, setShowConfig] = useState(false);
  const [leftPanelWidth, setLeftPanelWidth] = useState(0);
  const chunkRefs = useRef<(HTMLDivElement | null)[]>([]);

  const hideTimeoutRef = useRef<NodeJS.Timeout>();

  const [selectedView, setSelectedView] = useState<
    "html" | "markdown" | "json"
  >("html");

  const [activeSegment, setActiveSegment] = useState<{
    chunkId: string;
    segmentId: string;
  } | null>(null);

  const [loadedChunks, setLoadedChunks] = useState(CHUNK_LOAD_SIZE);
  const [loadedPages, setLoadedPages] = useState(PAGE_CHUNK_SIZE);
  const [numPages, setNumPages] = useState<number>();

  const scrollToSegment = useCallback(
    (chunkId: string, segmentId: string) => {
      // Find the chunk and segment
      const chunk = output?.chunks.find((c) => c.chunk_id === chunkId);
      const segment = chunk?.segments.find((s) => s.segment_id === segmentId);

      if (!chunk || !segment) return;

      const targetPage = segment.page_number;
      const chunkIndex =
        output?.chunks.findIndex((c) => c.chunk_id === chunkId) ?? -1;

      // First, ensure content is loaded
      const needsMorePages = targetPage && targetPage > loadedPages;
      const needsMoreChunks = chunkIndex >= loadedChunks;

      if (needsMorePages) {
        setLoadedPages(
          Math.ceil(targetPage / PAGE_CHUNK_SIZE) * PAGE_CHUNK_SIZE
        );
      }

      if (needsMoreChunks) {
        setLoadedChunks(
          Math.ceil((chunkIndex + 1) / CHUNK_LOAD_SIZE) * CHUNK_LOAD_SIZE
        );
      }

      // Wait for content to load before scrolling
      setTimeout(
        () => {
          setActiveSegment({ chunkId, segmentId });

          // Scroll PDF container to segment (keeping 30% for PDF)
          const pdfContainer = document.querySelector(".pdf-container");
          const targetSegmentElement = pdfContainer?.querySelector(
            `[data-chunk-id="${chunkId}"][data-segment-id="${segmentId}"]`
          );

          if (targetSegmentElement && pdfContainer) {
            const containerHeight = pdfContainer.clientHeight;
            const segmentRect = targetSegmentElement.getBoundingClientRect();
            const containerRect = pdfContainer.getBoundingClientRect();
            const relativeTop = segmentRect.top - containerRect.top;

            // Keep 30% for PDF view
            const targetPosition =
              pdfContainer.scrollTop + relativeTop - containerHeight * 0.3;

            pdfContainer.scrollTo({
              top: targetPosition,
              behavior: "smooth",
            });
          }

          // Scroll text content to the specific segment (20% from top)
          const textSegmentElement = document.querySelector(
            `.scrollable-content [data-chunk-id="${chunkId}"][data-segment-id="${segmentId}"]`
          );
          if (textSegmentElement) {
            const container = textSegmentElement.closest(".scrollable-content");
            if (container) {
              const containerHeight = container.clientHeight;
              const segmentRect = textSegmentElement.getBoundingClientRect();
              const containerRect = container.getBoundingClientRect();
              const relativeTop = segmentRect.top - containerRect.top;

              // Use 20% for text content
              const targetPosition =
                container.scrollTop + relativeTop - containerHeight * 0.2;

              container.scrollTo({
                top: targetPosition,
                behavior: "smooth",
              });
            }
          }
        },
        needsMorePages || needsMoreChunks ? 300 : 100
      );
    },
    [output?.chunks, loadedPages, loadedChunks]
  );

  // Update the handler for PDF segment clicks
  const handlePDFSegmentClick = useCallback(
    (chunkId: string, segmentId: string) => {
      const chunkIndex =
        output?.chunks.findIndex((c) => c.chunk_id === chunkId) ?? -1;

      // Ensure we load enough chunks
      if (chunkIndex >= loadedChunks) {
        setLoadedChunks(
          Math.ceil((chunkIndex + 1) / CHUNK_LOAD_SIZE) * CHUNK_LOAD_SIZE
        );

        // Wait for next render cycle when chunks are loaded
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            scrollToSegment(chunkId, segmentId);
          });
        });
      } else {
        scrollToSegment(chunkId, segmentId);
      }
    },
    [loadedChunks, scrollToSegment, output?.chunks]
  );

  const handleMouseEnter = () => {
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
    }
    setShowConfig(true);
  };

  const handleMouseLeave = () => {
    hideTimeoutRef.current = setTimeout(() => {
      setShowConfig(false);
    }, 150);
  };

  const handleDownloadOriginalFile = useCallback(() => {
    if (inputFileUrl) {
      fetch(inputFileUrl)
        .then((response) => response.blob())
        .then((blob) => {
          // Create blob URL
          const blobUrl = window.URL.createObjectURL(blob);

          // Create temporary anchor element
          const link = document.createElement("a");
          link.href = blobUrl;

          // Extract filename from URL - look for the last segment before query parameters
          // that ends with .pdf and decode it
          const filename = decodeURIComponent(
            inputFileUrl.split("?")[0].split("/").pop() || "document.pdf"
          );
          link.download = filename;

          // Trigger download
          document.body.appendChild(link);
          link.click();

          // Cleanup
          document.body.removeChild(link);
          window.URL.revokeObjectURL(blobUrl);
        })
        .catch((error) => {
          console.error("Error downloading PDF:", error);
        });
    }
    console.log("inputFileUrl", inputFileUrl);
  }, [inputFileUrl]);

  const handleDownloadPDF = useCallback(() => {
    if (task.output?.pdf_url) {
      const pdfURL = task.output.pdf_url;
      if (pdfURL) {
        window.open(pdfURL, "_blank");
      }
    }
  }, [task]);

  const handleDownloadJSON = useCallback(() => {
    if (task) {
      const jsonString = JSON.stringify(task, null, 2);
      const blob = new Blob([jsonString], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;

      // Extract original filename from inputFileUrl and create new JSON filename
      const originalFilename = decodeURIComponent(
        inputFileUrl?.split("?")[0].split("/").pop() || "document.pdf"
      );
      const baseFilename = originalFilename.replace(".pdf", "");
      a.download = `chunkr_json_${baseFilename}.json`;

      a.click();
      URL.revokeObjectURL(url);
    }
  }, [task, inputFileUrl]);

  const handleDownloadHTML = useCallback(() => {
    if (output) {
      const combinedHtml = output.chunks
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

      const originalFilename = decodeURIComponent(
        inputFileUrl?.split("?")[0].split("/").pop() || "document.pdf"
      );
      const baseFilename = originalFilename.replace(".pdf", "");
      a.download = `chunkr_html_${baseFilename}.html`;

      a.click();
      URL.revokeObjectURL(url);
    }
  }, [output, inputFileUrl]);

  const handleDownloadMarkdown = useCallback(() => {
    if (output) {
      const combinedMarkdown = output.chunks
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

      const originalFilename = decodeURIComponent(
        inputFileUrl?.split("?")[0].split("/").pop() || "document.pdf"
      );
      const baseFilename = originalFilename.replace(".pdf", "");
      a.download = `chunkr_markdown_${baseFilename}.md`;

      a.click();
      URL.revokeObjectURL(url);
    }
  }, [output, inputFileUrl]);

  // Add scroll handler for PDF container to load more pages
  const handleScroll = useCallback(
    (e: Event) => {
      const target = e.target as HTMLDivElement;
      const { scrollTop, scrollHeight, clientHeight } = target;
      const scrolledToBottom = scrollHeight - scrollTop <= clientHeight * 1.5;

      if (target.classList.contains("scrollable-content")) {
        // Handle chunk loading
        if (scrolledToBottom && loadedChunks < (output?.chunks?.length || 0)) {
          setLoadedChunks((prev) =>
            Math.min(prev + CHUNK_LOAD_SIZE, output?.chunks?.length || 0)
          );
        }
      } else if (target.classList.contains("pdf-container")) {
        // Handle page loading
        if (scrolledToBottom && loadedPages < (numPages || 0)) {
          setLoadedPages((prev) =>
            Math.min(prev + PAGE_CHUNK_SIZE, numPages || 0)
          );
        }
      }
    },
    [loadedChunks, output?.chunks?.length, loadedPages, numPages]
  );

  // Add scroll listeners to both containers
  useEffect(() => {
    const scrollableContent = document.querySelector(".scrollable-content");
    const pdfContainer = document.querySelector(".pdf-container");

    if (scrollableContent) {
      scrollableContent.addEventListener("scroll", handleScroll);
    }
    if (pdfContainer) {
      pdfContainer.addEventListener("scroll", handleScroll);
    }

    return () => {
      if (scrollableContent) {
        scrollableContent.removeEventListener("scroll", handleScroll);
      }
      if (pdfContainer) {
        pdfContainer.removeEventListener("scroll", handleScroll);
      }
    };
  }, [handleScroll]);

  // Change the state name to match our new dropdown
  const [showDownloadOptions, setShowDownloadOptions] = useState(false);

  // Add click outside handler for download dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        !event.target ||
        !(event.target as Element).closest(".download-dropdown-container")
      ) {
        setShowDownloadOptions(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  // Add a check for structured extraction availability

  const renderDownloadDropdown = () => (
    <div
      className="download-dropdown-container"
      style={{ position: "relative" }}
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
          <Flex className="download-options-menu" direction="column">
            <div
              className="download-options-menu-item"
              onClick={(e) => {
                e.stopPropagation(); // Prevent event from bubbling up
                handleDownloadOriginalFile();
                setShowDownloadOptions(false);
              }}
            >
              <Text size="2" weight="medium" style={{ color: "#FFF" }}>
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
              <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                PDF
              </Text>
            </div>
            <div
              className="download-options-menu-item"
              onClick={(e) => {
                console.log("JSON menu item clicked");
                e.stopPropagation();
                handleDownloadJSON();
                setShowDownloadOptions(false);
              }}
            >
              <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                JSON
              </Text>
            </div>
            <div
              className="download-options-menu-item"
              onClick={(e) => {
                e.stopPropagation(); // Prevent event from bubbling up
                handleDownloadHTML();
                setShowDownloadOptions(false);
              }}
            >
              <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                HTML
              </Text>
            </div>
            <div
              className="download-options-menu-item"
              onClick={(e) => {
                e.stopPropagation(); // Prevent event from bubbling up
                handleDownloadMarkdown();
                setShowDownloadOptions(false);
              }}
            >
              <Text size="2" weight="medium" style={{ color: "#FFF" }}>
                Markdown
              </Text>
            </div>
          </Flex>
        </div>
      )}
    </div>
  );

  if (!output) {
    return <Loader />;
  }

  return (
    <Flex direction="column" width="100%">
      <Flex
        direction="row"
        width="100%"
        justify="between"
        className="viewer-header"
      >
        <Flex className="viewer-header-left-buttons" gap="16px">
          <BetterButton
            onClick={() => setSelectedView("html")}
            active={selectedView === "html"}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M4.17456 5.15007C4.08271 4.54492 4.55117 4 5.16324 4H18.8368C19.4488 4 19.9173 4.54493 19.8254 5.15007L18.0801 16.6489C18.03 16.9786 17.8189 17.2617 17.5172 17.4037L12.4258 19.7996C12.1561 19.9265 11.8439 19.9265 11.5742 19.7996L6.4828 17.4037C6.18107 17.2617 5.96997 16.9786 5.91993 16.6489L4.17456 5.15007Z"
                stroke="#FFFFFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M15 7.5H9.5V11H14.5V14.5L12.3714 15.3514C12.133 15.4468 11.867 15.4468 11.6286 15.3514L9.5 14.5"
                stroke="#FFFFFF"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <Text
              size="2"
              weight="medium"
              style={{ color: "rgba(255, 255, 255, 0.95)" }}
            >
              HTML
            </Text>
          </BetterButton>
          <BetterButton
            onClick={() => setSelectedView("markdown")}
            active={selectedView === "markdown"}
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
                fill="#FFFFFF"
              />
            </svg>
            <Text
              size="2"
              weight="medium"
              style={{ color: "rgba(255, 255, 255, 0.95)" }}
            >
              Markdown
            </Text>
          </BetterButton>
        </Flex>
        <Flex className="viewer-header-right-buttons" gap="16px">
          {renderDownloadDropdown()}
          <Flex
            className="viewer-header-config-button"
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
          >
            <Flex align="center" gap="2">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <g clip-path="url(#clip0_304_23691)">
                  <path
                    d="M20.6519 14.43L19.0419 13.5C19.1519 13.02 19.2019 12.51 19.2019 12C19.2019 11.49 19.1519 10.98 19.0419 10.5L20.6519 9.57C20.8919 9.44 20.9719 9.13 20.8419 8.89L19.0919 5.86C18.9519 5.62 18.6419 5.54 18.4019 5.68L16.7819 6.62C16.0419 5.94 15.1719 5.42 14.2019 5.12V3.25C14.2019 2.97 13.9819 2.75 13.7019 2.75H10.2019C9.92193 2.75 9.70193 2.97 9.70193 3.25V5.12C8.73193 5.42 7.86193 5.94 7.12193 6.62L5.50193 5.68C5.26193 5.54 4.95193 5.62 4.81193 5.86L3.06193 8.89C2.93193 9.13 3.01193 9.44 3.25193 9.57L4.86193 10.5C4.75193 10.98 4.70193 11.49 4.70193 12C4.70193 12.51 4.75193 13.02 4.86193 13.5L3.25193 14.43C3.01193 14.56 2.93193 14.87 3.06193 15.11L4.81193 18.14C4.95193 18.38 5.26193 18.46 5.50193 18.32L7.12193 17.38C7.86193 18.06 8.73193 18.58 9.70193 18.88V20.75C9.70193 21.03 9.92193 21.25 10.2019 21.25H13.7019C13.9819 21.25 14.2019 21.03 14.2019 20.75V18.88C15.1719 18.58 16.0419 18.06 16.7819 17.38L18.4019 18.32C18.6419 18.46 18.9519 18.38 19.0919 18.14L20.8419 15.11C20.9719 14.87 20.8919 14.56 20.6519 14.43ZM15.0919 12.84C15.0119 13.11 14.9119 13.38 14.7619 13.62C14.4819 14.12 14.0719 14.53 13.5719 14.81C13.3319 14.96 13.0619 15.06 12.7919 15.14C12.5219 15.21 12.2419 15.25 11.9519 15.25C11.6619 15.25 11.3819 15.21 11.1119 15.14C10.8419 15.06 10.5719 14.96 10.3319 14.81C9.83193 14.53 9.42193 14.12 9.14193 13.62C8.99193 13.38 8.89193 13.11 8.81193 12.84C8.74193 12.57 8.70193 12.29 8.70193 12C8.70193 11.71 8.74193 11.43 8.81193 11.16C8.89193 10.89 8.99193 10.62 9.14193 10.38C9.42193 9.88 9.83193 9.47 10.3319 9.19C10.5719 9.04 10.8419 8.94 11.1119 8.86C11.3819 8.79 11.6619 8.75 11.9519 8.75C12.2419 8.75 12.5219 8.79 12.7919 8.86C13.0619 8.94 13.3319 9.04 13.5719 9.19C14.0719 9.47 14.4819 9.88 14.7619 10.38C14.9119 10.62 15.0119 10.89 15.0919 11.16C15.1619 11.43 15.2019 11.71 15.2019 12C15.2019 12.29 15.1619 12.57 15.0919 12.84Z"
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
            </Flex>

            {showConfig && (
              <div className="config-popup">
                <Flex
                  className="config-popup-content"
                  direction="column"
                  height="100%"
                  width="100%"
                  overflow="auto"
                >
                  <ReactJson
                    src={task.configuration}
                    theme="monokai"
                    displayDataTypes={false}
                    enableClipboard={false}
                    style={{
                      backgroundColor: "transparent",
                      padding: "12px",
                      fontSize: "12px",
                    }}
                    collapsed={1}
                    name={false}
                  />
                </Flex>
              </div>
            )}
          </Flex>
        </Flex>
      </Flex>
      <PanelGroup
        direction="horizontal"
        style={{ backgroundColor: "rgba(255, 255, 255, 0.00)" }}
      >
        <Panel
          defaultSize={50}
          minSize={20}
          onResize={() => {
            const panelElement = document.querySelector(".scrollable-content");
            if (panelElement) {
              setLeftPanelWidth(panelElement.clientWidth);
            }
          }}
        >
          <div className="scrollable-content">
            {output.chunks.length === 0 ? (
              <Text
                size="4"
                weight="medium"
                style={{ color: "rgba(255, 255, 255, 0.8)" }}
              >
                No content available for this PDF.
              </Text>
            ) : (
              <>
                {output.chunks
                  .slice(0, loadedChunks)
                  .map((chunk: Chunk, index: number) => (
                    <SegmentChunk
                      key={chunk.chunk_id}
                      chunk={chunk}
                      chunkId={chunk.chunk_id}
                      containerWidth={leftPanelWidth}
                      selectedView={selectedView}
                      ref={(el) => (chunkRefs.current[index] = el)}
                      onSegmentClick={scrollToSegment}
                      activeSegment={activeSegment}
                    />
                  ))}
                {loadedChunks < output.chunks.length && (
                  <div
                    className="loading-more-chunks"
                    style={{
                      textAlign: "center",
                      padding: "20px",
                      color: "rgba(255, 255, 255, 0.6)",
                    }}
                  >
                    Loading more chunks...
                  </div>
                )}
              </>
            )}
          </div>
        </Panel>

        <PanelResizeHandle
          className="resize-handle"
          hitAreaMargins={{ coarse: 15, fine: 5 }}
        />

        <Panel
          defaultSize={50}
          minSize={20}
          style={{ backgroundColor: "rgba(255, 255, 255, 0.03)" }}
        >
          {memoizedOutput && memoizedOutput.pdf_url && (
            <MemoizedPDF
              content={memoizedOutput.chunks}
              inputFileUrl={memoizedOutput.pdf_url}
              onSegmentClick={handlePDFSegmentClick}
              activeSegment={activeSegment}
              loadedPages={loadedPages}
              onLoadSuccess={(pages) => setNumPages(pages)}
            />
          )}
        </Panel>
      </PanelGroup>
    </Flex>
  );
}
