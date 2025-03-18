import {
  forwardRef,
  memo,
  useCallback,
  useMemo,
  useState,
  useEffect,
} from "react";
import { Chunk, Segment } from "../../models/taskResponse.model";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import "./SegmentChunk.css";
import ReactMarkdown from "react-markdown";
import ReactJson from "react-json-view";
import "katex/dist/katex.min.css";
import katex from "katex";
import { useHorizontalDragScroll } from "../../hooks/useHorizontalDragScroll";
import BetterButton from "../BetterButton/BetterButton";
import { Flex, Text } from "@radix-ui/themes";
import toast from "react-hot-toast";
// Memoized content renderers
const MemoizedHtml = memo(({ html }: { html: string }) => {
  const processedHtml = useMemo(() => {
    let tempHtml = html;

    // Process base64 images
    tempHtml = tempHtml.replace(
      /<img[^>]+src="([^"]+)"[^>]*>/g,
      (match, src) => {
        // Check if it's a base64 image that needs the data URI prefix
        if (src.startsWith("/9j/")) {
          return match.replace(src, `data:image/jpeg;base64,${src}`);
        }
        return match;
      }
    );

    return tempHtml;
  }, [html]);

  return (
    <div
      className="latex-content"
      dangerouslySetInnerHTML={{
        __html: processedHtml,
      }}
    />
  );
});

const MemoizedMarkdown = memo(({ content }: { content: string }) => (
  <ReactMarkdown
    className="cyan-2"
    remarkPlugins={[remarkMath, remarkGfm]}
    rehypePlugins={[rehypeKatex]}
    components={{
      img: ({ src, alt, ...props }) => {
        // Check if it's a base64 image that needs the data URI prefix
        const formattedSrc = src?.startsWith("/9j/")
          ? `data:image/jpeg;base64,${src}`
          : src;
        return <img src={formattedSrc} alt={alt} {...props} />;
      },
    }}
  >
    {content}
  </ReactMarkdown>
));

const MemoizedJson = memo(({ segment }: { segment: Segment }) => (
  <ReactJson
    src={segment}
    theme="monokai"
    displayDataTypes={false}
    enableClipboard={false}
    style={{ backgroundColor: "transparent" }}
    collapsed={1}
    name={false}
  />
));

export const SegmentChunk = memo(
  forwardRef<
    HTMLDivElement,
    {
      chunk: Chunk;
      chunkId: string;
      containerWidth: number;
      selectedView: "html" | "markdown" | "json";
      onSegmentClick?: (chunkId: string, segmentId: string) => void;
      activeSegment?: { chunkId: string; segmentId: string } | null;
    }
  >(
    (
      {
        chunk,
        chunkId,
        containerWidth,
        selectedView,
        onSegmentClick,
        activeSegment,
      },
      ref
    ) => {
      const [segmentDisplayModes, setSegmentDisplayModes] = useState<{
        [key: string]: {
          showJson: boolean;
          showLLM: boolean;
          showImage: boolean;
        };
      }>({});

      const handleSegmentDisplayMode = useCallback(
        (segmentId: string, mode: "json" | "llm" | "image") => {
          setSegmentDisplayModes((prev) => {
            const current = prev[segmentId] || {
              showJson: false,
              showLLM: false,
              showImage: false,
            };
            const updated = { ...current };

            if (mode === "json") {
              updated.showJson = !current.showJson;
              updated.showLLM = false;
              updated.showImage = false;
            } else if (mode === "llm") {
              updated.showLLM = !current.showLLM;
              updated.showJson = false;
              updated.showImage = false;
            } else if (mode === "image") {
              updated.showImage = !current.showImage;
              updated.showJson = false;
              updated.showLLM = false;
            }

            return { ...prev, [segmentId]: updated };
          });
        },
        []
      );

      const handleCopySegment = useCallback(
        (segment: Segment) => {
          const mode = segmentDisplayModes[segment.segment_id];
          let textToCopy = "";

          if (mode?.showJson) {
            textToCopy = JSON.stringify(segment, null, 2);
          } else if (mode?.showLLM) {
            textToCopy = segment.llm || "";
          } else if (mode?.showImage) {
            textToCopy = segment.image || "";
          } else {
            textToCopy =
              selectedView === "html"
                ? segment.html || ""
                : segment.markdown || segment.content || "";
          }

          navigator.clipboard.writeText(textToCopy);
          toast.success("Copied to clipboard");
        },
        [segmentDisplayModes, selectedView]
      );

      const renderSegmentHtml = useCallback(
        (segment: Segment) => {
          const mode = segmentDisplayModes[segment.segment_id];

          // Just return the content directly without the segment-item wrapper
          if (mode?.showJson) {
            return <MemoizedJson segment={segment} />;
          }

          if (mode?.showLLM && segment.llm) {
            return <MemoizedMarkdown content={segment.llm} />;
          }

          if (mode?.showImage && segment.image) {
            return (
              <img
                src={segment.image}
                alt="Cropped segment"
                style={{ maxWidth: "100%" }}
              />
            );
          }

          // Handle table images
          if (
            segment.segment_type === "Table" &&
            segment.html?.startsWith("<span class=")
          ) {
            return <img src={segment.image || ""} alt="Table" />;
          }

          // Handle formula segments with class="formula"
          if (segment.html?.includes('class="formula"')) {
            let html = segment.html;
            // Process all formula spans
            html = html.replace(
              /<span class="formula">(.*?)<\/span>/gs,
              (match, formula) => {
                try {
                  const processedFormula = formula
                    .replace(/&gt;/g, ">")
                    .replace(/&lt;/g, "<")
                    .replace(/&amp;/g, "&")
                    .replace(/\\\(|\\\)/g, "") // Remove \( and \) delimiters
                    .trim();
                  return katex.renderToString(processedFormula, {
                    displayMode: false,
                    throwOnError: false,
                  });
                } catch (error) {
                  console.error("KaTeX rendering error:", error);
                  return match; // Return original on error
                }
              }
            );

            return <div dangerouslySetInnerHTML={{ __html: html }} />;
          }

          // Handle content with LaTeX delimiters
          if (segment.content && segment.content.includes("\\")) {
            try {
              return (
                <div
                  dangerouslySetInnerHTML={{
                    __html: katex.renderToString(segment.content, {
                      displayMode: true,
                      throwOnError: false,
                    }),
                  }}
                />
              );
            } catch (error) {
              console.error("KaTeX rendering error:", error);
              return (
                <div className="math math-display">$$${segment.content}$$</div>
              );
            }
          }

          return <MemoizedHtml html={segment.html || ""} />;
        },
        [segmentDisplayModes]
      );

      const renderContent = () => {
        return chunk.segments.map((segment, segmentIndex) => {
          const isActive =
            activeSegment?.chunkId === chunkId &&
            activeSegment?.segmentId === segment.segment_id;

          const mode = segmentDisplayModes[segment.segment_id];

          const renderSegmentContent = () => {
            if (mode?.showJson) {
              return <MemoizedJson segment={segment} />;
            }

            if (mode?.showLLM && segment.llm) {
              if (
                segment.llm.includes("**") ||
                segment.llm.includes("*") ||
                segment.llm.includes("#")
              ) {
                return <MemoizedMarkdown content={segment.llm} />;
              }

              return (
                <div
                  className="latex-content"
                  dangerouslySetInnerHTML={{
                    __html: segment.llm
                      .replace(/\n\n/g, "<br/><br/>")
                      .replace(
                        /([a-zA-Z])<sub>([^<]+)<\/sub>/g,
                        "$1<sub>$2</sub>"
                      )
                      .replace(
                        /([a-zA-Z])<sup>([^<]+)<\/sup>/g,
                        "$1<sup>$2</sup>"
                      ),
                  }}
                />
              );
            }

            if (mode?.showImage && segment.image) {
              return (
                <img
                  src={segment.image}
                  alt="Cropped segment"
                  style={{ maxWidth: "100%" }}
                />
              );
            }

            // Regular content rendering based on selected view
            return selectedView === "html" ? (
              renderSegmentHtml(segment)
            ) : (
              <MemoizedMarkdown
                content={segment.markdown || segment.content || ""}
              />
            );
          };

          return (
            <div
              key={segmentIndex}
              className={`segment-item ${isActive ? "active" : ""}`}
              data-chunk-id={chunkId}
              data-segment-id={segment.segment_id}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                // Only call onSegmentClick if this segment is not already active
                if (!isActive) {
                  onSegmentClick?.(chunkId, segment.segment_id);
                }
              }}
              style={{ maxWidth: `calc(${containerWidth}px - 32px)` }}
            >
              {isActive && (
                <Flex mb="4" gap="4">
                  <BetterButton onClick={() => handleCopySegment(segment)}>
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        fill-rule="evenodd"
                        clip-rule="evenodd"
                        d="M15 1.25H10.9436C9.10583 1.24998 7.65019 1.24997 6.51098 1.40314C5.33856 1.56076 4.38961 1.89288 3.64124 2.64124C2.89288 3.38961 2.56076 4.33856 2.40314 5.51098C2.24997 6.65019 2.24998 8.10582 2.25 9.94357V16C2.25 17.8722 3.62205 19.424 5.41551 19.7047C5.55348 20.4687 5.81753 21.1208 6.34835 21.6517C6.95027 22.2536 7.70814 22.5125 8.60825 22.6335C9.47522 22.75 10.5775 22.75 11.9451 22.75H15.0549C16.4225 22.75 17.5248 22.75 18.3918 22.6335C19.2919 22.5125 20.0497 22.2536 20.6517 21.6517C21.2536 21.0497 21.5125 20.2919 21.6335 19.3918C21.75 18.5248 21.75 17.4225 21.75 16.0549V10.9451C21.75 9.57754 21.75 8.47522 21.6335 7.60825C21.5125 6.70814 21.2536 5.95027 20.6517 5.34835C20.1208 4.81753 19.4687 4.55348 18.7047 4.41551C18.424 2.62205 16.8722 1.25 15 1.25ZM17.1293 4.27117C16.8265 3.38623 15.9876 2.75 15 2.75H11C9.09318 2.75 7.73851 2.75159 6.71085 2.88976C5.70476 3.02502 5.12511 3.27869 4.7019 3.7019C4.27869 4.12511 4.02502 4.70476 3.88976 5.71085C3.75159 6.73851 3.75 8.09318 3.75 10V16C3.75 16.9876 4.38624 17.8265 5.27117 18.1293C5.24998 17.5194 5.24999 16.8297 5.25 16.0549V10.9451C5.24998 9.57754 5.24996 8.47522 5.36652 7.60825C5.48754 6.70814 5.74643 5.95027 6.34835 5.34835C6.95027 4.74643 7.70814 4.48754 8.60825 4.36652C9.47522 4.24996 10.5775 4.24998 11.9451 4.25H15.0549C15.8297 4.24999 16.5194 4.24998 17.1293 4.27117ZM7.40901 6.40901C7.68577 6.13225 8.07435 5.9518 8.80812 5.85315C9.56347 5.75159 10.5646 5.75 12 5.75H15C16.4354 5.75 17.4365 5.75159 18.1919 5.85315C18.9257 5.9518 19.3142 6.13225 19.591 6.40901C19.8678 6.68577 20.0482 7.07435 20.1469 7.80812C20.2484 8.56347 20.25 9.56458 20.25 11V16C20.25 17.4354 20.2484 18.4365 20.1469 19.1919C20.0482 19.9257 19.8678 20.3142 19.591 20.591C19.3142 20.8678 18.9257 21.0482 18.1919 21.1469C17.4365 21.2484 16.4354 21.25 15 21.25H12C10.5646 21.25 9.56347 21.2484 8.80812 21.1469C8.07435 21.0482 7.68577 20.8678 7.40901 20.591C7.13225 20.3142 6.9518 19.9257 6.85315 19.1919C6.75159 18.4365 6.75 17.4354 6.75 16V11C6.75 9.56458 6.75159 8.56347 6.85315 7.80812C6.9518 7.07435 7.13225 6.68577 7.40901 6.40901Z"
                        fill="#FFFFFF"
                      />
                    </svg>
                    <Text
                      size="1"
                      weight="medium"
                      style={{ color: "rgba(255, 255, 255, 0.95)" }}
                    >
                      Copy
                    </Text>
                  </BetterButton>
                  {mode?.showJson || mode?.showLLM || mode?.showImage ? (
                    <BetterButton
                      onClick={() => {
                        setSegmentDisplayModes((prev) => ({
                          ...prev,
                          [segment.segment_id]: {
                            showJson: false,
                            showLLM: false,
                            showImage: false,
                          },
                        }));
                      }}
                    >
                      <Flex align="center" gap="2">
                        <svg
                          width="16"
                          height="16"
                          viewBox="0 0 24 24"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"
                            fill="#FFFFFF"
                          />
                        </svg>
                        <Text
                          size="1"
                          weight="medium"
                          style={{ color: "rgba(255, 255, 255, 0.95)" }}
                        >
                          Return
                        </Text>
                      </Flex>
                    </BetterButton>
                  ) : (
                    <>
                      <BetterButton
                        active={mode?.showJson}
                        onClick={() =>
                          handleSegmentDisplayMode(segment.segment_id, "json")
                        }
                      >
                        <Flex align="center" gap="2">
                          <svg
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <rect width="24" height="24" fill="none" />
                            <path
                              d="M5,3H7V5H5v5a2,2,0,0,1-2,2,2,2,0,0,1,2,2v5H7v2H5c-1.07-.27-2-.9-2-2V15a2,2,0,0,0-2-2H0V11H1A2,2,0,0,0,3,9V5A2,2,0,0,1,5,3M19,3a2,2,0,0,1,2,2V9a2,2,0,0,0,2,2h1v2H23a2,2,0,0,0-2,2v4a2,2,0,0,1-2,2H17V19h2V14a2,2,0,0,1,2-2,2,2,0,0,1-2-2V5H17V3h2M12,15a1,1,0,1,1-1,1,1,1,0,0,1,1-1M8,15a1,1,0,1,1-1,1,1,1,0,0,1,1-1m8,0a1,1,0,1,1-1,1A1,1,0,0,1,16,15Z"
                              fill="#FFFFFF"
                            />
                          </svg>
                          <Text
                            size="1"
                            weight="medium"
                            style={{ color: "rgba(255, 255, 255, 0.95)" }}
                          >
                            JSON
                          </Text>
                        </Flex>
                      </BetterButton>
                      {segment.llm && (
                        <BetterButton
                          active={mode?.showLLM}
                          onClick={() =>
                            handleSegmentDisplayMode(segment.segment_id, "llm")
                          }
                        >
                          <Flex align="center" gap="2">
                            <svg
                              width="16px"
                              height="16px"
                              viewBox="0 0 24 24"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg"
                            >
                              <path
                                d="M19 3V7M17 5H21M19 17V21M17 19H21M10 5L8.53001 8.72721C8.3421 9.20367 8.24814 9.4419 8.10427 9.64278C7.97675 9.82084 7.82084 9.97675 7.64278 10.1043C7.4419 10.2481 7.20367 10.3421 6.72721 10.53L3 12L6.72721 13.47C7.20367 13.6579 7.4419 13.7519 7.64278 13.8957C7.82084 14.0233 7.97675 14.1792 8.10427 14.3572C8.24814 14.5581 8.3421 14.7963 8.53001 15.2728L10 19L11.47 15.2728C11.6579 14.7963 11.7519 14.5581 11.8957 14.3572C12.0233 14.1792 12.1792 14.0233 12.3572 13.8957C12.5581 13.7519 12.7963 13.6579 13.2728 13.47L17 12L13.2728 10.53C12.7963 10.3421 12.5581 10.2481 12.3572 10.1043C12.1792 9.97675 12.0233 9.82084 11.8957 9.64278C11.7519 9.4419 11.6579 9.20367 11.47 8.72721L10 5Z"
                                stroke="#FFFFFF"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                            </svg>
                            <Text
                              size="1"
                              weight="medium"
                              style={{ color: "rgba(255, 255, 255, 0.95)" }}
                            >
                              LLM
                            </Text>
                          </Flex>
                        </BetterButton>
                      )}
                      {segment.image && (
                        <BetterButton
                          active={mode?.showImage}
                          onClick={() =>
                            handleSegmentDisplayMode(
                              segment.segment_id,
                              "image"
                            )
                          }
                        >
                          <Flex align="center" gap="2">
                            <svg
                              width="16"
                              height="16"
                              viewBox="0 0 24 24"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg"
                            >
                              <path
                                d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19ZM13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z"
                                fill="#FFFFFF"
                              />
                            </svg>
                            <Text
                              size="1"
                              weight="medium"
                              style={{ color: "rgba(255, 255, 255, 0.95)" }}
                            >
                              Cropped Image
                            </Text>
                          </Flex>
                        </BetterButton>
                      )}
                    </>
                  )}
                </Flex>
              )}
              {renderSegmentContent()}
            </div>
          );
        });
      };

      const horizontalScrollRef = useHorizontalDragScroll();

      // Reset segment display mode when it becomes inactive
      useEffect(() => {
        chunk.segments.forEach((segment) => {
          if (activeSegment?.segmentId !== segment.segment_id) {
            setSegmentDisplayModes((prev) => ({
              ...prev,
              [segment.segment_id]: {
                showJson: false,
                showLLM: false,
                showImage: false,
              },
            }));
          }
        });
      }, [activeSegment, chunk.segments]);

      return (
        <div
          className="segment-chunk"
          ref={(el) => {
            if (typeof ref === "function") {
              ref(el);
            } else if (ref) {
              ref.current = el;
            }
            horizontalScrollRef.current = el;
          }}
          data-chunk-id={chunkId}
          style={{
            position: "relative",
          }}
        >
          <div className="segment-content">{renderContent()}</div>
        </div>
      );
    }
  )
);

SegmentChunk.displayName = "SegmentChunk";
