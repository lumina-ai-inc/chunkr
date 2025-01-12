import { forwardRef, memo, useCallback, useMemo, useState } from "react";
import { Chunk, Segment } from "../../models/taskResponse.model";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "./SegmentChunk.css";
import DOMPurify from "dompurify";
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

    // Skip if content is already rendered math
    if (tempHtml.includes('class="katex"')) {
      return tempHtml;
    }

    // Only process content within <span class="formula"> tags
    tempHtml = tempHtml.replace(
      /<span class="formula">(.*?)<\/span>/g,
      (match, content) => {
        try {
          // Handle display mode math ($$...$$)
          content = content.replace(/\$\$(.*?)\$\$/g, (formula: string) =>
            katex.renderToString(formula, {
              displayMode: true,
              throwOnError: false,
            })
          );

          // Handle inline math ($...$)
          content = content.replace(/\$(.*?)\$/g, (formula: string) =>
            katex.renderToString(formula, {
              displayMode: false,
              throwOnError: false,
            })
          );

          return `<span class="formula">${content}</span>`;
        } catch (err) {
          console.error("KaTeX processing error:", err);
          return match;
        }
      }
    );

    return tempHtml;
  }, [html]);

  return (
    <div
      className="latex-content"
      dangerouslySetInnerHTML={{
        __html: DOMPurify.sanitize(processedHtml, {
          ADD_TAGS: [
            "math",
            "semantics",
            "annotation",
            "mrow",
            "mi",
            "mo",
            "mtext",
            "mspace",
            "msup",
            "msub",
            "mfrac",
            "span",
            "svg",
            "path",
            "mstyle",
            "mn",
            "munderover",
            "mover",
            "munder",
            "mroot",
          ],
          ADD_ATTR: [
            "encoding",
            "class",
            "style",
            "viewBox",
            "d",
            "xmlns",
            "mathvariant",
          ],
        }),
      }}
    />
  );
});

const MemoizedMarkdown = memo(({ content }: { content: string }) => (
  <ReactMarkdown
    className="cyan-2"
    remarkPlugins={[remarkMath]}
    rehypePlugins={[rehypeKatex]}
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
  />
));

export const SegmentChunk = memo(
  forwardRef<
    HTMLDivElement,
    {
      chunk: Chunk;
      chunkIndex: number;
      containerWidth: number;
      selectedView: "html" | "markdown" | "json" | "structured";
      onSegmentClick?: (chunkIndex: number, segmentIndex: number) => void;
      activeSegment?: { chunkIndex: number; segmentIndex: number } | null;
    }
  >(
    (
      {
        chunk,
        chunkIndex,
        containerWidth,
        selectedView,
        onSegmentClick,
        activeSegment,
      },
      ref
    ) => {
      const [showJson, setShowJson] = useState(false);

      const combinedMarkdown = useMemo(() => {
        return chunk.segments
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
          .trim();
      }, [chunk.segments]);

      const combinedHtml = useMemo(() => {
        return chunk.segments
          .map((segment) => {
            // Handle table images
            if (
              segment.segment_type === "Table" &&
              segment.html?.startsWith("<span class=")
            ) {
              return `<br><img src="${segment.image}" />`;
            }

            // Handle formula segments with class="formula"
            if (segment.html?.includes('class="formula"')) {
              const formulaMatch = segment.html.match(
                /<span class="formula">(.*?)<\/span>/s
              );
              if (formulaMatch) {
                const formula = formulaMatch[1]
                  .replace(/&gt;/g, ">")
                  .replace(/&lt;/g, "<")
                  .replace(/&amp;/g, "&")
                  .trim();
                try {
                  return katex.renderToString(formula, {
                    displayMode: true,
                    throwOnError: false,
                    strict: false,
                    trust: true,
                    macros: {
                      "\\R": "\\mathbb{R}",
                    },
                  });
                } catch (error) {
                  console.error("KaTeX rendering error:", error);
                  return `<div class="math math-display">$$\\begin{aligned}${formula}\\end{aligned}$$</div>`;
                }
              }
            }

            // Handle content with LaTeX delimiters
            if (segment.content && segment.content.includes("\\")) {
              try {
                return katex.renderToString(segment.content, {
                  displayMode: true,
                  throwOnError: false,
                });
              } catch (error) {
                console.error("KaTeX rendering error:", error);
                return `<div class="math math-display">$$${segment.content}$$</div>`;
              }
            }

            return segment.html || "";
          })
          .filter(Boolean)
          .join("");
      }, [chunk.segments]);

      const isSegmentActive = useCallback(
        (segmentIndex: number) => {
          return (
            activeSegment?.chunkIndex === chunkIndex &&
            activeSegment?.segmentIndex === segmentIndex
          );
        },
        [activeSegment, chunkIndex]
      );

      const renderContent = () => {
        const firstPageNumber = chunk.segments[0]?.page_number;

        // If showJson is true, always show JSON view regardless of selectedView
        if (showJson) {
          return chunk.segments.map(
            (segment: Segment, segmentIndex: number) => (
              <div
                key={segmentIndex}
                className={`segment-item ${isSegmentActive(segmentIndex) ? "active" : ""}`}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onSegmentClick?.(chunkIndex, segmentIndex);
                }}
              >
                <MemoizedJson segment={segment} />
              </div>
            )
          );
        }

        // Original switch statement for other views
        switch (selectedView) {
          case "html":
          case "markdown":
            return (
              <div
                className={`segment-content-wrapper ${activeSegment?.chunkIndex === chunkIndex ? "active" : ""}`}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  if (firstPageNumber) {
                    onSegmentClick?.(chunkIndex, 0);
                    requestAnimationFrame(() => {
                      window.dispatchEvent(
                        new CustomEvent("scroll-to-page", {
                          detail: { pageNumber: firstPageNumber },
                        })
                      );
                    });
                  }
                }}
              >
                {selectedView === "html" ? (
                  <MemoizedHtml html={combinedHtml} />
                ) : (
                  <MemoizedMarkdown content={combinedMarkdown} />
                )}
              </div>
            );
          case "json":
            return chunk.segments.map(
              (segment: Segment, segmentIndex: number) => (
                <div
                  key={segmentIndex}
                  className={`segment-item ${isSegmentActive(segmentIndex) ? "active" : ""}`}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onSegmentClick?.(chunkIndex, segmentIndex);
                  }}
                >
                  <MemoizedJson segment={segment} />
                </div>
              )
            );
          case "structured":
            return chunk.segments.map(
              (segment: Segment, segmentIndex: number) => (
                <div
                  key={segmentIndex}
                  className={`structured-segment ${isSegmentActive(segmentIndex) ? "active" : ""}`}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onSegmentClick?.(chunkIndex, segmentIndex);
                  }}
                >
                  <div className="segment-type">{segment.segment_type}</div>
                  <div className="segment-content">{segment.content}</div>
                </div>
              )
            );
        }
      };

      const horizontalScrollRef = useHorizontalDragScroll();

      const handleCopy = () => {
        const textToCopy =
          selectedView === "html" ? combinedHtml : combinedMarkdown;
        navigator.clipboard.writeText(textToCopy);
        toast.success("Copied to clipboard");
      };

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
          data-chunk-index={chunkIndex}
          style={{
            maxWidth: containerWidth,
            position: "relative",
            overflow: "auto",
          }}
        >
          <Flex mb="4" gap="4">
            <BetterButton onClick={handleCopy}>
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
            <BetterButton
              active={showJson}
              onClick={() => setShowJson(!showJson)}
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
          </Flex>
          <div className="segment-content">{renderContent()}</div>
        </div>
      );
    }
  )
);

SegmentChunk.displayName = "SegmentChunk";
