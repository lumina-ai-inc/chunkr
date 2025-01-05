import { forwardRef, memo, useCallback, useMemo } from "react";
import { Chunk, Segment } from "../../models/chunk.model";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "./SegmentChunk.css";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import ReactJson from "react-json-view";
import "katex/dist/katex.min.css";
import katex from "katex";
import { useHorizontalDragScroll } from "../../hooks/useHorizontalDragScroll";

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
    rehypePlugins={[rehypeKatex, remarkGfm]}
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
    ({
      chunk,
      chunkIndex,
      containerWidth,
      selectedView,
      onSegmentClick,
      activeSegment,
    }) => {
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
                  <div className="segment-content">
                    {segment.content || segment.markdown || segment.html}
                  </div>
                </div>
              )
            );
        }
      };

      const scrollRef = useHorizontalDragScroll();

      return (
        <div
          className="segment-chunk"
          ref={scrollRef}
          style={{
            maxWidth: containerWidth,
            position: "relative",
            overflow: "auto",
          }}
        >
          <div className="segment-content">{renderContent()}</div>
        </div>
      );
    }
  )
);

SegmentChunk.displayName = "SegmentChunk";
