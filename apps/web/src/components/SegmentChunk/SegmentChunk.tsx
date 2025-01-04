import { useMemo, forwardRef } from "react";
import { Chunk, Segment } from "../../models/chunk.model";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "./SegmentChunk.css";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import ReactJson from "react-json-view";

export const SegmentChunk = forwardRef<
  HTMLDivElement,
  {
    chunk: Chunk;
    chunkIndex: number;
    containerWidth: number;
    selectedView: "html" | "markdown" | "json" | "structured";
  }
>(({ chunk, containerWidth, selectedView }, ref) => {
  const combinedMarkdown = useMemo(() => {
    let lastContent = "";
    return chunk.segments
      .map((segment) => {
        const textContent = segment.content || "";
        if (
          segment.segment_type === "Table" &&
          segment.html?.startsWith("<span class=")
        ) {
          return `![Image](${segment.image})`;
        }
        return segment.markdown || textContent;
      })
      .filter(Boolean)
      .join("\n\n")
      .trim();
  }, [chunk.segments]);

  const combinedHtml = useMemo(() => {
    return chunk.segments
      .map((segment) => {
        if (
          segment.segment_type === "Table" &&
          segment.html?.startsWith("<span class=")
        ) {
          return `<br><img src="${segment.image}" />`;
        }

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

  const renderContent = () => {
    switch (selectedView) {
      case "html":
        return (
          <div
            dangerouslySetInnerHTML={{
              __html: DOMPurify.sanitize(combinedHtml),
            }}
          />
        );
      case "markdown":
        return (
          <ReactMarkdown
            className="cyan-2"
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex, remarkGfm]}
          >
            {combinedMarkdown}
          </ReactMarkdown>
        );
      case "json":
        return chunk.segments.map((segment: Segment, segmentIndex: number) => (
          <div key={segmentIndex}>
            <ReactJson
              src={segment}
              theme="monokai"
              displayDataTypes={false}
              enableClipboard={false}
              style={{ backgroundColor: "transparent" }}
            />
          </div>
        ));
      case "structured":
        return chunk.segments.map((segment: Segment, segmentIndex: number) => (
          <div key={segmentIndex} className="structured-segment">
            <div className="segment-type">{segment.segment_type}</div>
            <div className="segment-content">
              {segment.content || segment.markdown || segment.html}
            </div>
          </div>
        ));
    }
  };

  return (
    <div
      className="segment-chunk"
      ref={ref}
      style={{ maxWidth: containerWidth }}
    >
      <div className="segment-content">{renderContent()}</div>
    </div>
  );
});
