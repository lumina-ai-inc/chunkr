import {
  forwardRef,
  memo,
  useCallback,
  useMemo,
  useState,
  useEffect,
} from "react";
import {
  Chunk,
  Segment,
  SegmentType,
  Configuration,
} from "../../models/taskResponse.model";
import { GenerationStrategy } from "../../models/taskConfig.model";
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
      selectedView: "html" | "markdown" | "json";
      onSegmentClick?: (chunkId: string, segmentId: string) => void;
      activeSegment?: { chunkId: string; segmentId: string } | null;
      config: Configuration;
    }
  >(
    (
      { chunk, chunkId, selectedView, onSegmentClick, activeSegment, config },
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
          const mode = segmentDisplayModes[segment.segment_id] || {
            showJson: false,
            showLLM: false,
            showImage: false,
          };

          // Just return the content directly without the segment-item wrapper
          if (mode.showJson) {
            return <MemoizedJson segment={segment} />;
          }

          if (mode.showLLM && segment.llm) {
            return <MemoizedMarkdown content={segment.llm} />;
          }

          if (mode.showImage && segment.image) {
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

          // Determine if the segment type is potentially special
          const isPotentiallySpecialType = [
            SegmentType.Picture,
            SegmentType.Table,
            SegmentType.Formula,
          ].includes(segment.segment_type);

          // Determine if the processing mode is 'llm' for this type
          let isLlmProcessing = false;

          // SAFELY grab the processing options for the current segment_type
          const processingConfig =
            config.segment_processing?.[segment.segment_type];

          if (processingConfig) {
            isLlmProcessing =
              processingConfig.html === GenerationStrategy.LLM ||
              processingConfig.markdown === GenerationStrategy.LLM;
          }

          // Apply special styling only if type is special AND processing is NOT 'llm'
          const isSpecialSegment = isPotentiallySpecialType && isLlmProcessing;

          // Determine the appropriate header text based on type
          let specialSegmentHeaderText = "";
          if (isSpecialSegment) {
            switch (segment.segment_type) {
              case SegmentType.Picture:
                specialSegmentHeaderText = "Image description";
                break;
              case SegmentType.Table:
                specialSegmentHeaderText = "Rendered table";
                break;
              case SegmentType.Formula:
                specialSegmentHeaderText = "Rendered formula";
                break;
            }
          }

          const typeClass = `type-${segment.segment_type
            .replace(/([a-z])([A-Z])/g, "$1-$2") // camel → kebab
            .toLowerCase()}`;

          const renderSegmentContent = () => {
            const mode = segmentDisplayModes[segment.segment_id] || {
              showJson: false,
              showLLM: false,
              showImage: false,
            };

            if (mode.showJson) {
              return <MemoizedJson segment={segment} />;
            }

            if (mode.showLLM && segment.llm) {
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

            if (mode.showImage && segment.image) {
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

          const currentButtonMode = segmentDisplayModes[segment.segment_id] || {
            showJson: false,
            showLLM: false,
            showImage: false,
          };

          return (
            <div
              key={segmentIndex}
              className={`segment-item
                          ${isActive ? "active" : ""}
                          ${isSpecialSegment ? "special-segment" : ""}
                          ${typeClass}`}
              data-chunk-id={chunkId}
              data-segment-id={segment.segment_id}
              data-segment-type={segment.segment_type}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                if (!isActive) {
                  onSegmentClick?.(chunkId, segment.segment_id);
                }
              }}
              style={{ width: "100%" }}
            >
              <div className="scroll-x">
                {isSpecialSegment && !isActive && specialSegmentHeaderText && (
                  <Flex
                    align="center"
                    gap="1"
                    className="special-segment-header"
                    style={{ marginBottom: "8px" }}
                  >
                    {segment.segment_type === SegmentType.Picture && (
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <g clip-path="url(#clip0_304_23709)">
                          <path
                            d="M20.25 4.75H3.75C3.19772 4.75 2.75 5.19772 2.75 5.75V18.25C2.75 18.8023 3.19772 19.25 3.75 19.25H20.25C20.8023 19.25 21.25 18.8023 21.25 18.25V5.75C21.25 5.19772 20.8023 4.75 20.25 4.75Z"
                            stroke="#ffffffb4"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M21 18.91L16.54 11.75L13.79 16.14"
                            stroke="#ffffffb4"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M15.7002 19.2502L9.23023 8.75L2.99023 18.9002"
                            stroke="#ffffffb4"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </g>
                        <defs>
                          <clipPath id="clip0_304_23709">
                            <rect width="24" height="24" fill="white" />
                          </clipPath>
                        </defs>
                      </svg>
                    )}
                    {segment.segment_type === SegmentType.Table && (
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M3 9H21M3 15H21M9 9L9 20M15 9L15 20M6.2 20H17.8C18.9201 20 19.4802 20 19.908 19.782C20.2843 19.5903 20.5903 19.2843 20.782 18.908C21 18.4802 21 17.9201 21 16.8V7.2C21 6.0799 21 5.51984 20.782 5.09202C20.5903 4.71569 20.2843 4.40973 19.908 4.21799C19.4802 4 18.9201 4 17.8 4H6.2C5.0799 4 4.51984 4 4.09202 4.21799C3.71569 4.40973 3.40973 4.71569 3.21799 5.09202C3 5.51984 3 6.07989 3 7.2V16.8C3 17.9201 3 18.4802 3.21799 18.908C3.40973 19.2843 3.71569 19.5903 4.09202 19.782C4.51984 20 5.07989 20 6.2 20Z"
                          stroke="#ffffffb4"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    )}
                    {segment.segment_type === SegmentType.Formula && (
                      <svg
                        fill="#ffffffb4"
                        xmlns="http://www.w3.org/2000/svg"
                        width="14"
                        height="14"
                        viewBox="0 0 52 52"
                        enableBackground="new 0 0 52 52"
                        xmlSpace="preserve"
                      >
                        <path
                          d="M30.2,3.2c-0.8-0.6-1.9-0.9-3.2-0.9c-0.3,0-0.6,0-0.9,0.1c-3.5,0.5-5.8,3.8-7.2,6.8
                   c-0.6,1.4-1.2,2.9-1.7,4.4c-0.3,0.7-0.5,1.4-0.8,2.1c0,0.1-0.4,1.5-0.5,1.5c0,0-3.2,0-3.2,0l0,0h-0.4c-0.5,0-0.9,0.4-0.9,0.9
                   c0,0.5,0.4,0.9,0.9,0.9h3.1l-1.8,7.5C12,34.7,9.7,44.2,9.1,45.9c-0.6,1.7-1.4,2.6-2.5,2.6c-0.2,0-0.4,0-0.5-0.1
                   c-0.1-0.1-0.2-0.2-0.2-0.4c0-0.1,0.1-0.4,0.3-0.7c0.2-0.3,0.3-0.7,0.3-1c0-0.7-0.2-1.2-0.7-1.6c-0.4-0.4-0.9-0.6-1.5-0.6
                   c-0.6,0-1.1,0.2-1.6,0.6c-0.5,0.4-0.7,1-0.7,1.7c0,1,0.4,1.8,1.2,2.5c0.8,0.7,1.8,1,3.1,1c2.1,0,4.1-1,5.5-2.6
                   c0.9-1,1.5-2.2,2.1-3.5c1.8-3.9,2.7-8.2,3.7-12.3c1-4.2,2-8.4,2.9-12.7H24c0.5,0,0.9-0.4,0.9-0.9c0-0.5-0.4-0.9-0.9-0.9H24v0h-3.1
                   c1.7-6.6,3.7-11.1,4.1-11.8c0.7-1.1,1.4-1.7,2.1-1.7c0.3,0,0.5,0.1,0.6,0.2c0.1,0.2,0.1,0.3,0.1,0.4c0,0.1-0.1,0.3-0.3,0.7
                   c-0.2,0.4-0.3,0.8-0.3,1.2c0,0.6,0.2,1,0.6,1.4c0.4,0.4,0.9,0.6,1.5,0.6c0.6,0,1.1-0.2,1.5-0.6c0.4-0.4,0.6-1,0.6-1.7
                   C31.5,4.6,31.1,3.8,30.2,3.2z"
                        />
                        <path
                          d="M46.1,23.2c1.3,0,3.8-1,3.8-4.4c0-3.3-2.4-3.5-3.1-3.5c-1.5,0-2.9,1.1-4.2,3.3c-1.3,2.3-2.7,4.7-2.7,4.7l0,0
                   c-0.3-1.6-0.6-2.9-0.7-3.5c-0.3-1.4-1.9-4.4-5.2-4.4c-3.3,0-6.3,1.9-6.3,1.9l0,0c-0.6,0.4-0.9,1-0.9,1.7c0,1.1,0.9,2,2,2
                   c0.3,0,0.6-0.1,0.9-0.2l0,0c0,0,2.5-1.4,3,0c0.2,0.4,0.3,0.9,0.4,1.4c0.6,2.2,1.2,4.7,1.7,7l-2.2,3.1c0,0-2.4-0.9-3.7-0.9
                   s-3.8,1-3.8,4.4s2.4,3.5,3.1,3.5c1.5,0,2.9-1.1,4.2-3.3c1.3-2.3,2.7-4.7,2.7-4.7c0.4,2,0.8,3.7,1,4.4c0.8,2.3,2.7,3.7,5.3,3.7
                   c0,0,2.6,0,5.7-1.7c0.7-0.3,1.3-1,1.3-1.9c0-1.1-0.9-2-2-2c-0.3,0-0.6,0.1-0.9,0.2l0,0c0,0-2.2,1.2-2.9,0.3c-0.5-1-1-2.4-1.3-4
                   c-0.3-1.5-0.7-3.2-1-4.9l2.2-3.2C42.4,22.3,44.9,23.2,46.1,23.2z"
                        />
                      </svg>
                    )}
                    <Text
                      size="1"
                      weight="medium"
                      style={{ color: "rgba(255,255,255,0.7)" }}
                    >
                      {specialSegmentHeaderText}
                    </Text>
                  </Flex>
                )}
                {isActive && (
                  <Flex mb="2" mt="2" gap="4">
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
                    {currentButtonMode.showJson ||
                    currentButtonMode.showLLM ||
                    currentButtonMode.showImage ? (
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
                          active={currentButtonMode.showJson}
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
                            active={currentButtonMode.showLLM}
                            onClick={() =>
                              handleSegmentDisplayMode(
                                segment.segment_id,
                                "llm"
                              )
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
                            active={currentButtonMode.showImage}
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
