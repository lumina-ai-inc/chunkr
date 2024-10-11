import { useState, useMemo, forwardRef } from "react";
import { Chunk, Segment } from "../../models/chunk.model";
import { Text, Flex } from "@radix-ui/themes";
import * as Accordion from "@radix-ui/react-accordion";
import Badge from "../Badge";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import "./SegmentChunk.css";
import DOMPurify from "dompurify";

import ReactMarkdown from "react-markdown";
import ReactJson from "react-json-view";
import BetterButton from "../BetterButton/BetterButton";

export const SegmentChunk = forwardRef<
  HTMLDivElement,
  {
    chunk: Chunk;
    chunkIndex: number;
    containerWidth: number;
  }
>(({ chunk, containerWidth }, ref) => {
  const [selectedView, setSelectedView] = useState<
    "html" | "markdown" | "json"
  >("html");
  const accordionWidth = containerWidth - 48;

  const segmentTypeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    chunk.segments.forEach((segment) => {
      counts[segment.segment_type] = (counts[segment.segment_type] || 0) + 1;
    });
    return counts;
  }, [chunk.segments]);

  const segmentTypeBadges = useMemo(() => {
    return Object.entries(segmentTypeCounts).map(([type, count]) => (
      <Badge
        key={type}
        className={`segment-badge ${type.toLowerCase().replace(" ", "-")}`}
      >
        <Text size="1" weight="medium" className="white">
          {`${type} x${count}`}
        </Text>
      </Badge>
    ));
  }, [segmentTypeCounts]);

  const combinedMarkdown = useMemo(() => {
    return chunk.segments
      .map((segment) => {
        const textContent = segment.content || "";
        if (segment.segment_type === "Table"  && segment.html?.startsWith("<span class=")) {
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
        if (segment.segment_type === "Table" && segment.html?.startsWith("<span class=")) {
          return `<br><img src="${segment.image}" />`;
        }
        return segment.html || "";
      })
      .filter(Boolean)
      .join("");
  }, [chunk.segments]);

  const handleCopy = () => {
    let textToCopy = "";
    if (selectedView === "html") {
      textToCopy = combinedHtml;
    } else if (selectedView === "markdown") {
      textToCopy = combinedMarkdown;
    } else if (selectedView === "json") {
      textToCopy = JSON.stringify(chunk.segments, null, 2);
    }
    navigator.clipboard
      .writeText(textToCopy)
      .then(() => {
        console.log("Copied to clipboard");
      })
      .catch((err) => {
        console.error("Failed to copy: ", err);
      });
  };

  return (
    <div ref={ref}>
      <Accordion.Root
        className="AccordionRoot"
        type="single"
        collapsible
        defaultValue="item-1"
        style={{ width: `${accordionWidth}px` }}
      >
        <Accordion.Item className="AccordionItem" value="item-1">
          <Accordion.Header className="AccordionHeader">
            <Accordion.Trigger asChild className="AccordionTrigger">
              <Flex
                justify="between"
                align="center"
                className="w-full AccordionTriggerFlex"
              >
                <Flex align="center" gap="4">
                  <div className="AccordionChevron" aria-hidden>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                    >
                      <rect
                        width="24"
                        height="24"
                        transform="matrix(-1 0 0 -1 24 24)"
                      />
                      <path
                        fillRule="evenodd"
                        clipRule="evenodd"
                        d="M18.9836 9.85285C18.6814 9.53051 18.1751 9.51419 17.8528 9.81637L11.9999 15.3034L6.14715 9.81637C5.82475 9.51419 5.31851 9.53051 5.01627 9.85285C4.71419 10.1752 4.73051 10.6815 5.05275 10.9836L11.4528 16.9836C11.7605 17.2721 12.2394 17.2721 12.5471 16.9836L18.9471 10.9836C19.2694 10.6815 19.2858 10.1752 18.9836 9.85285Z"
                        fill="hsla(0, 0%, 100%, 0.9)"
                      />
                    </svg>
                  </div>
                  <Flex gap="4" wrap="wrap">
                    {segmentTypeBadges}
                  </Flex>
                </Flex>

                <Flex gap="4" align="center">
                  {["html", "markdown", "json"].map((view) => (
                    <Flex key={view} align="center" gap="2">
                      <Text
                        size="2"
                        weight="medium"
                        style={{
                          color:
                            selectedView === view
                              ? "hsla(0, 0%, 100%, 0.9)"
                              : "#8A9BA8",
                          cursor: "pointer",
                        }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedView(view as "html" | "markdown" | "json");
                        }}
                      >
                        {view === "markdown" ? "Markdown" : view.toUpperCase()}
                      </Text>
                      {selectedView === view && (
                        <BetterButton
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCopy();
                          }}
                        >
                          <Text
                            weight="medium"
                            className="white"
                            style={{ fontSize: "10px" }}
                          >
                            Copy
                          </Text>
                        </BetterButton>
                      )}
                    </Flex>
                  ))}
                </Flex>
              </Flex>
            </Accordion.Trigger>
          </Accordion.Header>
          <Accordion.Content className="AccordionContent">
            <div className="AccordionContentText">
              {selectedView === "html" && (
                <div
                  dangerouslySetInnerHTML={{
                    __html: DOMPurify.sanitize(combinedHtml),
                  }}
                />
              )}
              {selectedView === "markdown" && (
                <ReactMarkdown
                  className="cyan-2"
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex, remarkGfm]}
                >
                  {combinedMarkdown}
                </ReactMarkdown>
              )}
              {selectedView === "json" &&
                chunk.segments.map((segment: Segment, segmentIndex: number) => (
                  <div key={segmentIndex}>
                    <ReactJson
                      src={segment}
                      theme="monokai"
                      displayDataTypes={false}
                      enableClipboard={false}
                      style={{ backgroundColor: "transparent" }}
                    />
                  </div>
                ))}
            </div>
          </Accordion.Content>
        </Accordion.Item>
      </Accordion.Root>
    </div>
  );
});
