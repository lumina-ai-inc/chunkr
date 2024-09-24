import { useState, useMemo, forwardRef } from "react";
import { Chunk, Segment } from "../../models/chunk.model";
import { Text, Flex } from "@radix-ui/themes";
import * as Accordion from "@radix-ui/react-accordion";
import Badge from "../Badge";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "./SegmentChunk.css";

import ReactMarkdown from "react-markdown";
import ReactJson from "react-json-view";

export const SegmentChunk = forwardRef<
  HTMLDivElement,
  {
    chunk: Chunk;
    chunkIndex: number;
    containerWidth: number;
  }
>(({ chunk, containerWidth }, ref) => {
  const [markdownSelected, setMarkdownSelected] = useState<boolean>(true);
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
        const textContent = segment.text_layer || segment.ocr_text || "";
        return segment.markdown ? segment.markdown : textContent;
      })
      .filter(Boolean)
      .join("\n\n")
      .trim();
  }, [chunk.segments]);

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

                <Flex gap="4">
                  <Text
                    size="3"
                    weight="medium"
                    style={{
                      color: markdownSelected
                        ? "hsla(0, 0%, 100%, 0.9)"
                        : "#8A9BA8",
                      cursor: "pointer",
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      setMarkdownSelected(true);
                    }}
                  >
                    Markdown
                  </Text>
                  <Text
                    size="3"
                    weight="medium"
                    style={{
                      color: !markdownSelected
                        ? "hsla(0, 0%, 100%, 0.9)"
                        : "#8A9BA8",
                      cursor: "pointer",
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      setMarkdownSelected(false);
                    }}
                  >
                    JSON
                  </Text>
                </Flex>
              </Flex>
            </Accordion.Trigger>
          </Accordion.Header>
          <Accordion.Content className="AccordionContent">
            <div className="AccordionContentText">
              {markdownSelected ? (
                <ReactMarkdown
                  className="cyan-2"
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {combinedMarkdown}
                </ReactMarkdown>
              ) : (
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
                ))
              )}
            </div>
          </Accordion.Content>
        </Accordion.Item>
      </Accordion.Root>
    </div>
  );
});
