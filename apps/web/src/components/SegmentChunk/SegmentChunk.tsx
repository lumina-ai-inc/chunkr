import { useState, useMemo } from "react";
import { Chunk } from "../../models/chunk.model";
import { Text, Flex, ScrollArea } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";
import * as Accordion from "@radix-ui/react-accordion";
import { ChevronDownIcon } from "@radix-ui/react-icons";
import "./SegmentChunk.css";

import ReactMarkdown from "react-markdown";
import ReactJson from "react-json-view";

export const SegmentChunk = ({
  chunk,
  containerWidth,
}: {
  chunk: Chunk;
  containerWidth: number;
}) => {
  const [markdownSelected, setMarkdownSelected] = useState<boolean>(true);
  const accordionWidth = containerWidth - 48;

  const segmentTypeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    chunk.segments.forEach((segment) => {
      counts[segment.type] = (counts[segment.type] || 0) + 1;
    });
    return counts;
  }, [chunk.segments]);

  const segmentTypeString = useMemo(() => {
    return Object.entries(segmentTypeCounts)
      .map(([type, count]) => `${type} x${count}`)
      .join("\u00A0\u00A0|\u00A0\u00A0");
  }, [segmentTypeCounts]);

  return (
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
                <ChevronDownIcon
                  className="AccordionChevron"
                  aria-hidden
                ></ChevronDownIcon>
                <Text size="4" weight="medium" className="cyan-5">
                  {segmentTypeString}
                </Text>
              </Flex>

              <Flex gap="5">
                <BetterButton
                  active={markdownSelected}
                  onClick={(e) => {
                    e.stopPropagation();
                    setMarkdownSelected(true);
                  }}
                >
                  <Text size="3" weight="regular">
                    Markdown
                  </Text>
                </BetterButton>
                <BetterButton
                  active={!markdownSelected}
                  onClick={(e) => {
                    e.stopPropagation();
                    setMarkdownSelected(false);
                  }}
                >
                  <Text size="3" weight="regular">
                    JSON
                  </Text>
                </BetterButton>
              </Flex>
            </Flex>
          </Accordion.Trigger>
        </Accordion.Header>
        <Accordion.Content className="AccordionContent">
          <ScrollArea>
            <div className="AccordionContentText">
              {markdownSelected ? (
                <ReactMarkdown className="cyan-2">
                  {chunk.markdown}
                </ReactMarkdown>
              ) : (
                <ReactJson
                  src={chunk}
                  theme="monokai"
                  displayDataTypes={false}
                  enableClipboard={false}
                  style={{ backgroundColor: "transparent" }}
                />
              )}
            </div>
          </ScrollArea>
        </Accordion.Content>
      </Accordion.Item>
    </Accordion.Root>
  );
};
