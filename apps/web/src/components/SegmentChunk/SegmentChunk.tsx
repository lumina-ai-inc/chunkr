import { useState } from "react";
import { Segment } from "../../types";
import { Text, Flex } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";
import * as Accordion from "@radix-ui/react-accordion";
import { ChevronDownIcon } from "@radix-ui/react-icons";
import "./SegmentChunk.css";

export const SegmentChunk = ({ segment }: { segment: Segment }) => {
  const [markdownSelected, setMarkdownSelected] = useState<boolean>(true);
  // const [isOpen, setIsOpen] = useState<boolean>(false);

  return (
    <Accordion.Root
      className="AccordionRoot"
      type="single"
      collapsible
      defaultValue="item-1"
    >
      <Accordion.Item className="AccordionItem" value="item-1">
        <Accordion.Header className="AccordionHeader">
          <Accordion.Trigger asChild className="AccordionTrigger">
            <Flex
              justify="between"
              align="center"
              className="w-full px-6 h-16 border-b-2 AccordionTriggerFlex"
            >
              <Flex align="center" gap="4">
                <ChevronDownIcon
                  className="AccordionChevron"
                  aria-hidden
                ></ChevronDownIcon>
                <Text size="5" weight="medium" className="cyan-3">
                  {segment.type}
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
        <Accordion.Content className="AccordionContent px-6 py-6">
          <div className="AccordionContentText">
            {markdownSelected ? (
              <Text className="text-[#e5e7eb]">{segment.text}</Text>
            ) : (
              <Text className="text-[#e5e7eb]">JSON VIEW</Text>
            )}
          </div>
        </Accordion.Content>
      </Accordion.Item>
    </Accordion.Root>
  );
};

/* <div className="border-2">
<div className="flex h-16 border-b-2 justify-between px-6 items-center">
  <Text size="5" weight="medium" className="cyan-3">
    {segment.type}
  </Text>
  <Flex gap="5">
    <BetterButton
      active={markdownSelected}
      onClick={() => {
        setMarkdownSelected(true);
      }}
    >
      <Text size="3" weight="regular">
        Markdown{" "}
      </Text>
    </BetterButton>
    <BetterButton
      active={!markdownSelected}
      onClick={() => {
        setMarkdownSelected(false);
      }}
    >
      <Text size="3" weight="regular">
        JSON
      </Text>
    </BetterButton>
  </Flex>
</div>
<div className="px-6 py-6">
  {markdownSelected && (
    <Text className="text-[#e5e7eb]">{segment.text}</Text>
  )}
  {!markdownSelected && <Text className="text-[#e5e7eb]">JSON VIEW</Text>}
</div>
</div> */
