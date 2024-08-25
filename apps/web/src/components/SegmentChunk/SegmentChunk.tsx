import { useState } from "react";
import { Segment } from "../../types";
import { Text } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";


export const SegmentChunk = ({
  segment,
}: {
  segment: Segment;
}) => {
  const [markdownSelected, setMarkdownSelected] = useState<boolean>(true);

  return (
    <div className="border-2">
      <div className="flex h-16 border-b-2 justify-between px-6 items-center">
        <Text size="6" className="cyan-3">{segment.type}</Text>
        <div className="flex space-x-2">
          <BetterButton active={markdownSelected} onClick={() => {
            setMarkdownSelected(true)
          }}>
            <Text>Markdown </Text>
          </BetterButton>
          <BetterButton active={!markdownSelected} onClick={() => {
            setMarkdownSelected(false)
          }}>
            <Text>JSON</Text>
          </BetterButton>
        </div>
      </div>
      <div className="px-6 py-6">
        {markdownSelected &&
          <Text className="text-[#e5e7eb]">{segment.text}</Text>
        }
        {!markdownSelected &&
          <Text className="text-[#e5e7eb]">
            JSON VIEW
          </Text>
        }
      </div>
    </div>
  );
};
