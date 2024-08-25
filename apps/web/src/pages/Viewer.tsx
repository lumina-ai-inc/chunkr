import { Flex, ScrollArea } from "@radix-ui/themes";
import { SegmentChunk } from "../components/SegmentChunk/SegmentChunk";
import { PDF } from "../components/PDF/PDF";
import Header from "../components/Header/Header";
import boundingBoxes from "../../bounding_boxes.json";
import { BoundingBoxes, Chunk } from "../models/chunk.model";
import { useRef, useState, useEffect } from "react";

export const Viewer = () => {
  const typedBoundingBoxes: BoundingBoxes = boundingBoxes as BoundingBoxes;
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [scrollAreaWidth, setScrollAreaWidth] = useState<number>(0);

  useEffect(() => {
    const updateWidth = () => {
      if (scrollAreaRef.current) {
        setScrollAreaWidth(scrollAreaRef.current.offsetWidth);
      }
    };

    updateWidth();
    window.addEventListener("resize", updateWidth);

    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  return (
    <Flex direction="column" width="100%">
      <Flex
        width="100%"
        direction="column"
        style={{ boxShadow: "0px 12px 12px 0px rgba(0, 0, 0, 0.12)" }}
      >
        <Header py="24px" px="24px" />
      </Flex>
      <Flex
        direction="row"
        width="100%"
        style={{ borderTop: "2px solid var(--cyan-12)" }}
      >
        <Flex width="100%" direction="column">
          <PDF />
        </Flex>
        <ScrollArea
          scrollbars="vertical"
          type="always"
          style={{
            height: "calc(100vh - 90px)",
            padding: "20px"
          }}
          ref={scrollAreaRef}
        >
          <Flex width="100%" height="100%" direction="column" p="24px" gap="7">
            {typedBoundingBoxes.map((chunk: Chunk, index: number) => (
              <SegmentChunk
                key={index}
                chunk={chunk}
                containerWidth={scrollAreaWidth}
              />
            ))}
          </Flex>
        </ScrollArea>
      </Flex>
    </Flex>
  );
};
