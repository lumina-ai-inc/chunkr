import React, { useRef, useState, useEffect } from "react";
import { Flex, ScrollArea } from "@radix-ui/themes";
import { SegmentChunk } from "../../components/SegmentChunk/SegmentChunk";
import { PDF } from "../../components/PDF/PDF";
import Header from "../../components/Header/Header";
import boundingBoxes from "../../../bounding_boxes.json";
import { BoundingBoxes, Chunk } from "../../models/chunk.model";
import { retrieveFileContent } from "../../services/chunkMyDocs";
import "./Viewer.css";

export const Viewer = () => {
  const typedBoundingBoxes: BoundingBoxes = boundingBoxes as BoundingBoxes;
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [scrollAreaWidth, setScrollAreaWidth] = useState<number>(0);
  const [pdfWidth, setPdfWidth] = useState<number>(50); // Initial width percentage
  const isDraggingRef = useRef<boolean>(false);
  const [pdfContent, setPdfContent] = useState<string | null>(null);

  useEffect(() => {
    const updateWidth = () => {
      if (scrollAreaRef.current) {
        const calculatedWidth = window.innerWidth * ((100 - pdfWidth) / 100);
        setScrollAreaWidth(calculatedWidth);
      }
    };

    updateWidth();
    window.addEventListener("resize", updateWidth);

    return () => {
      window.removeEventListener("resize", updateWidth);
    };
  }, [pdfWidth]);

  useEffect(() => {
    const fetchPdfContent = async () => {
      const urlParams = new URLSearchParams(window.location.search);
      const fileUrl = urlParams.get("file_url");
      if (fileUrl) {
        try {
          const content = await retrieveFileContent(fileUrl);
          setPdfContent(content);
        } catch (error) {
          console.error("Error fetching PDF content:", error);
        }
      }
    };

    fetchPdfContent();
  }, []);

  const handleMouseDown = () => {
    isDraggingRef.current = true;
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDraggingRef.current) {
      const newWidth = (e.clientX / window.innerWidth) * 100;
      setPdfWidth(Math.max(20, Math.min(80, newWidth))); // Limit between 20% and 80%
    }
  };

  useEffect(() => {
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  if (!pdfContent) {
    return <div>Loading...</div>;
  }

  console.log(pdfContent);

  return (
    <Flex direction="column" width="100%">
      <Flex
        width="100%"
        direction="column"
        style={{ boxShadow: "0px 12px 12px 0px rgba(0, 0, 0, 0.12)" }}
      >
        <Header py="24px" px="24px" download={true} home={false} />
      </Flex>
      <Flex
        direction="row"
        width="100%"
        style={{ borderTop: "2px solid var(--cyan-12)" }}
        onMouseMove={handleMouseMove}
      >
        <Flex
          width={`${pdfWidth}%`}
          direction="column"
          style={{
            borderRight: "2px solid var(--cyan-12)",
            position: "relative",
          }}
          ref={scrollAreaRef}
        >
          <PDF content={pdfContent} />
          <div
            style={{
              position: "absolute",
              right: "-14px",
              top: "calc(50% - 16px)",
              width: "24px",
              height: "32px",
              cursor: "col-resize",
              borderRadius: "4px",
              backgroundColor: "var(--cyan-5)",
              zIndex: 100,
            }}
            onMouseDown={handleMouseDown}
          />
        </Flex>
        <ScrollArea
          scrollbars="vertical"
          type="always"
          style={{
            height: "calc(100vh - 90px)",
            width: `${100 - pdfWidth}%`,
          }}
        >
          <Flex width="70%" height="100%" direction="column" p="24px" gap="9">
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
