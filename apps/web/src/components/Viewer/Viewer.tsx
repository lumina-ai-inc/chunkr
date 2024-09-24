import React, { useRef, useState, useEffect, useCallback } from "react";
import { Flex, ScrollArea, Text } from "@radix-ui/themes";
import { SegmentChunk } from "../SegmentChunk/SegmentChunk";
import { PDF } from "../PDF/PDF";
import Header from "../Header/Header";
import { Chunk } from "../../models/chunk.model";
import "./Viewer.css";
import Loader from "../../pages/Loader/Loader";
import { TaskResponse } from "../../models/task.model";
import TaskCard from "../TaskCard/TaskCard";

interface ViewerProps {
  // eslint-disable-next-line
  output: any;
  inputFileUrl: string;
  task: TaskResponse;
}

export const Viewer = ({ output, inputFileUrl, task }: ViewerProps) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [scrollAreaWidth, setScrollAreaWidth] = useState<number>(0);
  const [pdfWidth, setPdfWidth] = useState<number>(50);
  const isDraggingRef = useRef<boolean>(false);

  const chunkRefs = useRef<(HTMLDivElement | null)[]>([]);

  const scrollToSegment = useCallback((chunkIndex: number) => {
    const chunkElement = chunkRefs.current[chunkIndex];
    if (chunkElement) {
      const container = chunkElement.closest(".rt-ScrollAreaViewport");
      if (container) {
        const containerRect = container.getBoundingClientRect();
        const chunkRect = chunkElement.getBoundingClientRect();
        const scrollTop =
          chunkRect.top - containerRect.top + container.scrollTop - 24;
        container.scrollTo({
          top: scrollTop,
          behavior: "smooth",
        });
      }
    }
  }, []);

  useEffect(() => {
    const updateWidth = () => {
      if (scrollAreaRef.current) {
        const calculatedWidth = window.innerWidth * ((100 - pdfWidth) / 100);
        setScrollAreaWidth(calculatedWidth);
      }
    };

    // Initial update
    updateWidth();

    // Schedule additional updates
    const timeouts = [100, 500, 1000].map((delay) =>
      setTimeout(updateWidth, delay)
    );

    // Add resize event listener
    window.addEventListener("resize", updateWidth);

    return () => {
      // Clear timeouts and remove event listener on cleanup
      timeouts.forEach(clearTimeout);
      window.removeEventListener("resize", updateWidth);
    };
  }, [pdfWidth]);

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

  // TODO: Convert to show error message
  if (!output) {
    return <Loader />;
  }

  return (
    <Flex direction="column" width="100%">
      <Flex
        width="100%"
        direction="column"
        style={{ boxShadow: "0px 12px 12px 0px rgba(0, 0, 0, 0.12)" }}
      >
        <Header download={true} home={false} />
      </Flex>
      <Flex
        direction="row"
        width="100%"
        style={{ borderTop: "2px solid hsla(0, 0%, 0%, 0.4)" }}
        onMouseMove={handleMouseMove}
      >
        <Flex
          width={`${pdfWidth}%`}
          direction="column"
          style={{
            borderRight: "2px solid hsla(0, 0%, 0%, 0.4)",
            position: "relative",
          }}
          ref={scrollAreaRef}
        >
          {inputFileUrl && output && (
            <PDF
              content={output}
              inputFileUrl={inputFileUrl}
              onSegmentClick={scrollToSegment}
            />
          )}
          <div
            style={{
              position: "absolute",
              right: "-14px",
              top: "calc(50% - 32px)",
              width: "24px",
              height: "32px",
              cursor: "col-resize",
              borderRadius: "4px",
              backgroundColor: "#1C1C1E",
              zIndex: 10,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
            onMouseDown={handleMouseDown}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="12"
              height="12"
              viewBox="0 0 12 12"
              fill="none"
            >
              <rect width="12" height="12" fill="white" fillOpacity="0.01" />
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M11.5999 0.799988C11.379 0.799988 11.1999 0.979074 11.1999 1.19999V4.79995H0.799994V1.19999C0.799994 0.979074 0.620909 0.799988 0.399997 0.799988C0.179085 0.799988 9.65641e-09 0.979074 0 1.19999L3.43308e-07 10.7999C3.33651e-07 11.0208 0.179084 11.1999 0.399997 11.1999C0.620909 11.1999 0.799994 11.0208 0.799994 10.7999V7.19993H11.1999V10.7999C11.1999 11.0208 11.379 11.1999 11.5999 11.1999C11.8208 11.1999 11.9999 11.0208 11.9999 10.7999V1.19999C11.9999 0.979074 11.8208 0.799988 11.5999 0.799988Z"
                fill="white"
              />
            </svg>
          </div>
        </Flex>
        <ScrollArea
          scrollbars="vertical"
          type="always"
          style={{
            height: "calc(100vh - 90px)",
            width: `${100 - pdfWidth}%`,
            borderTop: "1px solid hsla(0, 0%, 100%, 0.1)",
          }}
        >
          <Flex
            width="100%"
            height="100%"
            direction="column"
            p="24px"
            gap="8"
            align="center"
            justify="center"
          >
            {output.length === 0 ? (
              <Text
                size="4"
                weight="medium"
                style={{ color: "rgba(255, 255, 255, 0.8)" }}
              >
                No content available for this PDF.
              </Text>
            ) : (
              <>
                <Flex direction="row" width="100%">
                  <TaskCard {...task}></TaskCard>
                </Flex>
                {output.map((chunk: Chunk, chunkIndex: number) => (
                  <SegmentChunk
                    key={chunkIndex}
                    chunk={chunk}
                    chunkIndex={chunkIndex}
                    containerWidth={scrollAreaWidth}
                    ref={(el) => (chunkRefs.current[chunkIndex] = el)}
                  />
                ))}
              </>
            )}
          </Flex>
        </ScrollArea>
      </Flex>
    </Flex>
  );
};
