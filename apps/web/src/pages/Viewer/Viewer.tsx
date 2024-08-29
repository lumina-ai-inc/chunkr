import React, { useRef, useState, useEffect } from "react";
import { Flex, ScrollArea } from "@radix-ui/themes";
import { SegmentChunk } from "../SegmentChunk/SegmentChunk";
import { PDF } from "../PDF/PDF";
import Header from "../Header/Header";
import { BoundingBoxes, Chunk } from "../../models/chunk.model";
import { retrieveFileContent } from "../../services/chunkMyDocs";
import { Link } from "react-router-dom";
import "./Viewer.css";
import Loader from "../../pages/Loader/Loader";

interface ViewerProps {
  outputFileUrl: string;
  inputFileUrl: string;
}

export const Viewer = ({ outputFileUrl, inputFileUrl }: ViewerProps) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [scrollAreaWidth, setScrollAreaWidth] = useState<number>(0);
  const [pdfWidth, setPdfWidth] = useState<number>(50); // Initial width percentage
  const isDraggingRef = useRef<boolean>(false);
  const [pdfContent, setPdfContent] = useState<BoundingBoxes>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchContent = async () => {
      setIsLoading(true);
      try {
        if (outputFileUrl) {
          const content = await retrieveFileContent(outputFileUrl);
          setPdfContent(content);
        }
      } catch (error) {
        console.error(error);
        if (String(error).includes("403")) {
          setError("Timeout Error: Failed to fetch file - try uploading again");
        } else if (String(error).includes("Invalid JSON")) {
          setError(`Server returned invalid data: ${error}`);
        } else {
          setError(`${error}`);
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchContent();
  }, [outputFileUrl, inputFileUrl]);

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

  if (isLoading) {
    return <Loader />;
  }

  if (!pdfContent) {
    return <Loader />;
  }

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
          {inputFileUrl && pdfContent && (
            <PDF content={pdfContent} inputFileUrl={inputFileUrl} />
          )}
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
              zIndex: 10,
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
          <Flex
            width="100%"
            height="100%"
            direction="column"
            p="24px"
            gap="8"
            align="center"
            justify="center"
          >
            {isLoading ? (
              <Loader />
            ) : error ? (
              <Link to="/" style={{ textDecoration: "none" }}>
                <div
                  style={{
                    color: "var(--red-9)",
                    padding: "8px 12px",
                    border: "2px solid var(--red-12)",
                    borderRadius: "4px",
                    backgroundColor: "var(--red-7)",
                    cursor: "pointer",
                    transition: "background-color 0.2s ease",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.backgroundColor = "var(--red-8)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.backgroundColor = "var(--red-7)")
                  }
                >
                  {error}
                </div>
              </Link>
            ) : pdfContent.length === 0 ? (
              <div>No content available for this PDF.</div>
            ) : (
              pdfContent.map((chunk: Chunk, index: number) => (
                <SegmentChunk
                  key={index}
                  chunk={chunk}
                  containerWidth={scrollAreaWidth}
                />
              ))
            )}
          </Flex>
        </ScrollArea>
      </Flex>
    </Flex>
  );
};
