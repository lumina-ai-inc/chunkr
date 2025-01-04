import { useRef, useState, useEffect, useMemo, memo } from "react";
import { pdfjs, Document, Page } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import { Flex } from "@radix-ui/themes";
import {
  Chunk,
  Segment,
  SegmentType,
  OCRResult,
  BoundingBox,
} from "../../models/chunk.model";
import "./PDF.css";
import { debounce } from "lodash";

declare global {
  interface PromiseConstructor {
    withResolvers<T>(): {
      promise: Promise<T>;
      resolve: (value: T | PromiseLike<T>) => void;
      reject: (reason?: unknown) => void;
    };
  }
}

Promise.withResolvers = function <T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/legacy/build/pdf.worker.min.mjs`;

const options = {
  cMapUrl: "/cmaps/",
  standardFontDataUrl: "/standard_fonts/",
};

const segmentColors: Record<SegmentType, string> = {
  Text: "--jade-10",
  Table: "--orange-9",
  Title: "--blue-9",
  Picture: "--pink-10",
  Formula: "--amber-8",
  Caption: "--crimson-8",
  Footnote: "--pink-10",
  "List item": "--bronze-10",
  "Page footer": "--red-12",
  "Page header": "--violet-9",
  "Section header": "--cyan-8",
};

const segmentLightColors: Record<SegmentType, string> = {
  Text: "--jade-4",
  Table: "--orange-4",
  Title: "--blue-4",
  Picture: "--pink-4",
  Formula: "--amber-3",
  Caption: "--crimson-2",
  Footnote: "--pink-4",
  "List item": "--bronze-4",
  "Page footer": "--red-4",
  "Page header": "--violet-4",
  "Section header": "--cyan-2",
};

const MemoizedOCRBoundingBoxes = memo(OCRBoundingBoxes);

const MemoizedSegmentOverlay = memo(SegmentOverlay);

const MemoizedCurrentPage = memo(CurrentPage);

const PAGE_CHUNK_SIZE = 20; // Number of pages to load at once

export const PDF = memo(
  ({
    content,
    inputFileUrl,
    onSegmentClick,
    activeSegment,
  }: {
    content: Chunk[];
    inputFileUrl: string;
    onSegmentClick: (chunkIndex: number, segmentIndex: number) => void;
    activeSegment?: { chunkIndex: number; segmentIndex: number } | null;
  }) => {
    const [numPages, setNumPages] = useState<number>();
    const [loadedPages, setLoadedPages] = useState<number>(PAGE_CHUNK_SIZE);
    const [isDocumentLoaded, setIsDocumentLoaded] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const [pdfWidth, setPdfWidth] = useState(800);

    const debouncedSetPdfWidth = useMemo(
      () => debounce((width: number) => setPdfWidth(width), 80),
      []
    );

    useEffect(() => {
      if (!containerRef.current) return;

      const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const width = Math.max(
            Math.min(Math.round(entry.contentRect.width - 32), 800),
            200
          );
          debouncedSetPdfWidth(width);
        }
      });

      resizeObserver.observe(containerRef.current);
      return () => {
        debouncedSetPdfWidth.cancel();
        resizeObserver.disconnect();
      };
    }, [debouncedSetPdfWidth]);

    // Handle scroll to load more pages
    useEffect(() => {
      const container = containerRef.current;
      if (!container) return;

      const handleScroll = debounce(() => {
        const { scrollTop, scrollHeight, clientHeight } = container;
        const scrolledToBottom = scrollHeight - scrollTop <= clientHeight * 1.5;

        if (scrolledToBottom && loadedPages < (numPages || 0)) {
          setLoadedPages((prev) =>
            Math.min(prev + PAGE_CHUNK_SIZE, numPages || 0)
          );
        }
      }, 100);

      container.addEventListener("scroll", handleScroll);
      return () => {
        container.removeEventListener("scroll", handleScroll);
        handleScroll.cancel();
      };
    }, [loadedPages, numPages]);

    useEffect(() => {
      const handleScrollToPage = (e: CustomEvent) => {
        const { pageNumber } = e.detail;
        const container = containerRef.current;
        if (container && pageNumber && isDocumentLoaded) {
          const pageElement = container.querySelector(
            `[data-page-number="${pageNumber}"]`
          );
          if (pageElement) {
            pageElement.scrollIntoView({ behavior: "smooth" });
          }
        }
      };

      window.addEventListener(
        "scroll-to-page",
        handleScrollToPage as EventListener
      );
      return () => {
        window.removeEventListener(
          "scroll-to-page",
          handleScrollToPage as EventListener
        );
      };
    }, [isDocumentLoaded]);

    // Memoize the page array with pagination
    const pages = useMemo(
      () =>
        Array.from(
          new Array(Math.min(loadedPages, numPages || 0)),
          (_, index) => (
            <MemoizedCurrentPage
              key={index}
              index={index}
              segments={content}
              onSegmentClick={onSegmentClick}
              width={pdfWidth}
              activeSegment={activeSegment}
            />
          )
        ),
      [loadedPages, numPages, content, onSegmentClick, pdfWidth, activeSegment]
    );

    return (
      <div ref={containerRef} className="pdf-container">
        <Document
          file={inputFileUrl}
          onLoadSuccess={(document: pdfjs.PDFDocumentProxy) => {
            setNumPages(document.numPages);
            setLoadedPages(Math.min(PAGE_CHUNK_SIZE, document.numPages));
            setIsDocumentLoaded(true);
          }}
          loading={<div className="loading">Loading PDF...</div>}
          error={<div className="error">Failed to load PDF</div>}
          options={options}
        >
          <Flex
            direction="column"
            align="center"
            justify="center"
            height="100%"
            width="100%"
          >
            {pages}
            {loadedPages < (numPages || 0) && (
              <div className="loading-more-pages">Loading more pages...</div>
            )}
          </Flex>
        </Document>
      </div>
    );
  }
);

function CurrentPage({
  index,
  segments,
  onSegmentClick,
  width,
  activeSegment,
}: {
  index: number;
  segments: Chunk[];
  onSegmentClick: (chunkIndex: number, segmentIndex: number) => void;
  width: number;
  activeSegment?: { chunkIndex: number; segmentIndex: number } | null;
}) {
  const pageNumber = index + 1;

  const pageSegments = useMemo(
    () =>
      segments.flatMap((chunk, chunkIndex) =>
        chunk.segments
          .filter((segment) => segment.page_number === pageNumber)
          .map((segment, segmentIndex) => (
            <MemoizedSegmentOverlay
              key={`${chunkIndex}-${segmentIndex}`}
              segment={segment}
              chunkIndex={chunkIndex}
              segmentIndex={segmentIndex}
              onClick={() => onSegmentClick(chunkIndex, segmentIndex)}
              isActive={activeSegment?.chunkIndex === chunkIndex}
            />
          ))
      ),
    [segments, pageNumber, onSegmentClick, activeSegment]
  );

  return (
    <div className="flex relative items-center">
      <Page key={`page_${pageNumber}`} pageNumber={pageNumber} width={width}>
        {pageSegments}
      </Page>
    </div>
  );
}

function SegmentOverlay({
  segment,
  onClick,
  chunkIndex,
  segmentIndex,
  isActive,
}: {
  segment: Segment;
  onClick: () => void;
  chunkIndex: number;
  segmentIndex: number;
  isActive?: boolean;
}) {
  const [isHovered, setIsHovered] = useState(false);

  const style = useMemo(
    () => ({
      width: `${(segment.bbox.width / segment.page_width) * 100}%`,
      height: `${(segment.bbox.height / segment.page_height) * 100}%`,
      left: `${(segment.bbox.left / segment.page_width) * 100}%`,
      top: `${(segment.bbox.top / segment.page_height) * 100}%`,
      borderColor: `var(${segmentColors[segment.segment_type as SegmentType] || "--border-black"})`,
      backgroundColor:
        isActive || isHovered
          ? `color-mix(in srgb, var(${segmentLightColors[segment.segment_type as SegmentType] || "--border-black"}) 30%, transparent)`
          : "transparent",
      transition: "background-color 0.2s ease-in-out",
    }),
    [segment, isActive, isHovered]
  );

  const handleClick = () => {
    onClick();
    // Dispatch event to highlight corresponding segment chunk
    window.dispatchEvent(
      new CustomEvent("highlight-segment", {
        detail: { chunkIndex, segmentIndex },
      })
    );
  };

  return (
    <div
      className={`segment visible absolute z-50 border-2 ${isActive ? "active" : ""}`}
      style={style}
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="w-full h-full bg-red-500 hidden"></div>
      <div
        className="segment-overlay"
        style={{
          borderColor: `var(${segmentColors[segment.segment_type as SegmentType] || "--border-black"}) !important`,
          color: `var(${segmentColors[segment.segment_type as SegmentType] || "--border-black"}) !important`,
          backgroundColor: isHovered
            ? `color-mix(in srgb, var(${segmentLightColors[segment.segment_type as SegmentType] || "--border-black"}) 100%, transparent)`
            : "transparent",
          opacity: "1 !important",
        }}
      >
        {segment.segment_type}
      </div>
      {isHovered && segment.ocr && (
        <MemoizedOCRBoundingBoxes
          ocr={segment.ocr}
          segmentBBox={segment.bbox}
          segmentType={segment.segment_type as SegmentType}
        />
      )}
    </div>
  );
}

function OCRBoundingBoxes({
  ocr,
  segmentBBox,
  segmentType,
}: {
  ocr: OCRResult[];
  segmentBBox: BoundingBox;
  segmentType: SegmentType;
}) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <>
      {ocr.map((result, index) => {
        const style = {
          position: "absolute" as const,
          left: `${(result.bbox.left / segmentBBox.width) * 100}%`,
          top: `${(result.bbox.top / segmentBBox.height) * 100}%`,
          width: `${(result.bbox.width / segmentBBox.width) * 100}%`,
          height: `${(result.bbox.height / segmentBBox.height) * 100}%`,
          border: `1px solid var(${segmentColors[segmentType] || "--border-black"})`,
          zIndex: 40,
        };

        return (
          <div
            key={index}
            style={style}
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            {hoveredIndex === index && (
              <Flex
                style={{
                  position: "absolute",
                  zIndex: 9999,
                  left: 0,
                  top: 0,
                  transform: "translateY(-100%)",
                  backgroundColor: `var(${segmentColors[segmentType] || "--border-black"}) !important`,
                  color: "white !important",
                  padding: "2px 4px",
                  borderRadius: "2px",
                  fontSize: "12px",
                  lineHeight: "1.2",
                  whiteSpace: "nowrap",
                  pointerEvents: "none",
                  isolation: "isolate",
                }}
              >
                {result.text}
              </Flex>
            )}
          </div>
        );
      })}
    </>
  );
}

PDF.displayName = "PDF";
