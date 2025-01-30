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
} from "../../models/taskResponse.model";
import "./PDF.css";
import { debounce } from "lodash";
import Loader from "../../pages/Loader/Loader";

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
  ListItem: "--bronze-10",
  PageFooter: "--red-12",
  PageHeader: "--violet-9",
  SectionHeader: "--cyan-8",
  Page: "--gray-8",
};

const segmentLightColors: Record<SegmentType, string> = {
  Text: "--jade-4",
  Table: "--orange-4",
  Title: "--blue-4",
  Picture: "--pink-4",
  Formula: "--amber-3",
  Caption: "--crimson-2",
  Footnote: "--pink-4",
  ListItem: "--bronze-4",
  PageFooter: "--red-4",
  PageHeader: "--violet-4",
  SectionHeader: "--cyan-2",
  Page: "--gray-3",
};

const MemoizedOCRBoundingBoxes = memo(OCRBoundingBoxes);

const MemoizedSegmentOverlay = memo(SegmentOverlay);

const MemoizedCurrentPage = memo(CurrentPage);

export const PDF = memo(
  ({
    content,
    inputFileUrl,
    onSegmentClick,
    activeSegment,
    loadedPages,
    onLoadSuccess,
    structureExtractionView = false,
  }: {
    content: Chunk[];
    inputFileUrl: string;
    onSegmentClick: (chunkId: string, segmentId: string) => void;
    activeSegment?: { chunkId: string; segmentId: string } | null;
    loadedPages: number;
    onLoadSuccess?: (numPages: number) => void;
    structureExtractionView?: boolean;
  }) => {
    const [numPages, setNumPages] = useState<number>();
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

    return (
      <div ref={containerRef} className="pdf-container">
        <Document
          file={inputFileUrl}
          onLoadSuccess={(document: pdfjs.PDFDocumentProxy) => {
            setNumPages(document.numPages);
            onLoadSuccess?.(document.numPages);
          }}
          loading={<Loader />}
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
            {Array.from(
              new Array(Math.min(loadedPages, numPages || 0)),
              (_, index) => (
                <MemoizedCurrentPage
                  key={index}
                  index={index}
                  segments={content}
                  onSegmentClick={onSegmentClick}
                  width={pdfWidth}
                  activeSegment={activeSegment}
                  structureExtractionView={structureExtractionView}
                />
              )
            )}
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
  structureExtractionView,
}: {
  index: number;
  segments: Chunk[];
  onSegmentClick: (chunkId: string, segmentId: string) => void;
  width: number;
  activeSegment?: { chunkId: string; segmentId: string } | null;
  structureExtractionView: boolean;
}) {
  const pageNumber = index + 1;

  const pageSegments = useMemo(
    () =>
      !structureExtractionView
        ? segments.flatMap((chunk) =>
            chunk.segments
              .filter((segment) => segment.page_number === pageNumber)
              .map((segment) => (
                <MemoizedSegmentOverlay
                  key={`${chunk.chunk_id}-${segment.segment_id}`}
                  segment={segment}
                  chunkId={chunk.chunk_id}
                  segmentId={segment.segment_id}
                  onClick={() =>
                    onSegmentClick(chunk.chunk_id, segment.segment_id)
                  }
                  isActive={
                    activeSegment?.chunkId === chunk.chunk_id &&
                    activeSegment?.segmentId === segment.segment_id
                  }
                />
              ))
          )
        : [],
    [
      segments,
      pageNumber,
      onSegmentClick,
      activeSegment,
      structureExtractionView,
    ]
  );

  return (
    <div className="flex relative items-center" data-page-number={pageNumber}>
      <Page key={`page_${pageNumber}`} pageNumber={pageNumber} width={width}>
        {pageSegments}
      </Page>
    </div>
  );
}

function SegmentOverlay({
  segment,
  onClick,
  chunkId,
  segmentId,
  isActive,
}: {
  segment: Segment;
  onClick: () => void;
  chunkId: string;
  segmentId: string;
  isActive?: boolean;
}) {
  const [isHovered, setIsHovered] = useState(false);

  const style = useMemo(
    () => ({
      width: `${(segment.bbox.width / segment.page_width) * 100}%`,
      height: `${(segment.bbox.height / segment.page_height) * 100}%`,
      left: `${(segment.bbox.left / segment.page_width) * 100}%`,
      top: `${(segment.bbox.top / segment.page_height) * 100}%`,
      borderColor: `var(${
        segmentColors[segment.segment_type as SegmentType] || "--border-black"
      })`,
      backgroundColor:
        isActive || isHovered
          ? `color-mix(in srgb, var(${
              segmentLightColors[segment.segment_type as SegmentType] ||
              "--border-black"
            }) 30%, transparent)`
          : "transparent",
      transition: "background-color 0.2s ease-in-out",
    }),
    [segment, isActive, isHovered]
  );

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onClick();
  };

  return (
    <div
      className={`segment visible absolute z-50 border-2 ${
        isActive ? "active" : ""
      }`}
      style={style}
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      data-chunk-id={chunkId}
      data-segment-id={segmentId}
    >
      <div className="w-full h-full bg-red-500 hidden"></div>
      <div
        className="segment-overlay"
        style={{
          borderColor: `var(${
            segmentColors[segment.segment_type as SegmentType] ||
            "--border-black"
          }) !important`,
          color: `var(${
            segmentColors[segment.segment_type as SegmentType] ||
            "--border-black"
          }) !important`,
          backgroundColor: isHovered
            ? `color-mix(in srgb, var(${
                segmentLightColors[segment.segment_type as SegmentType] ||
                "--border-black"
              }) 100%, transparent)`
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
          border: `1px solid var(${
            segmentColors[segmentType] || "--border-black"
          })`,
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
                  left: -1,
                  top: -4,
                  transform: "translateY(-100%)",
                  backgroundColor: `color-mix(in srgb, var(${
                    segmentColors[segmentType] || "--border-black"
                  }) 90%, transparent) !important`,
                  color: `var(--color-background) !important`,
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
