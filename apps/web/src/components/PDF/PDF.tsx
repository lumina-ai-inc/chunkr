import { useState, useEffect, useMemo, memo } from "react";
import { pdfjs, Document, Page } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import { Flex } from "@radix-ui/themes";
import {
  Chunk,
  Segment,
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
    containerRef,
  }: {
    content: Chunk[];
    inputFileUrl: string;
    onSegmentClick: (chunkId: string, segmentId: string) => void;
    activeSegment?: { chunkId: string; segmentId: string } | null;
    loadedPages: number;
    onLoadSuccess?: (numPages: number) => void;
    structureExtractionView?: boolean;
    containerRef: React.RefObject<HTMLDivElement>;
  }) => {
    const [numPages, setNumPages] = useState<number>();
    const [pdfWidth, setPdfWidth] = useState(800);

    const debouncedSetPdfWidth = useMemo(
      () => debounce((width: number) => setPdfWidth(width), 80),
      []
    );

    useEffect(() => {
      const observedContainer = containerRef.current;
      if (!observedContainer) return;

      const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const width = Math.max(
            Math.min(Math.round(entry.contentRect.width - 32), 800),
            200
          );
          debouncedSetPdfWidth(width);
        }
      });

      resizeObserver.observe(observedContainer);
      return () => {
        debouncedSetPdfWidth.cancel();
        resizeObserver.disconnect();
      };
    }, [debouncedSetPdfWidth, containerRef]);

    const segmentsByPage = useMemo(() => {
      const map = new Map<number, { chunkId: string; segment: Segment }[]>();
      if (!structureExtractionView) {
        content.forEach((chunk) => {
          chunk.segments.forEach((segment) => {
            if (segment.page_number) {
              const pageNum = segment.page_number;
              if (!map.has(pageNum)) {
                map.set(pageNum, []);
              }
              map
                .get(pageNum)
                ?.push({ chunkId: chunk.chunk_id, segment: segment });
            }
          });
        });
      }
      return map;
    }, [content, structureExtractionView]);

    return (
      <div ref={containerRef} className="pdf-container">
        <Document
          file={inputFileUrl}
          onLoadSuccess={(document: pdfjs.PDFDocumentProxy) => {
            setNumPages(document.numPages);
            onLoadSuccess?.(document.numPages);
          }}
          loading={
            <div style={{ width: "100%", height: "calc(100vh - 132px)" }}>
              <Loader />
            </div>
          }
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
                  pageSegmentsData={segmentsByPage.get(index + 1) || []}
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
  pageSegmentsData,
  onSegmentClick,
  width,
  activeSegment,
  structureExtractionView,
}: {
  index: number;
  pageSegmentsData: { chunkId: string; segment: Segment }[];
  onSegmentClick: (chunkId: string, segmentId: string) => void;
  width: number;
  activeSegment?: { chunkId: string; segmentId: string } | null;
  structureExtractionView: boolean;
}) {
  const pageNumber = index + 1;

  const pageSegmentElements = useMemo(
    () =>
      pageSegmentsData.map(({ chunkId, segment }) => (
        <MemoizedSegmentOverlay
          key={`${chunkId}-${segment.segment_id}`}
          segment={segment}
          chunkId={chunkId}
          segmentId={segment.segment_id}
          onClick={() => onSegmentClick(chunkId, segment.segment_id)}
          isActive={
            activeSegment?.chunkId === chunkId &&
            activeSegment?.segmentId === segment.segment_id
          }
        />
      )),
    [pageSegmentsData, onSegmentClick, activeSegment]
  );

  return (
    <div className="flex relative items-center" data-page-number={pageNumber}>
      <Page
        key={`page_${pageNumber}`}
        pageNumber={pageNumber}
        width={width}
        renderAnnotationLayer={false}
        renderTextLayer={false}
      >
        {!structureExtractionView && pageSegmentElements}
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
  const style = useMemo(
    () => ({
      width: `${(segment.bbox.width / segment.page_width) * 100}%`,
      height: `${(segment.bbox.height / segment.page_height) * 100}%`,
      left: `${(segment.bbox.left / segment.page_width) * 100}%`,
      top: `${(segment.bbox.top / segment.page_height) * 100}%`,
      transition:
        "background-color 0.05s ease-in-out, border-color 0.05s ease-in-out",
    }),
    [segment]
  );

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onClick();
  };

  const segmentTypeClass = `type-${segment.segment_type.toLowerCase()}`;

  return (
    <div
      className={`segment visible absolute z-50 border-2 ${segmentTypeClass} ${
        isActive ? "active" : ""
      }`}
      style={style}
      onClick={handleClick}
      data-chunk-id={chunkId}
      data-segment-id={segmentId}
    >
      <div className="w-full h-full bg-red-500 hidden"></div>
      <div className="segment-overlay">{segment.segment_type}</div>
      {segment.ocr && (
        <MemoizedOCRBoundingBoxes
          ocr={segment.ocr}
          segmentBBox={segment.bbox}
        />
      )}
    </div>
  );
}

function OCRBoundingBoxes({
  ocr,
  segmentBBox,
}: {
  ocr: OCRResult[];
  segmentBBox: BoundingBox;
}) {
  return (
    <>
      {ocr.map((result, index) => {
        const style = {
          position: "absolute" as const,
          left: `${(result.bbox.left / segmentBBox.width) * 100}%`,
          top: `${(result.bbox.top / segmentBBox.height) * 100}%`,
          width: `${(result.bbox.width / segmentBBox.width) * 100}%`,
          height: `${(result.bbox.height / segmentBBox.height) * 100}%`,
          zIndex: 40,
        };

        return (
          <div key={index} className="ocr-box" style={style}>
            <Flex className="ocr-tooltip">{result.text}</Flex>
          </div>
        );
      })}
    </>
  );
}

PDF.displayName = "PDF";
