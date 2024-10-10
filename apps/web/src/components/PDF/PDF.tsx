import { useRef, useState } from "react";
import { pdfjs, Document, Page } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import { ScrollArea, Box, Text } from "@radix-ui/themes";
import {
  Chunk,
  Segment,
  SegmentType,
  OCRResult,
  BoundingBox,
} from "../../models/chunk.model";
import "./PDF.css";

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

export function PDF({
  content,
  inputFileUrl,
  onSegmentClick,
}: {
  content: Chunk[];
  inputFileUrl: string;
  onSegmentClick: (chunkIndex: number, segmentIndex: number) => void;
}) {
  const [numPages, setNumPages] = useState<number>();

  function onDocumentLoadSuccess(document: pdfjs.PDFDocumentProxy): void {
    setNumPages(document.numPages);
  }

  return (
    <Document
      file={inputFileUrl}
      onLoadSuccess={onDocumentLoadSuccess}
      options={options}
    >
      <ScrollArea
        scrollbars="both"
        type="always"
        style={{ height: "calc(100vh - 90px)" }}
      >
        <div className="flex flex-col items-center space-y-2">
          {Array.from(new Array(numPages), (_, index) => (
            <CurrentPage
              key={index}
              index={index}
              segments={content}
              onSegmentClick={onSegmentClick}
            />
          ))}
        </div>
      </ScrollArea>
    </Document>
  );
}

function CurrentPage({
  index,
  segments,
  onSegmentClick,
}: {
  index: number;
  segments: Chunk[];
  onSegmentClick: (chunkIndex: number, segmentIndex: number) => void;
}) {
  const pageNumber = index + 1;
  const thingsToRender = segments.flatMap((chunk, chunkIndex) =>
    chunk.segments
      .filter((segment) => segment.page_number === pageNumber)
      .map((segment, segmentIndex) => ({ segment, chunkIndex, segmentIndex }))
  );

  const pageRef = useRef<HTMLDivElement>(null);

  return (
    <div ref={pageRef} className="flex relative items-center">
      <Page key={`page_${pageNumber}`} pageNumber={pageNumber} width={800}>
        {thingsToRender.map(({ segment, chunkIndex, segmentIndex }) => (
          <SegmentOverlay
            key={`${chunkIndex}-${segmentIndex}`}
            segment={segment}
            onClick={() => onSegmentClick(chunkIndex, segmentIndex)}
            chunkIndex={chunkIndex}
            segmentIndex={segmentIndex}
          />
        ))}
      </Page>
    </div>
  );
}

function SegmentOverlay({
  segment,
  onClick,
}: {
  segment: Segment;
  onClick: () => void;
  chunkIndex: number;
  segmentIndex: number;
}) {
  const [isHovered, setIsHovered] = useState(false);
  const scaledLeft = `${(segment.bbox.left / segment.page_width) * 100}%`;
  const scaledTop = `${(segment.bbox.top / segment.page_height) * 100}%`;
  const scaledHeight = `${(segment.bbox.height / segment.page_height) * 100}%`;
  const scaledWidth = `${(segment.bbox.width / segment.page_width) * 100}%`;

  const baseColor =
    segmentColors[segment.segment_type as SegmentType] || "--border-black";
  const lightColor =
    segmentLightColors[segment.segment_type as SegmentType] || "--border-black";

  return (
    <div
      className="segment visible absolute z-50 border-2"
      style={{
        width: scaledWidth,
        height: scaledHeight,
        left: scaledLeft,
        top: scaledTop,
        borderColor: `var(${baseColor})`,
      }}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="w-full h-full bg-red-500 hidden"></div>
      <div
        className="segment-overlay"
        style={{
          border: `2px solid var(${baseColor})`,
          backgroundColor: `var(${lightColor})`,
          color: `var(${baseColor})`,
          fontSize: "12px",
        }}
      >
        {segment.segment_type}
      </div>
      {isHovered && segment.ocr && (
        <OCRBoundingBoxes
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

  const segmentWidth = segmentBBox.width;
  const segmentHeight = segmentBBox.height;

  const baseColor = segmentColors[segmentType] || "--border-black";
  const lightColor = segmentLightColors[segmentType] || "--border-black";

  return (
    <>
      {ocr.map((result, index) => {
        const relativeLeft = result.bbox.left;
        const relativeTop = result.bbox.top;
        const width = result.bbox.width;
        const height = result.bbox.height;

        const scaledRelativeLeft = `${(relativeLeft / segmentWidth) * 100}%`;
        const scaledRelativeTop = `${(relativeTop / segmentHeight) * 100}%`;
        const scaledWidth = `${(width / segmentWidth) * 100}%`;
        const scaledHeight = `${(height / segmentHeight) * 100}%`;

        return (
          <Box
            key={index}
            style={{
              position: "absolute",
              left: scaledRelativeLeft,
              top: scaledRelativeTop,
              width: scaledWidth,
              height: scaledHeight,
              border: `1px solid var(${baseColor})`,
              zIndex: 40,
            }}
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            {hoveredIndex === index && (
              <Text
                size="1"
                style={{
                  position: "absolute",
                  bottom: "100%",
                  left: "0",
                  backgroundColor: `var(${lightColor})`,
                  color: `var(${baseColor})`,
                  padding: "2px 4px",
                  borderRadius: "2px",
                  zIndex: 50,
                  width: "fit-content",
                  whiteSpace: "nowrap",
                  marginBottom: "2px",
                }}
              >
                {result.text}
              </Text>
            )}
          </Box>
        );
      })}
    </>
  );
}
