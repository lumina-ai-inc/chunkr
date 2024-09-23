import { useRef, useState } from "react";
import { pdfjs, Document, Page } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import { ScrollArea } from "@radix-ui/themes";
import { Chunk, Segment, SegmentType } from "../../models/chunk.model";
import "./PDF.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

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
  const scaledLeft = `${(segment.bbox.top_left[0] / segment.page_width) * 100}%`;
  const scaledTop = `${(segment.bbox.top_left[1] / segment.page_height) * 100}%`;
  const scaledHeight = `${((segment.bbox.bottom_right[1] - segment.bbox.top_left[1]) / segment.page_height) * 100}%`;
  const scaledWidth = `${((segment.bbox.bottom_right[0] - segment.bbox.top_left[0]) / segment.page_width) * 100}%`;

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
    </div>
  );
}
