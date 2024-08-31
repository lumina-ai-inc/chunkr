import { useRef, useState } from "react";
import { pdfjs, Document, Page } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import { ScrollArea } from "@radix-ui/themes";
import { Chunk, Segment, SegmentType } from "../../models/chunk.model";
import "./Pdf.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

const options = {
  cMapUrl: "/cmaps/",
  standardFontDataUrl: "/standard_fonts/",
};

const segmentColors: Record<SegmentType, string> = {
  Text: "--teal-10",
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
  Text: "--teal-4",
  Table: "--orange-4",
  Title: "--blue-4",
  Picture: "--pink-4",
  Formula: "--amber-4",
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
}: {
  content: Chunk[];
  inputFileUrl: string;
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
            <CurrentPage key={index} index={index} segments={content} />
          ))}
        </div>
      </ScrollArea>
    </Document>
  );
}

function CurrentPage({
  index,
  segments,
}: {
  index: number;
  segments: Chunk[];
}) {
  const pageNumber = index + 1;
  const thingsToRender = segments.flatMap((segment) =>
    segment.segments.filter((chunk) => chunk.page_number === pageNumber)
  );

  const pageRef = useRef<HTMLDivElement>(null);

  return (
    <div ref={pageRef} className="flex relative items-center">
      <Page
        key={`page_${pageNumber}`}
        pageNumber={pageNumber}
        width={800}
        onClick={(event) => console.log("Page clicked", event)}
      >
        {thingsToRender.map((segment, j) => (
          <SegmentOverlay key={j} segment={segment} />
        ))}
      </Page>
    </div>
  );
}

function SegmentOverlay({ segment }: { segment: Segment }) {
  const scaledLeft = `${(segment.left / segment.page_width) * 100}%`;
  const scaledTop = `${(segment.top / segment.page_height) * 100}%`;
  const scaledHeight = `${(segment.height / segment.page_height) * 100}%`;
  const scaledWidth = `${(segment.width / segment.page_width) * 100}%`;

  const baseColor =
    segmentColors[segment.type as SegmentType] || "--border-black";
  const lightColor =
    segmentLightColors[segment.type as SegmentType] || "--border-black";

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
        {segment.type}
      </div>
    </div>
  );
}
