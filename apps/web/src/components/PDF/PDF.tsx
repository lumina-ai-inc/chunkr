import { useEffect, useRef, useState } from "react";
import { pdfjs, Document, Page } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";

import type { PDFDocumentProxy } from "pdfjs-dist";
import { ScrollArea } from "@radix-ui/themes";
import { Chunk } from "../../models/chunk.model";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

const options = {
  cMapUrl: "/cmaps/",
  standardFontDataUrl: "/standard_fonts/",
};

const maxWidth = 800;

type PDFFile = string | File | null;
type SegmentType =
  | "Text"
  | "Title"
  | "Table"
  | "Section header"
  | "Picture"
  | "Page footer"
  | "Page header"
  | "List item"
  | "Formula"
  | "Footnote"
  | "Caption";

export function PDF({ content }: { content: Chunk[] }) {
  const [file, setFile] = useState<PDFFile>("example.pdf");
  const [numPages, setNumPages] = useState<number>();

  const segments = content;

  function onDocumentLoadSuccess({
    numPages: nextNumPages,
  }: PDFDocumentProxy): void {
    setNumPages(nextNumPages);
  }

  return (
    <Document
      file={file}
      onLoadSuccess={onDocumentLoadSuccess}
      options={options}
    >
      <ScrollArea
        scrollbars="both"
        type="always"
        style={{
          height: "calc(100vh - 90px)",
        }}
      >
        <div className="flex flex-col items-center space-y-2" style={{}}>
          {Array.from(new Array(numPages), (_el, index) => {
            return <CurrentPage index={index} segments={segments} />;
          })}
        </div>
      </ScrollArea>
    </Document>
  );
}

function CurrentPage({ index, segments }: { index: number; segments: any }) {
  // Get all segments for this page
  const pageNumber = index + 1;
  const thingsToRender = segments.flatMap((segment) => {
    const inSegmentChunks = segment.segments.filter((chunk) => {
      return chunk.page_number == pageNumber;
    });

    return inSegmentChunks;
  });

  const pageRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    const pageDiv = pageRef.current.querySelector(".react-pdf__Page");
    setContainerWidth(pageDiv.width);
  });

  return (
    <div ref={pageRef} className="flex relative items-center">
      <Page
        key={`page_${index + 1}`}
        pageNumber={index + 1}
        width={containerWidth ? Math.min(containerWidth, maxWidth) : maxWidth}
        onClick={(event) => {
          console.log("hi", event);
        }}
      >
        {thingsToRender.map((segment: any, j: number) => {
          const scaledLeft = `${(segment.left / segment.page_width) * 100}%`;
          const scaledTop = `${(segment.top / segment.page_height) * 100}%`;
          const scaledHeight = `${(segment.height / segment.page_height) * 100}%`;
          const scaledWidth = `${(segment.width / segment.page_width) * 100}%`;

          let color;
          const t: SegmentType = segment.type;

          if (t == "Text") {
            color = "--teal-10";
          } else if (t == "Table") {
            color = "--orange-9";
          } else if (t == "Title") {
            color = "--amber-8";
          } else if (t == "Picture") {
            color = "--pink-10";
          } else if (t == "Formula") {
            color = "--cyan-8";
          } else if (t == "Caption") {
            color = "--jade-10";
          } else if (t == "Footnote") {
            color = "--pink-10";
          } else if (t == "List item") {
            color = "--bronze-10";
          } else if (t == "Page footer") {
            color = "--red-12";
          } else if (t == "Page header") {
            color = "--violet-8";
          } else if (t == "Section header") {
            color = "--yellow-7";
          } else {
            color = "border-black";
          }

          const [hovered, setHovered] = useState(false);
          return (
            <div
              className="page visible absolute z-50 border-2"
              style={{
                width: scaledWidth,
                height: scaledHeight,
                left: scaledLeft,
                top: scaledTop,
                borderColor: `var(${color})`,
              }}
              key={j}
            >
              <div className="w-full h-full bg-red-500 hidden"></div>
            </div>
          );
        })}
      </Page>
    </div>
  );
}
