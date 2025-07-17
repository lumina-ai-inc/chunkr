import { useRef, useState, useCallback, useMemo, useEffect, memo } from "react";
import { Flex, Text } from "@radix-ui/themes";
import { SegmentChunk } from "../SegmentChunk/SegmentChunk";
import { PDF } from "../PDF/PDF";
import "./Viewer.css";
import Loader from "../../pages/Loader/Loader";
import { TaskResponse, Chunk } from "../../models/taskResponse.model";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { debounce } from "lodash";
import React from "react";
import { ExcelViewerProps } from "../../models/excelViewer.model";

const MemoizedPDF = memo(PDF);

const CHUNK_LOAD_SIZE = 5; // Number of chunks to load at a time
const PAGE_CHUNK_SIZE = 5; // Number of pages to load at a time

interface ViewerProps {
  task: TaskResponse;
  hideHeader?: boolean; // New prop to hide header
  rightPanelContent?: React.ReactNode; // Custom content for the right panel
}

export default function Viewer({ task, rightPanelContent }: ViewerProps) {
  const output = task.output;
  const memoizedOutput = useMemo(() => output, [output]);
  const chunkRefs = useRef<(HTMLDivElement | null)[]>([]);

  const [activeSegment, setActiveSegment] = useState<{
    chunkId: string;
    segmentId: string;
  } | null>(null);

  const [loadedChunks, setLoadedChunks] = useState(CHUNK_LOAD_SIZE);
  const [loadedPages, setLoadedPages] = useState(PAGE_CHUNK_SIZE);
  const [numPages, setNumPages] = useState<number>();

  const scrollableContentRef = useRef<HTMLDivElement>(null);
  const pdfContainerRef = useRef<HTMLDivElement>(null);

  const scrollToSegment = useCallback(
    (chunkId: string, segmentId: string) => {
      // Find the chunk and segment
      const chunk = output?.chunks.find((c) => c.chunk_id === chunkId);
      const segment = chunk?.segments.find((s) => s.segment_id === segmentId);

      if (!chunk || !segment) return;

      const targetPage = segment.page_number;
      const chunkIndex =
        output?.chunks.findIndex((c) => c.chunk_id === chunkId) ?? -1;

      // First, ensure content is loaded
      const needsMorePages = targetPage && targetPage > loadedPages;
      const needsMoreChunks = chunkIndex >= loadedChunks;

      if (needsMorePages) {
        setLoadedPages(
          Math.ceil(targetPage / PAGE_CHUNK_SIZE) * PAGE_CHUNK_SIZE
        );
      }

      if (needsMoreChunks) {
        setLoadedChunks(
          Math.ceil((chunkIndex + 1) / CHUNK_LOAD_SIZE) * CHUNK_LOAD_SIZE
        );
      }

      // Wait for content to potentially load and render before scrolling
      // Use requestAnimationFrame for smoother timing with browser rendering
      requestAnimationFrame(() => {
        setActiveSegment({ chunkId, segmentId });

        // Scroll PDF container using the ref
        const pdfContainer = pdfContainerRef.current;
        if (pdfContainer) {
          // Use more specific selector within the container
          const targetSegmentElement = pdfContainer.querySelector(
            `.flex[data-page-number="${targetPage}"] [data-chunk-id="${chunkId}"][data-segment-id="${segmentId}"]`
          );

          if (targetSegmentElement) {
            const containerHeight = pdfContainer.clientHeight;
            const segmentRect = targetSegmentElement.getBoundingClientRect();
            const containerRect = pdfContainer.getBoundingClientRect();
            const relativeTop = segmentRect.top - containerRect.top;

            const targetPosition =
              pdfContainer.scrollTop + relativeTop - containerHeight * 0.3;

            pdfContainer.scrollTo({
              top: targetPosition,
              behavior: "smooth",
            });
          } else {
            // Fallback or attempt scroll to page if segment not found (might be on unloaded page)
            const pageElement = pdfContainer.querySelector(
              `.flex[data-page-number="${targetPage}"]`
            );
            if (pageElement) {
              const containerHeight = pdfContainer.clientHeight;
              const pageRect = pageElement.getBoundingClientRect();
              const containerRect = pdfContainer.getBoundingClientRect();
              const relativeTop = pageRect.top - containerRect.top;
              const targetPosition =
                pdfContainer.scrollTop + relativeTop - containerHeight * 0.1; // Scroll closer to top for page
              pdfContainer.scrollTo({
                top: targetPosition,
                behavior: "smooth",
              });
            }
          }
        }

        // Scroll text content using the ref
        const scrollableContent = scrollableContentRef.current;
        if (scrollableContent) {
          // Use more specific selector within the container
          const textSegmentElement = scrollableContent.querySelector(
            `.segment-item[data-chunk-id="${chunkId}"][data-segment-id="${segmentId}"]`
          );
          if (textSegmentElement) {
            const containerHeight = scrollableContent.clientHeight;
            const segmentRect = textSegmentElement.getBoundingClientRect();
            const containerRect = scrollableContent.getBoundingClientRect();
            const relativeTop = segmentRect.top - containerRect.top;

            const targetPosition =
              scrollableContent.scrollTop + relativeTop - containerHeight * 0.2;

            scrollableContent.scrollTo({
              top: targetPosition,
              behavior: "smooth",
            });
          } else {
            // Fallback: Scroll towards the chunk if segment isn't rendered yet
            const chunkElement = scrollableContent.querySelector(
              `.segment-chunk[data-chunk-id="${chunkId}"]`
            );
            if (chunkElement) {
              const containerHeight = scrollableContent.clientHeight;
              const chunkRect = chunkElement.getBoundingClientRect();
              const containerRect = scrollableContent.getBoundingClientRect();
              const relativeTop = chunkRect.top - containerRect.top;
              const targetPosition =
                scrollableContent.scrollTop +
                relativeTop -
                containerHeight * 0.1; // Scroll closer to top for chunk
              scrollableContent.scrollTo({
                top: targetPosition,
                behavior: "smooth",
              });
            }
          }
        }
      });
    },
    [
      output?.chunks,
      loadedPages,
      loadedChunks,
      pdfContainerRef,
      scrollableContentRef,
    ]
  );

  // Update the handler for PDF segment clicks
  const handlePDFSegmentClick = useCallback(
    (chunkId: string, segmentId: string) => {
      const chunkIndex =
        output?.chunks.findIndex((c) => c.chunk_id === chunkId) ?? -1;

      // Ensure we load enough chunks
      if (chunkIndex >= loadedChunks) {
        setLoadedChunks(
          Math.ceil((chunkIndex + 1) / CHUNK_LOAD_SIZE) * CHUNK_LOAD_SIZE
        );

        // Wait for next render cycle when chunks are loaded

        requestAnimationFrame(() => {
          scrollToSegment(chunkId, segmentId);
        });
      } else {
        scrollToSegment(chunkId, segmentId);
      }
    },
    [loadedChunks, scrollToSegment, output?.chunks]
  );

  /* Keep ONE debounced fn for the lifetime of the component */
  const debouncedScrollHandler = useRef<ReturnType<typeof debounce>>();

  /* Stable listener that never changes */
  const onScroll = useCallback((e: Event) => {
    debouncedScrollHandler.current!(e.target as HTMLDivElement);
  }, []);

  /* Create the debounce only once */
  useEffect(() => {
    debouncedScrollHandler.current = debounce((target: HTMLDivElement) => {
      // <— your old handleScrollLogic body
      const { scrollTop, scrollHeight, clientHeight } = target;
      // use functional updates so we do NOT depend on loadedChunks/pages
      if (target.classList.contains("scrollable-content")) {
        setLoadedChunks((prev) => {
          const totalChunks = output?.chunks.length ?? 0;
          return scrollHeight - scrollTop <= clientHeight * 1.8
            ? Math.min(prev + CHUNK_LOAD_SIZE, totalChunks)
            : prev;
        });
      } else if (target.classList.contains("pdf-container")) {
        setLoadedPages((prev) => {
          const totalPages = numPages ?? 0;
          return scrollHeight - scrollTop <= clientHeight * 1.8
            ? Math.min(prev + PAGE_CHUNK_SIZE, totalPages)
            : prev;
        });
      }
    }, 200);
  }, [output?.chunks, numPages]);

  /* Attach listener once */
  useEffect(() => {
    const content = scrollableContentRef.current;
    const pdf = pdfContainerRef.current;

    content?.addEventListener("scroll", onScroll);
    pdf?.addEventListener("scroll", onScroll);

    return () => {
      debouncedScrollHandler.current?.cancel();
      content?.removeEventListener("scroll", onScroll);
      pdf?.removeEventListener("scroll", onScroll);
    };
  }, [onScroll]);

  if (!output) {
    return <Loader />;
  }

  return (
    <Flex direction="column" width="100%" height="100%">
      <PanelGroup
        direction="horizontal"
        style={{ backgroundColor: "var(--bg-0)" }}
      >
        <Panel
          defaultSize={50}
          minSize={20}
          style={{ backgroundColor: "var(--bg-0)" }}
        >
          <div className="scrollable-content" ref={scrollableContentRef}>
            {output.chunks.length === 0 ? (
              <Text
                size="4"
                weight="medium"
                style={{ color: "rgba(255, 255, 255, 0.8)" }}
              >
                No content available for this PDF.
              </Text>
            ) : (
              <>
                {output.chunks
                  .slice(0, loadedChunks)
                  .map((chunk: Chunk, index: number) => (
                    <SegmentChunk
                      key={chunk.chunk_id}
                      chunk={chunk}
                      chunkId={chunk.chunk_id}
                      chunkIndex={index}
                      ref={(el) => (chunkRefs.current[index] = el)}
                      onSegmentClick={scrollToSegment}
                      activeSegment={activeSegment}
                    />
                  ))}
                {loadedChunks < output.chunks.length && (
                  <div
                    className="loading-more-chunks"
                    style={{
                      textAlign: "center",
                      padding: "20px",
                      color: "rgba(255, 255, 255, 0.6)",
                    }}
                  >
                    Loading more chunks...
                  </div>
                )}
              </>
            )}
          </div>
        </Panel>

        <PanelResizeHandle
          className="resize-handle"
          hitAreaMargins={{ coarse: 15, fine: 5 }}
        />

        <Panel
          defaultSize={50}
          minSize={20}
          style={{ backgroundColor: "#0d0d0d" }}
        >
          {rightPanelContent
            ?
              React.isValidElement(rightPanelContent)
              ? React.cloneElement(
                  rightPanelContent as React.ReactElement<ExcelViewerProps>,
                  { activeSegment, onRangeClick: scrollToSegment }
                )
              : rightPanelContent
            : memoizedOutput &&
              memoizedOutput.pdf_url && (
                <MemoizedPDF
                  containerRef={pdfContainerRef}
                  content={memoizedOutput.chunks}
                  inputFileUrl={memoizedOutput.pdf_url}
                  onSegmentClick={handlePDFSegmentClick}
                  activeSegment={activeSegment}
                  loadedPages={loadedPages}
                  onLoadSuccess={(pages) => setNumPages(pages)}
                />
              )}
        </Panel>
      </PanelGroup>
    </Flex>
  );
}
