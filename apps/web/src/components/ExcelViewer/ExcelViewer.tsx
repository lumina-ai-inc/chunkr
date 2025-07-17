import { useState, useEffect, useCallback, useRef } from "react";
import { VariableSizeGrid as Grid } from "react-window";
import "./ExcelViewer.css";
import * as XLSX from "xlsx";
import Cell from "./Cell";
import type {
  CellHighlight,
  ImageSegment,
  ExcelViewerProps,
  CellData,
  SheetData,
  HighlightState,
} from "../../models/excelViewer.model";
import type {
  Chunk,
  Segment,
  Page,
  Cell as RawCell,
} from "../../models/taskResponse.model";

function ExcelViewer({
  taskResponse,
  activeSegment,
  onRangeClick,
}: ExcelViewerProps) {
  const [sheets, setSheets] = useState<SheetData[]>([]);
  const [activeSheet, setActiveSheet] = useState<number>(0);
  const [containerDimensions, setContainerDimensions] = useState({
    width: 1200,
    height: 600,
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<Grid>(null);
  const segmentToChunkMap = useRef<Map<string, string>>(new Map());
  const isManualSheetChange = useRef<boolean>(false);

  const [hoveredHighlight, setHoveredHighlight] =
    useState<HighlightState | null>(null);
  const [selectedHighlight, setSelectedHighlight] =
    useState<HighlightState | null>(null);
  const [hoveredFormula, setHoveredFormula] = useState<string | null>(null);

  useEffect(() => {
    if (activeSegment && !isManualSheetChange.current) {
      const targetSheetIndex = sheets.findIndex((sheet) =>
        sheet.segmentRanges.has(activeSegment.segmentId)
      );
      if (targetSheetIndex >= 0 && targetSheetIndex !== activeSheet) {
        setActiveSheet(targetSheetIndex);
      }
    }
    // Reset the manual sheet change flag after the effect runs
    isManualSheetChange.current = false;
  }, [activeSegment, sheets, activeSheet]);

  useEffect(() => {
    if (activeSegment) {
      const currentSheet = sheets[activeSheet];
      const info = currentSheet?.segmentRanges.get(activeSegment.segmentId);
      if (info) {
        setHoveredHighlight({
          id: activeSegment.segmentId,
          type: info.type,
          range: info.range,
        });
        setSelectedHighlight({
          id: activeSegment.segmentId,
          type: info.type,
          range: info.range,
        });
        if (gridRef.current) {
          gridRef.current.scrollToItem({
            columnIndex: info.range.startCol + 1,
            rowIndex: info.range.startRow + 1,
            align: "center",
          });
        }
      } else {
        setSelectedHighlight(null);
        setHoveredHighlight(null);
      }
    } else {
      setSelectedHighlight(null);
      setHoveredHighlight(null);
    }
  }, [activeSegment, sheets, activeSheet]);

  useEffect(() => {
    if (taskResponse?.output?.chunks && taskResponse.output.pages) {
      segmentToChunkMap.current.clear();
      taskResponse.output.chunks.forEach((chunk: Chunk) => {
        chunk.segments.forEach((seg: Segment) => {
          segmentToChunkMap.current.set(seg.segment_id, chunk.chunk_id);
        });
      });
      const segments: Segment[] = taskResponse.output.chunks.flatMap(
        (chunk: Chunk) => chunk.segments
      );
      const sheetsData = taskResponse.output.pages.map((page: Page) => {
        const pageNum = page.page_number;
        const name: string = page.ss_sheet_name ?? `Sheet ${pageNum}`;
        const cellSegments = segments.filter(
          (seg: Segment) =>
            seg.page_number === pageNum &&
            seg.ss_cells != null &&
            seg.ss_cells.length > 0
        );
        if (cellSegments.length > 0) {
          const allCells: RawCell[] = cellSegments.flatMap(
            (seg: Segment) => seg.ss_cells!
          );
          const decoded: { r: number; c: number; cell: RawCell }[] =
            allCells.map((cell: RawCell) => {
              const firstCellRange = cell.range.includes(":")
                ? cell.range.split(":")[0]
                : cell.range;
              const { r, c } = XLSX.utils.decode_cell(firstCellRange);
              return { r, c, cell };
            });
          // Compute mergedRanges based on RawCell.range
          const mergedRanges = allCells
            .filter((cell: RawCell) => cell.range.includes(":"))
            .map((cell: RawCell) => {
              const rangeDecoded = XLSX.utils.decode_range(cell.range);
              return {
                startRow: rangeDecoded.s.r,
                startCol: rangeDecoded.s.c,
                endRow: rangeDecoded.e.r,
                endCol: rangeDecoded.e.c,
              };
            });
          const maxRowFromCells = decoded.reduce(
            (max: number, d: { r: number; c: number; cell: RawCell }) =>
              Math.max(max, d.r),
            0
          );
          const maxColFromCells = decoded.reduce(
            (max: number, d: { r: number; c: number; cell: RawCell }) =>
              Math.max(max, d.c),
            0
          );

          const rangeSegments: Segment[] = segments.filter(
            (seg: Segment) => seg.page_number === pageNum && seg.ss_range
          );
          let maxRowFromRanges = -1;
          let maxColFromRanges = -1;

          rangeSegments.forEach((seg: Segment) => {
            const range = XLSX.utils.decode_range(seg.ss_range!);
            maxRowFromRanges = Math.max(maxRowFromRanges, range.e.r);
            maxColFromRanges = Math.max(maxColFromRanges, range.e.c);
          });

          const maxRow = Math.max(maxRowFromCells, maxRowFromRanges);
          const maxCol = Math.max(maxColFromCells, maxColFromRanges);

          const rowsMatrix: string[][] = Array.from(
            { length: maxRow + 1 },
            () => Array(maxCol + 1).fill("")
          );
          const cellsMatrix: (RawCell | null)[][] = Array.from(
            { length: maxRow + 1 },
            () => Array(maxCol + 1).fill(null)
          );
          const formulaMatrix: boolean[][] = Array.from(
            { length: maxRow + 1 },
            () => Array(maxCol + 1).fill(false)
          );
          const formulaStringsMatrix: string[][] = Array.from(
            { length: maxRow + 1 },
            () => Array(maxCol + 1).fill("")
          );
          decoded.forEach(
            ({ r, c, cell }: { r: number; c: number; cell: RawCell }) => {
              cellsMatrix[r][c] = cell;
              if (cell.formula != null) {
                rowsMatrix[r][c] = cell.value != null ? cell.value : cell.text;
                formulaMatrix[r][c] = true;
                formulaStringsMatrix[r][c] = cell.formula;
              } else {
                rowsMatrix[r][c] = cell.value != null ? cell.value : cell.text;
                formulaMatrix[r][c] = false;
                formulaStringsMatrix[r][c] = "";
              }
            }
          );

          const highlightMatrix: (CellHighlight | undefined)[][] = Array.from(
            { length: maxRow + 1 },
            () => Array(maxCol + 1).fill(undefined)
          );
          const pageSegmentRanges = new Map<
            string,
            {
              type: string;
              range: {
                startRow: number;
                startCol: number;
                endRow: number;
                endCol: number;
              };
              headerRange?: {
                startRow: number;
                startCol: number;
                endRow: number;
                endCol: number;
              };
            }
          >();

          const imageSegments: ImageSegment[] = [];

          rangeSegments.forEach((seg: Segment) => {
            const type = seg.segment_type;
            const id = seg.segment_id;
            const range = XLSX.utils.decode_range(seg.ss_range!);

            pageSegmentRanges.set(id, {
              type,
              range: {
                startRow: range.s.r,
                startCol: range.s.c,
                endRow: range.e.r,
                endCol: range.e.c,
              },
            });

            if (seg.image) {
              imageSegments.push({
                id,
                imageUrl: seg.image,
                range: {
                  startRow: range.s.r,
                  startCol: range.s.c,
                  endRow: range.e.r,
                  endCol: range.e.c,
                },
              });
            }

            for (let r = range.s.r; r <= range.e.r; r++) {
              for (let c = range.s.c; c <= range.e.c; c++) {
                highlightMatrix[r][c] = { id, type };
              }
            }
          });

          const headerSegments: Segment[] = segments.filter(
            (seg: Segment) => seg.page_number === pageNum && seg.ss_header_range
          );

          headerSegments.forEach((seg: Segment) => {
            const type = seg.segment_type;
            const id = seg.segment_id;
            const headerRangeDecoded = XLSX.utils.decode_range(
              seg.ss_header_range!
            );
            const headerCellRange = {
              startRow: headerRangeDecoded.s.r,
              startCol: headerRangeDecoded.s.c,
              endRow: headerRangeDecoded.e.r,
              endCol: headerRangeDecoded.e.c,
            };
            const existing = pageSegmentRanges.get(id);
            if (existing) {
              existing.headerRange = headerCellRange;
            } else {
              pageSegmentRanges.set(id, {
                type,
                range: headerCellRange,
                headerRange: headerCellRange,
              });
            }

            for (
              let r = headerRangeDecoded.s.r;
              r <= headerRangeDecoded.e.r;
              r++
            ) {
              for (
                let c = headerRangeDecoded.s.c;
                c <= headerRangeDecoded.e.c;
                c++
              ) {
                if (!highlightMatrix[r][c]) {
                  highlightMatrix[r][c] = { id, type };
                }
              }
            }
          });

          // Replace manual matrix adjustment and copying with slicing-based adjustment
          const actualColumnCount = maxColFromRanges + 1;
          const actualRowCount = maxRowFromRanges + 1;

          const adjustedRowsMatrix = rowsMatrix
            .slice(0, actualRowCount)
            .map((row) => row.slice(0, actualColumnCount));

          const adjustedHighlightMatrix = highlightMatrix
            .slice(0, actualRowCount)
            .map((row) => row.slice(0, actualColumnCount));

          const adjustedFormulaMatrix = formulaMatrix
            .slice(0, actualRowCount)
            .map((row) => row.slice(0, actualColumnCount));

          const adjustedFormulaStringsMatrix = formulaStringsMatrix
            .slice(0, actualRowCount)
            .map((row) => row.slice(0, actualColumnCount));

          const adjustedCellsMatrix = cellsMatrix
            .slice(0, actualRowCount)
            .map((row) => row.slice(0, actualColumnCount));

          return {
            name,
            rows: adjustedRowsMatrix,
            cells: adjustedCellsMatrix,
            highlights: adjustedHighlightMatrix,
            formulas: adjustedFormulaMatrix,
            formulaStrings: adjustedFormulaStringsMatrix,
            segmentRanges: pageSegmentRanges,
            imageSegments,
            mergedRanges,
            maxColumnFromRanges: maxColFromRanges,
            maxRowFromRanges,
          };
        }
        return {
          name,
          rows: [],
          cells: [],
          highlights: [],
          formulas: [],
          formulaStrings: [],
          segmentRanges: new Map(),
          imageSegments: [],
          mergedRanges: [],
          maxColumnFromRanges: -1,
          maxRowFromRanges: -1,
        };
      });
      setSheets(sheetsData);
      setActiveSheet(0);
    }
  }, [taskResponse]);

  // Update container dimensions when window is resized
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setContainerDimensions({
          width: width > 0 ? width : 1200,
          height: height > 0 ? height : 600,
        });
      }
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);

    let resizeObserver: ResizeObserver | null = null;

    const setupObserver = () => {
      if (containerRef.current && !resizeObserver) {
        resizeObserver = new ResizeObserver((entries) => {
          for (const entry of entries) {
            const { width, height } = entry.contentRect;
            setContainerDimensions({
              width: width > 0 ? width : 1200,
              height: height > 0 ? height : 600,
            });
          }
        });
        resizeObserver.observe(containerRef.current);
      }
    };

    setupObserver();

    if (!containerRef.current) {
      const retryId = setTimeout(setupObserver, 10);
      return () => {
        clearTimeout(retryId);
        window.removeEventListener("resize", updateDimensions);
        if (resizeObserver) {
          resizeObserver.disconnect();
        }
      };
    }

    return () => {
      window.removeEventListener("resize", updateDimensions);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, []);

  // Update when sheets change
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setContainerDimensions({
          width: width > 0 ? width : 1200,
          height: height > 0 ? height : 600,
        });
      }
    };

    const timeoutId = setTimeout(updateDimensions, 10);

    return () => clearTimeout(timeoutId);
  }, [sheets, activeSheet]);

  // Calculate optimal column widths based on content
  const calculateColumnWidth = useCallback(
    (columnIndex: number): number => {
      if (columnIndex === 0) return ROW_HEADER_WIDTH;

      const dataColumnIndex = columnIndex - 1;
      let maxWidth = MIN_CELL_WIDTH;

      const sheet = sheets[activeSheet];
      // First 50 rows to calculate width
      const sampleRows = sheet?.rows?.slice(0, 50) || [];

      for (const row of sampleRows) {
        const cellValue = row[dataColumnIndex];
        if (cellValue != null) {
          const estimatedWidth = String(cellValue).length * 8 + 20;
          maxWidth = Math.max(maxWidth, estimatedWidth);
        }
      }

      return Math.min(maxWidth, MAX_CELL_WIDTH);
    },
    [sheets, activeSheet]
  );

  const itemData = useCallback((): CellData => {
    const sheet = sheets[activeSheet];
    if (!sheet) {
      return {
        rows: [],
        cells: [],
        highlights: [],
        formulas: [],
        formulaStrings: [],
        segmentRanges: new Map(),
        hoveredHighlight,
        selectedHighlight,
        setHoveredHighlight,
        setHoveredFormula,
        onRangeClick: undefined,
        columnWidths: [],
        imageSegments: [],
        mergedRanges: [],
      };
    }
    const colCount = Math.max(1, sheet.maxColumnFromRanges + 1) + 2;
    return {
      rows: sheet.rows,
      cells: sheet.cells,
      highlights: sheet.highlights,
      formulas: sheet.formulas,
      formulaStrings: sheet.formulaStrings,
      segmentRanges: sheet.segmentRanges,
      hoveredHighlight,
      selectedHighlight,
      setHoveredHighlight,
      setHoveredFormula,
      onRangeClick: (segmentId: string) => {
        const chunkId = segmentToChunkMap.current.get(segmentId);
        if (chunkId && onRangeClick) {
          onRangeClick(chunkId, segmentId);
        }
      },
      columnWidths: Array.from({ length: colCount }, (_, idx) =>
        calculateColumnWidth(idx)
      ),
      imageSegments: sheet.imageSegments,
      mergedRanges: sheet.mergedRanges,
    };
  }, [
    sheets,
    activeSheet,
    hoveredHighlight,
    selectedHighlight,
    onRangeClick,
    calculateColumnWidth,
  ]);

  if (!taskResponse?.output?.chunks) {
    return (
      <p className="p-4 text-center text-gray-500">No Excel data to display</p>
    );
  }

  // Prepare sheet and compute dimensions
  const sheet = sheets[activeSheet];
  if (!sheet || sheets.length === 0) {
    return <p className="p-4 text-center text-gray-500">Loading...</p>;
  }

  const rowCount = Math.max(1, sheet.maxRowFromRanges + 1) + 1 + 1; // +1 for header row, +1 extra row
  const columnCount = Math.max(1, sheet.maxColumnFromRanges + 1) + 1 + 1; // +1 for row header column, +1 extra column

  const CELL_HEIGHT = 24;
  const MIN_CELL_WIDTH = 100;
  const MAX_CELL_WIDTH = 300;
  const ROW_HEADER_WIDTH = 60;
  const COLUMN_HEADER_HEIGHT = 32;

  return (
    <div className="flex flex-col h-full bg-white text-black">
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden relative"
        onMouseLeave={() => {
          setHoveredHighlight(null);
          setHoveredFormula(null);
        }}
      >
        {hoveredFormula && (
          <div
            className="absolute left-0 ml-2 px-2 py-1 text-xs text-white bg-black rounded pointer-events-none"
            style={{ bottom: "16px", zIndex: 1 }}
          >
            Formula: {hoveredFormula}
          </div>
        )}

        {sheet && (
          <>
            <Grid
              columnCount={columnCount}
              columnWidth={calculateColumnWidth}
              height={containerDimensions.height}
              rowCount={rowCount}
              rowHeight={(index: number) =>
                index === 0 ? COLUMN_HEADER_HEIGHT : CELL_HEIGHT
              }
              width={containerDimensions.width}
              className="font-medium excel-grid-scrollbar"
              itemData={itemData()}
              ref={gridRef}
            >
              {Cell}
            </Grid>
          </>
        )}
      </div>
      <div className="flex border-t bg-gray-100 overflow-x-auto">
        {sheets.map((sheet, idx) => (
          <button
            key={idx}
            className={`px-3 text-xs py-2 focus:outline-none rounded-none font-medium flex-shrink-0 w-32 h-8 truncate ${
              idx === activeSheet ? "bg-white" : "bg-gray-100"
            }`}
            onClick={() => {
              isManualSheetChange.current = true;
              setActiveSheet(idx);
            }}
            title={sheet.name}
          >
            {sheet.name}
          </button>
        ))}
      </div>
    </div>
  );
}

export default ExcelViewer;
