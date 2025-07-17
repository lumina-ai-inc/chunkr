import type { TaskResponse } from "./taskResponse.model";
import type { Cell } from "./taskResponse.model";

export interface CellRange {
  startRow: number;
  startCol: number;
  endRow: number;
  endCol: number;
}

export interface CellHighlight {
  id: string;
  type: string;
}

export interface ImageSegment {
  id: string;
  imageUrl: string;
  range: CellRange;
}

export interface ExcelViewerProps {
  taskResponse: TaskResponse;
  activeSegment?: { chunkId: string; segmentId: string } | null;
  onRangeClick?: (chunkId: string, segmentId: string) => void;
}

export interface CellData {
  rows: string[][];
  cells: (Cell | null)[][];
  highlights: (CellHighlight | undefined)[][];
  formulas: boolean[][];
  formulaStrings: (string | null)[][];
  segmentRanges: Map<string, SegmentRangeInfo>;
  hoveredHighlight: HighlightState | null;
  selectedHighlight: HighlightState | null;
  setHoveredHighlight: (highlight: HighlightState | null) => void;
  setHoveredFormula: (formula: string | null) => void;
  onRangeClick?: (segmentId: string) => void;
  columnWidths: number[];
  imageSegments: ImageSegment[];
  mergedRanges: CellRange[];
}

export interface HighlightState {
  id: string;
  type: string;
  range: CellRange;
}

export interface SegmentRangeInfo {
  type: string;
  range: CellRange;
  headerRange?: CellRange;
}

export interface SheetData {
  name: string;
  rows: string[][];
  cells: (Cell | null)[][];
  highlights: (CellHighlight | undefined)[][];
  formulas: boolean[][];
  formulaStrings: (string | null)[][];
  segmentRanges: Map<string, SegmentRangeInfo>;
  imageSegments: ImageSegment[];
  mergedRanges: CellRange[];
  maxColumnFromRanges: number;
  maxRowFromRanges: number;
}
