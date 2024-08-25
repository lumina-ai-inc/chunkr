// Define the structure for a single segment
export interface Segment {
  left: number;
  top: number;
  width: number;
  height: number;
  page_number: number;
  page_width: number;
  page_height: number;
  text: string;
  type: string;
}

// Define the structure for a chunk, which includes segments and markdown
export interface Chunk {
  segments: Segment[];
  markdown: string;
}

// Define the overall structure of the bounding boxes JSON
export type BoundingBoxes = Chunk[];
