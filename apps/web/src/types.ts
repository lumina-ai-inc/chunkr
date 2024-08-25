export interface Segment {
  left: string,
  top: string,
  width: string,
  height: string,
  page_number: number,
  page_width: string,
  page_height: string,
  text: string,
  type: "Caption" | "Footnote" | "Formula" | "ListItem" | "PageFooter" | "PageHeader" | "Picture" | "SectionHeader" | "Table" | "Text" | "Title"
};
