use crate::models::chunkr::output::{BoundingBox, Chunk, OCRResult, Segment, SegmentType};
use crate::utils::services::html::convert_table_to_markdown;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AzureAnalysisResponse {
    pub status: String,
    pub created_date_time: Option<String>,
    pub last_updated_date_time: Option<String>,
    pub analyze_result: Option<AnalyzeResult>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeResult {
    pub api_version: Option<String>,
    pub model_id: Option<String>,
    pub string_index_type: Option<String>,
    pub content: Option<String>,
    pub pages: Option<Vec<Page>>,
    pub tables: Option<Vec<Table>>,
    pub paragraphs: Option<Vec<Paragraph>>,
    pub styles: Option<Vec<Value>>,
    pub content_format: Option<String>,
    pub sections: Option<Vec<Section>>,
    pub figures: Option<Vec<Figure>>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Span {
    pub offset: Option<i64>,
    pub length: Option<i64>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Page {
    pub page_number: Option<i64>,
    pub angle: Option<f64>,
    pub width: Option<f64>,
    pub height: Option<f64>,
    pub unit: Option<String>,
    pub words: Option<Vec<Word>>,
    pub selection_marks: Option<Vec<SelectionMark>>,
    pub lines: Option<Vec<Line>>,
    pub spans: Option<Vec<Span>>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Word {
    pub content: Option<String>,
    pub polygon: Option<Vec<f64>>,
    pub confidence: Option<f64>,
    pub span: Option<Span>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectionMark {
    pub state: Option<String>,
    pub polygon: Option<Vec<f64>>,
    pub confidence: Option<f64>,
    pub span: Option<Span>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Line {
    pub content: Option<String>,
    pub polygon: Option<Vec<f64>>,
    pub spans: Option<Vec<Span>>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Table {
    pub row_count: Option<i64>,
    pub column_count: Option<i64>,
    pub cells: Option<Vec<Cell>>,
    pub bounding_regions: Option<Vec<BoundingRegion>>,
    pub spans: Option<Vec<Span>>,
    pub caption: Option<Caption>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cell {
    pub kind: Option<String>,
    pub row_index: Option<i64>,
    pub column_index: Option<i64>,
    pub content: Option<String>,
    pub bounding_regions: Option<Vec<BoundingRegion>>,
    pub spans: Option<Vec<Span>>,
    #[serde(default)]
    pub elements: Option<Vec<String>>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundingRegion {
    pub page_number: Option<i64>,
    pub polygon: Option<Vec<f64>>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Caption {
    pub content: Option<String>,
    pub bounding_regions: Option<Vec<BoundingRegion>>,
    pub spans: Option<Vec<Span>>,
    pub elements: Option<Vec<String>>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Paragraph {
    pub spans: Option<Vec<Span>>,
    pub bounding_regions: Option<Vec<BoundingRegion>>,
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Section {
    pub spans: Option<Vec<Span>>,
    pub elements: Option<Vec<String>>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Figure {
    pub id: String,
    pub bounding_regions: Vec<BoundingRegion>,
    pub spans: Option<Vec<Span>>,
    pub elements: Option<Vec<String>>,
    pub caption: Option<Caption>,
}

impl AzureAnalysisResponse {
    pub fn to_chunks(&self) -> Result<Vec<Chunk>, Box<dyn Error>> {
        let mut all_segments: Vec<Segment> = Vec::new();

        if let Some(analyze_result) = &self.analyze_result {
            if let Some(paragraphs) = &analyze_result.paragraphs {
                let mut replacements: std::collections::BTreeMap<usize, Vec<Segment>> =
                    std::collections::BTreeMap::new();
                let mut skip_paragraphs = std::collections::HashSet::new();

                let page_dimensions = if let Some(pages) = &analyze_result.pages {
                    pages
                        .iter()
                        .map(|page| {
                            let page_number = page.page_number.unwrap_or(1) as u32;
                            let (width, height) = match page.unit.as_deref() {
                                Some("inch") => (
                                    inches_to_pixels(page.width.unwrap_or(0.0) as f64),
                                    inches_to_pixels(page.height.unwrap_or(0.0) as f64),
                                ),
                                _ => (0.0, 0.0),
                            };
                            (page_number, (width, height))
                        })
                        .collect::<std::collections::HashMap<u32, (f32, f32)>>()
                } else {
                    std::collections::HashMap::new()
                };

                if let Some(tables) = &analyze_result.tables {
                    for table in tables {
                        let mut min_paragraph_idx = usize::MAX;

                        if let Some(cells) = &table.cells {
                            for cell in cells {
                                if let Some(elements) = &cell.elements {
                                    for element in elements {
                                        if let Some(idx) = extract_paragraph_index(element) {
                                            min_paragraph_idx = min_paragraph_idx.min(idx);
                                            skip_paragraphs.insert(idx);
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(regions) = &table.bounding_regions {
                            if let Some(first_region) = regions.first() {
                                let page_number = first_region.page_number.unwrap_or(1) as u32;
                                let (page_width, page_height) = page_dimensions
                                    .get(&page_number)
                                    .copied()
                                    .unwrap_or((0.0, 0.0));

                                let bbox = create_bounding_box(first_region);
                                let segment = Segment {
                                    bbox,
                                    confidence: None,
                                    content: table_to_text(table),
                                    html: Some(table_to_html(table)),
                                    markdown: Some(table_to_markdown(table)),
                                    image: None,
                                    llm: None,
                                    ocr: Vec::new(),
                                    page_height,
                                    page_width,
                                    page_number,
                                    segment_id: uuid::Uuid::new_v4().to_string(),
                                    segment_type: SegmentType::Table,
                                };

                                if min_paragraph_idx != usize::MAX {
                                    replacements
                                        .entry(min_paragraph_idx)
                                        .or_insert_with(Vec::new)
                                        .push(segment);
                                }
                            }
                        }

                        if let Some(caption) = &table.caption {
                            process_caption(
                                caption,
                                &mut replacements,
                                &mut skip_paragraphs,
                                &page_dimensions,
                            );
                        }
                    }
                }

                if let Some(figures) = &analyze_result.figures {
                    for figure in figures {
                        let mut min_paragraph_idx = usize::MAX;

                        if let Some(elements) = &figure.elements {
                            for element in elements {
                                if let Some(idx) = extract_paragraph_index(element) {
                                    min_paragraph_idx = min_paragraph_idx.min(idx);
                                    skip_paragraphs.insert(idx);
                                }
                            }
                        }

                        if !figure.bounding_regions.is_empty() {
                            let first_region = &figure.bounding_regions[0];
                            let page_number = first_region.page_number.unwrap_or(1) as u32;
                            let (page_width, page_height) = page_dimensions
                                .get(&page_number)
                                .copied()
                                .unwrap_or((0.0, 0.0));

                            let bbox = create_bounding_box(first_region);
                            let segment = Segment {
                                bbox,
                                confidence: None,
                                content: String::new(),
                                html: None,
                                markdown: None,
                                image: None,
                                llm: None,
                                ocr: Vec::new(),
                                page_height,
                                page_width,
                                page_number,
                                segment_id: uuid::Uuid::new_v4().to_string(),
                                segment_type: SegmentType::Picture,
                            };

                            if min_paragraph_idx != usize::MAX {
                                replacements
                                    .entry(min_paragraph_idx)
                                    .or_insert_with(Vec::new)
                                    .push(segment);
                            }
                        }

                        if let Some(caption) = &figure.caption {
                            process_caption(
                                caption,
                                &mut replacements,
                                &mut skip_paragraphs,
                                &page_dimensions,
                            );
                        }
                    }
                }

                for (idx, paragraph) in paragraphs.iter().enumerate() {
                    if skip_paragraphs.contains(&idx) {
                        if let Some(replacement_segments) = replacements.get(&idx) {
                            all_segments.extend(replacement_segments.clone());
                        }
                        continue;
                    }

                    if let Some(regions) = &paragraph.bounding_regions {
                        if let Some(first_region) = regions.first() {
                            let page_number = first_region.page_number.unwrap_or(1) as u32;
                            let (page_width, page_height) = page_dimensions
                                .get(&page_number)
                                .copied()
                                .unwrap_or((0.0, 0.0));

                            let bbox = create_bounding_box(first_region);
                            let segment_type = match paragraph.role.as_deref() {
                                Some("title") => SegmentType::Title,
                                Some("sectionHeading") => SegmentType::SectionHeader,
                                Some("pageHeader") => SegmentType::PageHeader,
                                Some("pageNumber") => SegmentType::PageFooter,
                                Some("pageFooter") => SegmentType::PageFooter,
                                _ => SegmentType::Text,
                            };

                            let segment = Segment {
                                bbox,
                                confidence: None,
                                content: paragraph
                                    .content
                                    .clone()
                                    .unwrap_or_default()
                                    .replace(":selected:", ""),
                                html: None,
                                markdown: None,
                                image: None,
                                llm: None,
                                ocr: Vec::new(),
                                page_height,
                                page_width,
                                page_number,
                                segment_id: uuid::Uuid::new_v4().to_string(),
                                segment_type,
                            };
                            all_segments.push(segment);
                        }
                    }
                }

                // Assign OCR words to segments based on intersection area
                if let Some(pages) = &analyze_result.pages {
                    for page in pages {
                        let page_number = page.page_number.unwrap_or(1) as u32;

                        if let Some(words) = &page.words {
                            for word in words {
                                if let (Some(polygon), Some(content), Some(confidence)) =
                                    (&word.polygon, &word.content, &word.confidence)
                                {
                                    let word_bbox = create_word_bbox(polygon)?;
                                    let mut max_area = 0.0;
                                    let mut best_segment_idx = None;

                                    for (idx, segment) in all_segments.iter().enumerate() {
                                        if segment.page_number == page_number {
                                            let area = segment.bbox.intersection_area(&word_bbox);
                                            if area > max_area {
                                                max_area = area;
                                                best_segment_idx = Some(idx);
                                            }
                                        }
                                    }

                                    if let Some(idx) = best_segment_idx {
                                        let segment = &all_segments[idx];
                                        let relative_bbox = BoundingBox::new(
                                            word_bbox.left - segment.bbox.left,
                                            word_bbox.top - segment.bbox.top,
                                            word_bbox.width,
                                            word_bbox.height,
                                        );

                                        all_segments[idx].ocr.push(OCRResult {
                                            text: content.clone().replace(":selected:", ""),
                                            confidence: Some(*confidence as f32),
                                            bbox: relative_bbox,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(all_segments
            .into_iter()
            .map(|segment| Chunk::new(vec![segment]))
            .collect())
    }
}

fn process_caption(
    caption: &Caption,
    replacements: &mut std::collections::BTreeMap<usize, Vec<Segment>>,
    skip_paragraphs: &mut std::collections::HashSet<usize>,
    page_dimensions: &std::collections::HashMap<u32, (f32, f32)>,
) {
    if let Some(elements) = &caption.elements {
        if let Some(first_idx) = elements.first().and_then(|e| extract_paragraph_index(e)) {
            for element in elements {
                if let Some(idx) = extract_paragraph_index(element) {
                    skip_paragraphs.insert(idx);
                }
            }

            if let Some(regions) = &caption.bounding_regions {
                if let Some(first_region) = regions.first() {
                    let page_number = first_region.page_number.unwrap_or(1) as u32;
                    let (page_width, page_height) = page_dimensions
                        .get(&page_number)
                        .copied()
                        .unwrap_or((0.0, 0.0));

                    let bbox = create_bounding_box(first_region);
                    let segment = Segment {
                        bbox,
                        confidence: None,
                        content: caption
                            .content
                            .clone()
                            .unwrap_or_default()
                            .replace(":selected:", ""),
                        html: None,
                        markdown: None,
                        image: None,
                        llm: None,
                        ocr: Vec::new(),
                        page_height,
                        page_width,
                        page_number,
                        segment_id: uuid::Uuid::new_v4().to_string(),
                        segment_type: SegmentType::Caption,
                    };
                    replacements
                        .entry(first_idx)
                        .or_insert_with(Vec::new)
                        .push(segment);
                }
            }
        }
    }
}

fn extract_paragraph_index(element: &str) -> Option<usize> {
    element.strip_prefix("/paragraphs/")?.parse::<usize>().ok()
}

fn inches_to_pixels(inches: f64) -> f32 {
    (inches * 72.0) as f32
}

fn create_bounding_box(region: &BoundingRegion) -> BoundingBox {
    if let Some(polygon) = &region.polygon {
        if polygon.len() >= 8 {
            let points: Vec<f32> = polygon
                .iter()
                .map(|&coord| inches_to_pixels(coord))
                .collect();

            let left = points
                .iter()
                .step_by(2)
                .fold(f32::INFINITY, |acc, &x| acc.min(x));
            let top = points
                .iter()
                .skip(1)
                .step_by(2)
                .fold(f32::INFINITY, |acc, &y| acc.min(y));
            let right = points
                .iter()
                .step_by(2)
                .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            let bottom = points
                .iter()
                .skip(1)
                .step_by(2)
                .fold(f32::NEG_INFINITY, |acc, &y| acc.max(y));

            return BoundingBox::new(left, top, right - left, bottom - top);
        }
    }
    BoundingBox::new(0.0, 0.0, 0.0, 0.0)
}

fn table_to_text(table: &Table) -> String {
    table
        .cells
        .as_ref()
        .map(|cells| {
            cells
                .iter()
                .filter_map(|cell| cell.content.as_ref().map(|s| s.replace(":selected:", "")))
                .collect::<Vec<String>>()
                .join(" ")
        })
        .unwrap_or_default()
}

fn table_to_html(table: &Table) -> String {
    let cells = match &table.cells {
        Some(cells) => cells,
        None => return String::new(),
    };

    let row_count = table.row_count.unwrap_or(0) as usize;
    let col_count = table.column_count.unwrap_or(0) as usize;
    if row_count == 0 || col_count == 0 {
        return String::new();
    }

    let mut html = String::from("<table>");

    let mut grid = vec![vec![None; col_count]; row_count];
    for cell in cells {
        if let (Some(row), Some(col), Some(content)) =
            (cell.row_index, cell.column_index, cell.content.as_ref())
        {
            if (row as usize) < row_count && (col as usize) < col_count {
                grid[row as usize][col as usize] = Some(content.replace(":selected:", ""));
            }
        }
    }

    for row in grid {
        html.push_str("<tr>");
        for cell in row {
            html.push_str("<td>");
            html.push_str(cell.as_deref().unwrap_or(""));
            html.push_str("</td>");
        }
        html.push_str("</tr>");
    }

    html.push_str("</table>");
    html
}

fn table_to_markdown(table: &Table) -> String {
    convert_table_to_markdown(table_to_html(table))
}

fn create_word_bbox(polygon: &[f64]) -> Result<BoundingBox, Box<dyn Error>> {
    if polygon.len() >= 8 {
        let points: Vec<f32> = polygon
            .iter()
            .map(|&coord| inches_to_pixels(coord))
            .collect();

        let left = points
            .iter()
            .step_by(2)
            .fold(f32::INFINITY, |acc, &x| acc.min(x));
        let top = points
            .iter()
            .skip(1)
            .step_by(2)
            .fold(f32::INFINITY, |acc, &y| acc.min(y));
        let right = points
            .iter()
            .step_by(2)
            .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let bottom = points
            .iter()
            .skip(1)
            .step_by(2)
            .fold(f32::NEG_INFINITY, |acc, &y| acc.max(y));

        Ok(BoundingBox::new(left, top, right - left, bottom - top))
    } else {
        Err("Invalid polygon length".into())
    }
}
