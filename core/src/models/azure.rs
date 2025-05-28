use crate::models::output::{BoundingBox, Chunk, OCRResult, Segment, SegmentType};
use crate::utils::services::html::convert_table_to_markdown;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;

use super::upload::SegmentationStrategy;

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
    pub bounding_regions: Option<Vec<BoundingRegion>>,
    pub column_index: Option<i64>,
    pub column_span: Option<i64>,
    pub content: Option<String>,
    pub elements: Option<Vec<String>>,
    #[serde(default)]
    pub kind: Option<String>,
    pub row_index: Option<i64>,
    pub row_span: Option<i64>,
    pub spans: Option<Vec<Span>>,
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

#[derive(Debug, Clone)]
pub enum DocumentAnalysisFeature {
    Barcodes,
    Formulas,
    KeyValuePairs,
    Languages,
    OcrHighResolution,
    QueryFields,
    StyleFont,
}

impl DocumentAnalysisFeature {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Barcodes => "barcodes",
            Self::Formulas => "formulas",
            Self::KeyValuePairs => "keyValuePairs",
            Self::Languages => "languages",
            Self::OcrHighResolution => "ocrHighResolution",
            Self::QueryFields => "queryFields",
            Self::StyleFont => "styleFont",
        }
    }
}

impl AzureAnalysisResponse {
    pub fn to_chunks(
        &self,
        segmentation_strategy: SegmentationStrategy,
    ) -> Result<Vec<Chunk>, Box<dyn Error>> {
        match segmentation_strategy {
            SegmentationStrategy::Page => {
                let mut page_segments = Vec::new();

                if let Some(analyze_result) = &self.analyze_result {
                    if let Some(pages) = &analyze_result.pages {
                        for page in pages {
                            let page_number = page.page_number.unwrap_or(1) as u32;
                            let unit = page.unit.as_deref();
                            let (width, height) = (
                                convert_unit_to_pixels(page.width.unwrap_or(0.0), unit),
                                convert_unit_to_pixels(page.height.unwrap_or(0.0), unit),
                            );

                            let mut ocr_results = Vec::new();
                            if let Some(words) = &page.words {
                                for word in words {
                                    if let (Some(polygon), Some(content), Some(confidence)) =
                                        (&word.polygon, &word.content, &word.confidence)
                                    {
                                        if let Ok(word_bbox) = create_word_bbox(polygon, unit) {
                                            let ocr_result = OCRResult {
                                                text: content.clone(),
                                                confidence: Some(*confidence as f32),
                                                bbox: word_bbox,
                                            };
                                            ocr_results.push(ocr_result);
                                        }
                                    }
                                }
                            }
                            let segment = Segment::new(
                                BoundingBox::new(0.0, 0.0, width, height),
                                Some(1.0),
                                ocr_results,
                                height,
                                width,
                                page_number,
                                SegmentType::Page,
                            );

                            page_segments.push(segment);
                        }
                    }
                }

                Ok(page_segments
                    .into_iter()
                    .map(|segment| Chunk::new(vec![segment]))
                    .collect())
            }
            SegmentationStrategy::LayoutAnalysis => {
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
                                    let unit = page.unit.as_deref();
                                    let (width, height, unit) = (
                                        convert_unit_to_pixels(
                                            page.width.unwrap_or(0.0),
                                            unit,
                                        ),
                                        convert_unit_to_pixels(
                                            page.height.unwrap_or(0.0),
                                            unit,
                                        ),
                                        unit,
                                    );
                                    (page_number, (width, height, unit))
                                })
                                .collect::<std::collections::HashMap<u32, (f32, f32, Option<&str>)>>()
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
                                                if let Some(idx) = extract_paragraph_index(element)
                                                {
                                                    min_paragraph_idx = min_paragraph_idx.min(idx);
                                                    skip_paragraphs.insert(idx);
                                                }
                                            }
                                        }
                                    }
                                }

                                if let Some(regions) = &table.bounding_regions {
                                    if let Some(first_region) = regions.first() {
                                        let page_number =
                                            first_region.page_number.unwrap_or(1) as u32;
                                        let (page_width, page_height, unit) = page_dimensions
                                            .get(&page_number)
                                            .copied()
                                            .unwrap_or((0.0, 0.0, None));

                                        let bbox = create_bounding_box(first_region, unit);
                                        let segment = Segment {
                                            bbox,
                                            confidence: None,
                                            content: table_to_text(table),
                                            html: table_to_html(table),
                                            markdown: table_to_markdown(table),
                                            image: None,
                                            llm: None,
                                            ocr: None,
                                            page_height,
                                            page_width,
                                            page_number,
                                            segment_id: uuid::Uuid::new_v4().to_string(),
                                            segment_type: SegmentType::Table,
                                        };

                                        if min_paragraph_idx != usize::MAX {
                                            replacements
                                                .entry(min_paragraph_idx)
                                                .or_default()
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
                                    let (page_width, page_height, unit) = page_dimensions
                                        .get(&page_number)
                                        .copied()
                                        .unwrap_or((0.0, 0.0, None));

                                    let bbox = create_bounding_box(first_region, unit);
                                    let segment = Segment {
                                        bbox,
                                        confidence: None,
                                        content: String::new(),
                                        html: String::new(),
                                        markdown: String::new(),
                                        image: None,
                                        llm: None,
                                        ocr: None,
                                        page_height,
                                        page_width,
                                        page_number,
                                        segment_id: uuid::Uuid::new_v4().to_string(),
                                        segment_type: SegmentType::Picture,
                                    };

                                    if min_paragraph_idx != usize::MAX {
                                        replacements
                                            .entry(min_paragraph_idx)
                                            .or_default()
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
                                    let (page_width, page_height, unit) = page_dimensions
                                        .get(&page_number)
                                        .copied()
                                        .unwrap_or((0.0, 0.0, None));

                                    let bbox = create_bounding_box(first_region, unit);
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
                                        content: paragraph.content.clone().unwrap_or_default(),
                                        html: String::new(),
                                        markdown: String::new(),
                                        image: None,
                                        llm: None,
                                        ocr: None,
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

                        // Initialize OCR results for all segments
                        for segment in &mut all_segments {
                            segment.ocr = Some(Vec::new());
                        }

                        // Assign OCR words to segments based on intersection area
                        if let Some(pages) = &analyze_result.pages {
                            for page in pages {
                                let page_number = page.page_number.unwrap_or(1) as u32;
                                let unit = page.unit.as_deref();
                                if let Some(words) = &page.words {
                                    for word in words {
                                        if let (Some(polygon), Some(content), Some(confidence)) =
                                            (&word.polygon, &word.content, &word.confidence)
                                        {
                                            let word_bbox = create_word_bbox(polygon, unit)?;
                                            let mut max_area = 0.0;
                                            let mut best_segment_idx = None;

                                            for (idx, segment) in all_segments.iter().enumerate() {
                                                if segment.page_number == page_number {
                                                    let area =
                                                        segment.bbox.intersection_area(&word_bbox);
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

                                                let ocr_result = OCRResult {
                                                    text: content.clone(),
                                                    confidence: Some(*confidence as f32),
                                                    bbox: relative_bbox,
                                                };

                                                if let Some(ocr_vec) = &mut all_segments[idx].ocr {
                                                    ocr_vec.push(ocr_result);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Update segment content fields to reflect their OCR results
                        for segment in &mut all_segments {
                            if let Some(ocr_results) = &segment.ocr {
                                if !ocr_results.is_empty() {
                                    segment.content = ocr_results
                                        .iter()
                                        .map(|ocr_result| ocr_result.text.clone())
                                        .collect::<Vec<String>>()
                                        .join(" ");
                                }
                            }
                        }

                        // Ensure each page has at least one segment
                        if let Some(pages) = &analyze_result.pages {
                            let mut pages_with_segments = std::collections::HashMap::new();

                            // Initialize all pages as having no segments
                            for page in pages {
                                let page_number = page.page_number.unwrap_or(1) as u32;
                                pages_with_segments.insert(page_number, false);
                            }

                            // Mark pages that have segments
                            for segment in &all_segments {
                                pages_with_segments.insert(segment.page_number, true);
                            }

                            // Add full-page segments for pages without segments
                            for page in pages {
                                let page_number = page.page_number.unwrap_or(1) as u32;
                                if !pages_with_segments.get(&page_number).unwrap_or(&true) {
                                    let (width, height, unit) = page_dimensions
                                        .get(&page_number)
                                        .copied()
                                        .unwrap_or((0.0, 0.0, None));

                                    println!(
                                        "No segments detected for page {}. Adding full-page segment with dimensions {:?}",
                                        page_number,
                                        (width, height, unit)
                                    );

                                    let mut ocr_results = Vec::new();
                                    if let Some(words) = &page.words {
                                        for word in words {
                                            if let (
                                                Some(polygon),
                                                Some(content),
                                                Some(confidence),
                                            ) = (&word.polygon, &word.content, &word.confidence)
                                            {
                                                if let Ok(word_bbox) =
                                                    create_word_bbox(polygon, unit)
                                                {
                                                    let ocr_result = OCRResult {
                                                        text: content.clone(),
                                                        confidence: Some(*confidence as f32),
                                                        bbox: word_bbox,
                                                    };
                                                    ocr_results.push(ocr_result);
                                                }
                                            }
                                        }
                                    }

                                    let segment = Segment::new(
                                        BoundingBox::new(0.0, 0.0, width, height),
                                        Some(1.0),
                                        ocr_results,
                                        height,
                                        width,
                                        page_number,
                                        SegmentType::Page,
                                    );

                                    all_segments.push(segment);
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
    }
}

fn process_caption(
    caption: &Caption,
    replacements: &mut std::collections::BTreeMap<usize, Vec<Segment>>,
    skip_paragraphs: &mut std::collections::HashSet<usize>,
    page_dimensions: &std::collections::HashMap<u32, (f32, f32, Option<&str>)>,
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
                    let (page_width, page_height, unit) = page_dimensions
                        .get(&page_number)
                        .copied()
                        .unwrap_or((0.0, 0.0, None));

                    let bbox = create_bounding_box(first_region, unit);
                    let segment = Segment {
                        bbox,
                        confidence: None,
                        content: caption.content.clone().unwrap_or_default(),
                        html: String::new(),
                        markdown: String::new(),
                        image: None,
                        llm: None,
                        ocr: None,
                        page_height,
                        page_width,
                        page_number,
                        segment_id: uuid::Uuid::new_v4().to_string(),
                        segment_type: SegmentType::Caption,
                    };
                    replacements.entry(first_idx).or_default().push(segment);
                }
            }
        }
    }
}

fn extract_paragraph_index(element: &str) -> Option<usize> {
    element.strip_prefix("/paragraphs/")?.parse::<usize>().ok()
}

fn convert_unit_to_pixels(value: f64, unit: Option<&str>) -> f32 {
    match unit {
        Some("inch") => (value * 72.0) as f32,
        Some("pixel") => value as f32,
        _ => {
            // If unit is unknown, log it and default to treating as pixels
            if let Some(unit_str) = unit {
                println!("Unknown unit: {}", unit_str);
            }
            value as f32
        }
    }
}

fn create_bounding_box(region: &BoundingRegion, unit: Option<&str>) -> BoundingBox {
    if let Some(polygon) = &region.polygon {
        if polygon.len() >= 8 {
            let points: Vec<f32> = polygon
                .iter()
                .map(|&coord| convert_unit_to_pixels(coord, unit))
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

fn get_cell_content(cell: &Cell) -> String {
    cell.content.as_deref().unwrap_or("").to_string()
}

fn table_to_text(table: &Table) -> String {
    table
        .cells
        .as_ref()
        .map(|cells| {
            cells
                .iter()
                .filter_map(|cell| {
                    let content: String = get_cell_content(cell);
                    if content.is_empty() {
                        None
                    } else {
                        Some(content)
                    }
                })
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

    let mut covered = vec![vec![false; col_count]; row_count];

    for row_idx in 0..row_count {
        html.push_str("<tr>");
        for col_idx in 0..col_count {
            if covered[row_idx][col_idx] {
                continue;
            }

            if let Some(cell) = cells.iter().find(|c| {
                c.row_index.is_some_and(|r| r as usize == row_idx)
                    && c.column_index.is_some_and(|c| c as usize == col_idx)
            }) {
                let content = get_cell_content(cell);
                let rowspan = cell.row_span.unwrap_or(1);
                let colspan = cell.column_span.unwrap_or(1);

                for r in 0..rowspan as usize {
                    for c in 0..colspan as usize {
                        if row_idx + r < row_count && col_idx + c < col_count {
                            covered[row_idx + r][col_idx + c] = true;
                        }
                    }
                }

                if rowspan > 1 || colspan > 1 {
                    html.push_str(&format!(
                        "<td rowspan=\"{}\" colspan=\"{}\">",
                        rowspan, colspan
                    ));
                } else {
                    html.push_str("<td>");
                }
                html.push_str(&content);
                html.push_str("</td>");
            } else {
                html.push_str("<td></td>");
            }
        }
        html.push_str("</tr>");
    }

    html.push_str("</table>");
    html
}

fn table_to_markdown(table: &Table) -> Result<String, Box<dyn Error>> {
    convert_html_to_markdown(table_to_html(table)?)
}

fn create_word_bbox(polygon: &[f64], unit: Option<&str>) -> Result<BoundingBox, Box<dyn Error>> {
    if polygon.len() >= 8 {
        let points: Vec<f32> = polygon
            .iter()
            .map(|&coord| convert_unit_to_pixels(coord, unit))
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
