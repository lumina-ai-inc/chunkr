use crate::configs::{feature_config, otel_config, worker_config};
use crate::models::excel::{IdentifiedElement, IdentifiedElements};
use crate::models::file_operations::ImageConversionResult;
use crate::models::output::{Cell, Chunk, Segment, SegmentCreationError, SegmentType};
use crate::models::task::{Status, Task, TaskPayload};
use crate::utils::services::excel::count_sheets;
use crate::utils::services::file_operations::{check_is_spreadsheet, convert_to_pdf};
use crate::utils::services::html::{
    add_cell_references_to_html, add_excel_headers_to_html, extract_cells_from_ranges,
    extract_rows_from_indicies, get_cell_ref_for_image_src, get_img_sources, indices_to_range,
    parse_range,
};
use crate::utils::services::pdf::{combine_pdfs, count_pages, create_pdf_from_image};
use crate::utils::services::renderer::{render_html_to_image, Capture};
use crate::utils::storage::services::download_to_tempfile;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use opentelemetry::trace::{Span, TraceContextExt, Tracer};
use opentelemetry::Context;
use rayon::prelude::*;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use strum_macros::{Display, EnumString};
use tempfile::{Builder, NamedTempFile};

#[cfg(feature = "memory_profiling")]
use memtrack::track_mem;

fn generate_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[derive(Debug, Clone)]
pub struct Indices {
    pub start_row: usize,
    pub start_col: usize,
    pub end_row: usize,
    pub end_col: usize,
}

impl Indices {
    /// Returns a new Indices with the start and end rows and columns offset by the given values.
    ///
    /// This method subtracts the specified offsets from all row and column indices. If an offset
    /// would result in a negative value, the index is clamped to 0 to prevent underflow.
    ///
    /// This is commonly used when converting between different coordinate systems, such as
    /// adjusting Excel cell references when extracting a subset of data.
    ///
    /// ### Arguments
    /// * `row_offset` - The number of rows to subtract from all row indices (start_row and end_row). If None, no row offset is applied.
    /// * `col_offset` - The number of columns to subtract from all column indices (start_col and end_col). If None, no column offset is applied.
    ///
    /// ### Returns
    /// A new `Indices` instance with the offsets applied. Values are clamped to 0 if the offset
    /// would result in a negative value.
    ///
    /// ### Examples
    /// ```
    /// # use crate::models::pipeline::Indices;
    /// // Basic offset example
    /// let indices = Indices { start_row: 5, start_col: 3, end_row: 10, end_col: 8 };
    /// let offset_indices = indices.with_offset(Some(2), Some(1));
    /// // offset_indices = Indices { start_row: 3, start_col: 2, end_row: 8, end_col: 7 }
    ///
    /// // Clamping to prevent negative values
    /// let small_indices = Indices { start_row: 1, start_col: 0, end_row: 3, end_col: 2 };
    /// let clamped_indices = small_indices.with_offset(Some(5), Some(1));
    /// // clamped_indices = Indices { start_row: 0, start_col: 0, end_row: 0, end_col: 1 }
    ///
    /// // No offset applied when None is passed
    /// let unchanged = indices.with_offset(None, None);
    /// // unchanged equals original indices
    /// ```
    pub fn with_offset(self, row_offset: Option<u32>, col_offset: Option<u32>) -> Self {
        let row_offset_val = row_offset.unwrap_or(0) as usize;
        let col_offset_val = col_offset.unwrap_or(0) as usize;

        Self {
            start_row: self.start_row.saturating_sub(row_offset_val),
            start_col: self.start_col.saturating_sub(col_offset_val),
            end_row: self.end_row.saturating_sub(row_offset_val),
            end_col: self.end_col.saturating_sub(col_offset_val),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SheetInfo {
    pub name: String,
    pub start_row: Option<u32>,
    pub start_column: Option<u32>,
    pub end_row: Option<u32>,
    pub end_column: Option<u32>,
    pub sheet_number: u32,
}

#[derive(Debug, Clone)]
pub struct Element {
    pub element_id: String,
    pub range: String,
    pub header_range: Option<String>,
    pub html: Option<Arc<NamedTempFile>>,
    pub capture: Option<Capture>,
    pub segment_type: SegmentType,
}

impl Element {
    /// Create a new Element
    ///
    /// The element will create the HTML and render it for the image.
    ///
    /// ### Arguments
    /// * `range` - The range of the element in Excel notation (e.g., "A1:D10")
    /// * `header_range` - The range of the element header in Excel notation (e.g., "A1:D1")
    ///
    /// ### Returns
    /// A new Table
    pub fn new(
        range: String,
        mut header_range: Option<String>,
        segment_type: SegmentType,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let header_indices = header_range
            .as_ref()
            .map(|header_range| parse_range(header_range))
            .transpose()?;
        let table_indices = parse_range(&range)?;

        // If the header range is inside the table range and the header start row is not the first row, remove the header range
        if header_indices.is_some() {
            let header_indices = header_indices.unwrap();
            let header_inside_table = header_indices.start_row >= table_indices.start_row
                && header_indices.end_row <= table_indices.end_row
                && header_indices.start_col >= table_indices.start_col
                && header_indices.end_col <= table_indices.end_col;
            let header_start_row_not_first_row = header_indices.start_row > table_indices.start_row;
            if header_inside_table && header_start_row_not_first_row {
                header_range = None;
            }
        }

        Ok(Self {
            element_id: generate_uuid(),
            range,
            header_range,
            html: None,
            capture: None,
            segment_type,
        })
    }

    /// Create a new Table with ranges converted from row/column indices
    ///
    /// ### Arguments
    /// * `start_row` - Zero-based starting row index
    /// * `start_col` - Zero-based starting column index
    /// * `end_row` - Zero-based ending row index (inclusive)
    /// * `end_col` - Zero-based ending column index (inclusive)
    /// * `header_end_row` - Optional zero-based ending row index for header (inclusive)
    ///
    /// ### Returns
    /// A new Table with properly formatted Excel ranges
    pub fn from_indices(
        table_indices: Indices,
        header_indices: Option<Indices>,
        segment_type: SegmentType,
    ) -> Self {
        let table_range = indices_to_range(&table_indices);
        let header_range = header_indices.map(|indices| indices_to_range(&indices));

        Self {
            element_id: generate_uuid(),
            range: table_range,
            header_range,
            html: None,
            capture: None,
            segment_type,
        }
    }

    /// Remove row overlap between header and table ranges
    ///
    /// The header range is never changed. If there's overlap in rows,
    /// the table range is adjusted to start after the header ends.
    ///
    /// ### Arguments
    /// * `header_range` - The header range string (e.g., "A1:D1")
    /// * `table_range` - The table range string (e.g., "A1:D8")
    ///
    /// ### Returns
    /// Indices of the table without overlap
    ///
    /// ### Example
    /// ```
    /// let table = Table::from_indices(Indices { start_row: 0, start_col: 0, end_row: 4, end_col: 3 }, Some(Indices { start_row: 0, start_col: 0, end_row: 0, end_col: 3 }));
    /// let table_without_overlap = table.table_indices_without_overlap()?;
    /// // table_without_overlap = Indices { start_row: 1, start_col: 0, end_row: 4, end_col: 3 }
    /// ```
    fn table_indices_without_overlap(
        &mut self,
    ) -> Result<Option<Indices>, Box<dyn Error + Send + Sync>> {
        let table_indices = parse_range(&self.range)?;
        if self.header_range.is_none() {
            return Ok(Some(table_indices));
        }

        let header_indices = parse_range(self.header_range.as_ref().unwrap())?;
        parse_range(&self.range)?;

        // Check if there's row overlap
        if table_indices.start_row <= header_indices.end_row
            && table_indices.end_row >= header_indices.start_row
        {
            // There's overlap, adjust table to start after header ends
            let new_table_start_row = header_indices.end_row + 1;

            // If the adjusted table start is beyond the table end, return empty table range
            if new_table_start_row > table_indices.end_row {
                return Ok(None);
            }

            let adjusted_table_range = Indices {
                start_row: new_table_start_row,
                start_col: table_indices.start_col,
                end_row: table_indices.end_row,
                end_col: table_indices.end_col,
            };

            Ok(Some(adjusted_table_range))
        } else {
            // No overlap, return indices as-is
            Ok(Some(table_indices))
        }
    }

    /// Calculate header alignment parameters
    ///
    /// Determines if header should be aligned with table body and calculates padding needed.
    ///
    /// ### Arguments
    /// * `header_range` - The header range string (e.g., "C3:K3")
    /// * `table_range` - The table range string (e.g., "A4:J20")
    ///
    /// ### Returns
    /// Option<(left_padding, right_padding)> where padding values are number of empty cells to add
    /// Returns None if alignment is not possible or not sensible
    fn calculate_header_alignment(
        &self,
        header_range: &str,
        table_range: &str,
    ) -> Result<Option<(usize, usize)>, Box<dyn Error + Send + Sync>> {
        let header_indices = parse_range(header_range)?;
        let table_indices = parse_range(table_range)?;

        let header_col_count = header_indices.end_col - header_indices.start_col + 1;
        let table_col_count = table_indices.end_col - table_indices.start_col + 1;

        // If header is larger than table, don't align
        if header_col_count > table_col_count {
            return Ok(None);
        }

        // If header is completely to the left of the table, don't align
        if header_indices.end_col < table_indices.start_col {
            return Ok(None);
        }

        // If header is completely to the right of the table, don't align
        if header_indices.start_col > table_indices.end_col {
            return Ok(None);
        }

        // Calculate left padding (how many empty cells to add at the start)
        let left_padding = header_indices
            .start_col
            .saturating_sub(table_indices.start_col);

        // Calculate right padding (how many empty cells to add at the end)
        let right_padding = table_indices.end_col.saturating_sub(header_indices.end_col);

        Ok(Some((left_padding, right_padding)))
    }

    /// Add padding cells to header HTML
    ///
    /// Adds empty cells to the beginning and end of header rows to align with table body
    ///
    /// ### Arguments
    /// * `header_html` - Original header HTML
    /// * `left_padding` - Number of empty cells to add at the beginning
    /// * `right_padding` - Number of empty cells to add at the end
    ///
    /// ### Returns
    /// Modified header HTML with padding cells added
    fn add_padding_to_header(
        &self,
        header_html: &str,
        left_padding: usize,
        right_padding: usize,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        if left_padding == 0 && right_padding == 0 {
            return Ok(header_html.to_string());
        }

        // Use string replacement to handle HTML where <tr> and </tr> might be on the same line
        let mut result = header_html.to_string();

        // Generate padding cells
        let left_padding_cells = (0..left_padding)
            .map(|_| "<td></td>")
            .collect::<Vec<_>>()
            .join("");
        let right_padding_cells = (0..right_padding)
            .map(|_| "<td></td>")
            .collect::<Vec<_>>()
            .join("");

        // Find and replace <tr> with <tr> + left padding cells (place padding INSIDE the tr tag)
        if left_padding > 0 {
            result = result.replace("<tr>", &format!("<tr>{left_padding_cells}"));
        }

        // Find and replace </tr> with right padding cells + </tr> (place padding INSIDE the tr tag)
        if right_padding > 0 {
            result = result.replace("</tr>", &format!("{right_padding_cells}</tr>"));
        }

        Ok(result)
    }

    pub fn create_html(
        &mut self,
        sheet_html: &str,
        row_offset: Option<u32>,
        col_offset: Option<u32>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut table_html = "<table>".to_string();

        // Check feature flag to include headers
        let include_headers = feature_config::Config::from_env()?.include_excel_headers;

        if !include_headers {
            // When headers are disabled, just render the table range as-is without header processing
            let indices = parse_range(&self.range)?.with_offset(row_offset, col_offset);
            table_html += &extract_rows_from_indicies(sheet_html, &indices, Some("<tbody>"))?;
            table_html += "</table>";
            self.html = Some(Arc::new(Builder::new().suffix(".html").tempfile()?));
            fs::write(self.html.as_ref().unwrap().path(), table_html)?;
            return Ok(());
        }

        let table_indices_without_overlap = self.table_indices_without_overlap()?;

        if self.header_range.is_none() && table_indices_without_overlap.is_none() {
            return Err("Table has no header and no table without overlap".into());
        }

        // Handle header with potential alignment
        if let Some(header_range) = &self.header_range {
            let indices = parse_range(header_range)?.with_offset(row_offset, col_offset);
            let mut header_html =
                extract_rows_from_indicies(sheet_html, &indices, Some("<thead>"))?;

            // Try to align header with table body if table exists
            if let Some(table_indices) = &table_indices_without_overlap {
                if let Ok(Some((left_padding, right_padding))) =
                    self.calculate_header_alignment(header_range, &indices_to_range(table_indices))
                {
                    // Apply alignment by adding padding cells
                    header_html =
                        self.add_padding_to_header(&header_html, left_padding, right_padding)?;
                }
            }

            table_html += &header_html;
        }

        // Handle table body
        if let Some(table_indices) = table_indices_without_overlap {
            let indices = table_indices.with_offset(row_offset, col_offset);
            table_html += &extract_rows_from_indicies(sheet_html, &indices, Some("<tbody>"))?;
        }

        table_html += "</table>";
        self.html = Some(Arc::new(Builder::new().suffix(".html").tempfile()?));
        fs::write(self.html.as_ref().unwrap().path(), table_html)?;
        Ok(())
    }
}

impl From<SheetInfo> for Element {
    /// Convert a SheetInfo into an Table
    ///
    /// This conversion treats the entire sheet range (starting from the first filled cell to the last filled cell) as the table range
    /// and does not set a header range
    ///
    /// ### Arguments
    /// * `sheet_info` - The SheetInfo to convert
    ///
    /// ### Returns
    /// A Table
    fn from(sheet_info: SheetInfo) -> Self {
        Self::from_indices(
            Indices {
                start_row: sheet_info.start_row.unwrap_or(0) as usize,
                start_col: sheet_info.start_column.unwrap_or(0) as usize,
                end_row: sheet_info.end_row.unwrap_or(0) as usize,
                end_col: sheet_info.end_column.unwrap_or(0) as usize,
            },
            None,
            SegmentType::Table,
        )
    }
}

impl From<SheetInfo> for Vec<Element> {
    /// Convert a SheetInfo into an Table
    ///
    /// This conversion treats the entire sheet range (starting from the first filled cell to the last filled cell) as the table range
    /// and does not set a header range
    ///
    /// ### Arguments
    /// * `sheet_info` - The SheetInfo to convert
    ///
    /// ### Returns
    /// A list of Tables
    fn from(sheet_info: SheetInfo) -> Self {
        vec![sheet_info.into()]
    }
}

impl TryFrom<IdentifiedElement> for Element {
    type Error = Box<dyn Error + Send + Sync>;

    /// Convert an IdentifiedElement into a Element
    ///
    /// ### Arguments
    /// * `identified_element` - The IdentifiedElement to convert
    ///
    /// ### Returns
    fn try_from(identified_element: IdentifiedElement) -> Result<Self, Self::Error> {
        Element::new(
            identified_element.range,
            identified_element.header_range,
            identified_element.r#type.try_into()?,
        )
    }
}

impl TryFrom<IdentifiedElements> for Vec<Element> {
    type Error = Box<dyn Error + Send + Sync>;
    /// Convert a list of IdentifiedElements into a list of Elements
    ///
    /// ### Arguments
    /// * `identified_elements` - The IdentifiedElements to convert
    ///
    /// ### Returns
    /// A list of Elements
    fn try_from(identified_elements: IdentifiedElements) -> Result<Self, Self::Error> {
        identified_elements
            .elements
            .into_iter()
            .map(|element| element.try_into())
            .collect::<Result<Vec<_>, _>>()
    }
}

#[derive(Debug, Clone)]
pub enum ReadingPattern {
    RowBased,
    ColumnBased,
}

/// Represents an individual sheet extracted from Excel HTML
#[derive(Debug, Clone)]
pub struct Sheet {
    pub html_file: Arc<NamedTempFile>,
    pub html_file_with_headers: Arc<NamedTempFile>,
    pub sheet_capture: Capture,
    pub sheet_capture_with_headers: Capture,
    pub embedded_images: Vec<ImageConversionResult>,
    pub elements: Option<Vec<Element>>,
    pub sheet_info: SheetInfo,
    pub pdf_file: Option<Arc<NamedTempFile>>,
}

impl Sheet {
    /// Create a new sheet
    pub fn new(
        sheet_info: SheetInfo,
        html_content: String,
        embedded_images: Vec<ImageConversionResult>,
        tracer: &opentelemetry::global::BoxedTracer,
        sheet_info_context: &Context,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Create temporary HTML files for processing
        let mut span = tracer.start_with_context(
            otel_config::SpanName::CreateHtmlFiles.to_string(),
            sheet_info_context,
        );
        let base_html_file = Self::create_temp_html_file().inspect_err(|e| {
            span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            span.record_error(e.as_ref());
            span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;
        let headers_html_file = Self::create_temp_html_file().inspect_err(|e| {
            span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            span.record_error(e.as_ref());
            span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;
        span.end();

        // Process HTML content with different enhancements in parallel
        let span = tracer.start_with_context(
            otel_config::SpanName::AddEnhancementsToHtml.to_string(),
            sheet_info_context,
        );
        let add_enhancements_context = sheet_info_context.with_span(span);
        let (html_with_cell_refs, html_with_headers) = rayon::join(
            || {
                let mut span = tracer.start_with_context(
                    otel_config::SpanName::AddCellReferencesToHtml.to_string(),
                    &add_enhancements_context,
                );
                Self::add_cell_references(&html_content, &sheet_info).inspect_err(|e| {
                    span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                    span.record_error(e.as_ref());
                    span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
                })
            },
            || {
                let mut span = tracer.start_with_context(
                    otel_config::SpanName::AddHeadersToHtml.to_string(),
                    &add_enhancements_context,
                );
                Self::add_headers(&html_content, &sheet_info).inspect_err(|e| {
                    span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                    span.record_error(e.as_ref());
                    span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
                })
            },
        );
        let html_with_cell_refs = html_with_cell_refs.inspect_err(|e| {
            let span = add_enhancements_context.span();
            span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            span.record_error(e.as_ref());
            span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;
        let html_with_headers = html_with_headers.inspect_err(|e| {
            let span = add_enhancements_context.span();
            span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            span.record_error(e.as_ref());
            span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;
        add_enhancements_context.span().end();

        // Filter embedded images to only include those referenced in the HTML
        let mut span = tracer.start_with_context(
            otel_config::SpanName::FilterRelevantImages.to_string(),
            sheet_info_context,
        );
        let mut relevant_embedded_images =
            Self::filter_relevant_images(&html_with_headers, embedded_images);
        span.end();

        let mut span = tracer.start_with_context(
            otel_config::SpanName::SetRelevantImagesRange.to_string(),
            sheet_info_context,
        );
        relevant_embedded_images.iter_mut().for_each(|image| {
            let cell_ref =
                get_cell_ref_for_image_src(&html_with_cell_refs, &image.html_reference).unwrap();
            image.set_range(cell_ref);
        });
        span.end();

        // Write processed HTML to files and generate images in parallel
        let span = tracer.start_with_context(
            otel_config::SpanName::CreateSheetImages.to_string(),
            sheet_info_context,
        );
        let create_sheet_images_context = sheet_info_context.with_span(span);
        let (sheet_capture, sheet_capture_with_headers) = rayon::join(
            || {
                let mut span = tracer.start_with_context(
                    otel_config::SpanName::CreateSheetImageCellReferences.to_string(),
                    &create_sheet_images_context,
                );
                Self::create_sheet_image(&base_html_file, &html_with_cell_refs).inspect_err(|e| {
                    span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                    span.record_error(e.as_ref());
                    span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
                })
            },
            || {
                let mut span = tracer.start_with_context(
                    otel_config::SpanName::CreateSheetImageHeaders.to_string(),
                    &create_sheet_images_context,
                );
                Self::create_sheet_image(&headers_html_file, &html_with_headers).inspect_err(|e| {
                    span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                    span.record_error(e.as_ref());
                    span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
                })
            },
        );
        let sheet_capture = sheet_capture.inspect_err(|e| {
            let span = create_sheet_images_context.span();
            span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            span.record_error(e.as_ref());
            span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;
        let sheet_capture_with_headers = sheet_capture_with_headers.inspect_err(|e| {
            let span = create_sheet_images_context.span();
            span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            span.record_error(e.as_ref());
            span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;
        create_sheet_images_context.span().end();

        Ok(Self {
            html_file: Arc::new(base_html_file),
            html_file_with_headers: Arc::new(headers_html_file),
            sheet_capture,
            sheet_capture_with_headers,
            embedded_images: relevant_embedded_images,
            elements: None,
            sheet_info,
            pdf_file: None,
        })
    }

    /// Create a temporary HTML file
    fn create_temp_html_file() -> Result<NamedTempFile, Box<dyn Error + Send + Sync>> {
        Builder::new()
            .suffix(".html")
            .tempfile()
            .map_err(|e| e.into())
    }

    /// Add cell references to HTML content
    fn add_cell_references(
        html_content: &str,
        sheet_info: &SheetInfo,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        add_cell_references_to_html(
            html_content,
            sheet_info.start_row.unwrap_or(0) as usize,
            sheet_info.start_column.unwrap_or(0) as usize,
        )
    }

    /// Add Excel headers to HTML content
    fn add_headers(
        html_content: &str,
        sheet_info: &SheetInfo,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        add_excel_headers_to_html(
            html_content,
            sheet_info.start_column.unwrap_or(0) as usize,
            sheet_info.start_row.unwrap_or(0) as usize,
        )
    }

    /// Filter embedded images to only include those referenced in the HTML
    fn filter_relevant_images(
        html_content: &str,
        embedded_images: Vec<ImageConversionResult>,
    ) -> Vec<ImageConversionResult> {
        let referenced_image_sources = get_img_sources(html_content);
        embedded_images
            .into_iter()
            .filter(|img| referenced_image_sources.contains(&img.html_reference))
            .collect()
    }

    /// Create a sheet image from HTML content
    fn create_sheet_image(
        html_file: &NamedTempFile,
        html_content: &str,
    ) -> Result<Capture, Box<dyn Error + Send + Sync>> {
        fs::write(html_file.path(), html_content)?;
        render_html_to_image(html_file).map_err(|e| e.into())
    }

    /// Set the tables for the sheet
    ///
    /// This method processes each table by creating HTML content from the sheet's HTML,
    /// applying necessary row offsets for proper cell reference alignment.
    ///
    /// ### Arguments
    /// * `tables` - The tables to set for this sheet
    ///
    /// ### Returns
    /// A Result indicating success or failure
    ///
    /// ### Note
    /// - **Row offset**: Uses `self.sheet_info.start_row` because rows don't have identifying
    ///   attributes and need manual offset adjustment for LibreOffice-generated HTML
    /// - **Column offset**: Set to 0 because we use `data-cell-ref` attributes to identify
    ///   cells within the desired range, and these attributes are already properly offset
    ///
    /*
    TODO: Add row attributes (e.g., `data-row-ref`) to HTML generation to make row identification
    robust like column identification, eliminating the need for manual offsets that could
    break if we replace LibreOffice with a different HTML generation method.
    */
    pub fn set_elements(
        &mut self,
        mut elements: Vec<Element>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        elements.par_iter_mut().try_for_each(
            |element| -> Result<(), Box<dyn Error + Send + Sync>> {
                let sheet_html = fs::read_to_string(self.html_file.path())?;
                element.create_html(&sheet_html, self.sheet_info.start_row, Some(0))?;
                Ok(())
            },
        )?;
        self.elements = Some(elements);
        Ok(())
    }

    /// Get HTML file from a table using table_id (which corresponds to segment_id)
    ///
    /// ### Arguments
    /// * `table_id` - The table_id (same as segment_id for table segments)
    ///
    /// ### Returns
    /// Option containing Arc<NamedTempFile> of the HTML file, or None if not found
    pub fn get_table_html_by_id(&self, table_id: &str) -> Option<Arc<NamedTempFile>> {
        self.elements.as_ref()?.iter().find_map(|table| {
            if table.element_id == table_id {
                table.html.clone()
            } else {
                None
            }
        })
    }

    /// Get image file from embedded images using image_id (which corresponds to segment_id)
    ///
    /// ### Arguments
    /// * `image_id` - The image_id (same as segment_id for picture segments)
    ///
    /// ### Returns
    /// Option containing Arc<NamedTempFile> of the image file, or None if not found
    pub fn get_embedded_image_by_id(&self, image_id: &str) -> Option<Arc<NamedTempFile>> {
        self.embedded_images.iter().find_map(|image| {
            if image.image_id == image_id {
                Some(image.image_file.clone())
            } else {
                None
            }
        })
    }

    /// Determine reading pattern based on consecutive elements' spatial positions
    fn analyze_reading_pattern(chunks: &[Chunk]) -> ReadingPattern {
        if chunks.len() < 2 {
            return ReadingPattern::RowBased; // Default fallback
        }

        let mut row_based_score = 0;
        let mut col_based_score = 0;

        // Analyze consecutive pairs of chunks
        for pair in chunks.windows(2) {
            let bbox1 = &pair[0].segments[0].bbox;
            let bbox2 = &pair[1].segments[0].bbox;

            let (center1_x, center1_y) = bbox1.centroid();
            let (center2_x, center2_y) = bbox2.centroid();

            // Tolerance for considering elements on the same row/column
            const ALIGNMENT_TOLERANCE: f32 = 20.0;

            // Check if they are roughly on the same row (row-based reading)
            if (center1_y - center2_y).abs() <= ALIGNMENT_TOLERANCE {
                if center2_x > center1_x {
                    row_based_score += 1; // Left to right movement
                }
            }
            // Check if they are roughly in the same column (column-based reading)
            else if (center1_x - center2_x).abs() <= ALIGNMENT_TOLERANCE {
                if center2_y > center1_y {
                    col_based_score += 1; // Top to bottom movement
                }
            }
            // Check for row-based pattern with line breaks
            else if center2_y > center1_y && center2_x < center1_x {
                row_based_score += 1; // Next row, back to left
            }
            // Check for column-based pattern with column breaks
            else if center2_x > center1_x && center2_y < center1_y {
                col_based_score += 1; // Next column, back to top
            }
        }

        if row_based_score >= col_based_score {
            ReadingPattern::RowBased
        } else {
            ReadingPattern::ColumnBased
        }
    }

    /// Find the best insertion point for a new chunk based on reading pattern
    fn find_insertion_point(
        new_chunk: &Chunk,
        existing_chunks: &[Chunk],
        pattern: ReadingPattern,
    ) -> usize {
        if existing_chunks.is_empty() {
            return 0;
        }

        let new_bbox = &new_chunk.segments[0].bbox;
        let (new_x, new_y) = new_bbox.centroid();

        for (i, existing_chunk) in existing_chunks.iter().enumerate() {
            let existing_bbox = &existing_chunk.segments[0].bbox;
            let (existing_x, existing_y) = existing_bbox.centroid();

            let should_insert_before = match pattern {
                ReadingPattern::RowBased => {
                    // For row-based: insert before if new element is above, or on same row but to the left
                    const ROW_TOLERANCE: f32 = 20.0;
                    if new_y < existing_y - ROW_TOLERANCE {
                        true // New element is clearly above
                    } else if (new_y - existing_y).abs() <= ROW_TOLERANCE {
                        new_x < existing_x // Same row, check horizontal position
                    } else {
                        false // New element is below
                    }
                }
                ReadingPattern::ColumnBased => {
                    // For column-based: insert before if new element is to the left, or in same column but above
                    const COL_TOLERANCE: f32 = 20.0;
                    if new_x < existing_x - COL_TOLERANCE {
                        true // New element is clearly to the left
                    } else if (new_x - existing_x).abs() <= COL_TOLERANCE {
                        new_y < existing_y // Same column, check vertical position
                    } else {
                        false // New element is to the right
                    }
                }
            };

            if should_insert_before {
                return i;
            }
        }

        // If we didn't find a position to insert before, insert at the end
        existing_chunks.len()
    }

    /// Insert new chunks while preserving the original reading order
    fn insert_chunks_preserving_order(
        original_chunks: Vec<Chunk>,
        new_chunks: Vec<Chunk>,
        pattern: ReadingPattern,
    ) -> Vec<Chunk> {
        let mut result = original_chunks;

        // Sort new chunks by reading order first to ensure consistent insertion
        let mut sorted_new_chunks = new_chunks;

        // Try to sort with panic catching
        let sort_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sorted_new_chunks.sort_by(|a, b| {
                let bbox_a = &a.segments[0].bbox;
                let bbox_b = &b.segments[0].bbox;
                let (x_a, y_a) = bbox_a.centroid();
                let (x_b, y_b) = bbox_b.centroid();

                match pattern {
                    ReadingPattern::RowBased => {
                        // Always sort by Y first, then by X (maintains total order)
                        let y_cmp = y_a.partial_cmp(&y_b).unwrap_or(std::cmp::Ordering::Equal);
                        if y_cmp == std::cmp::Ordering::Equal {
                            x_a.partial_cmp(&x_b).unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            y_cmp
                        }
                    }
                    ReadingPattern::ColumnBased => {
                        // Always sort by X first, then by Y (maintains total order)
                        let x_cmp = x_a.partial_cmp(&x_b).unwrap_or(std::cmp::Ordering::Equal);
                        if x_cmp == std::cmp::Ordering::Equal {
                            y_a.partial_cmp(&y_b).unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            x_cmp
                        }
                    }
                }
            });
        }));

        match sort_result {
            Ok(_) => {
                // Sorting succeeded, continue normally
            }
            Err(_) => {
                // This should no longer happen with the fixed comparison function
                eprintln!("WARNING: Sorting panic occurred despite fixed comparison function");

                // Fallback: Use safe sorting that handles NaN/infinite values
                sorted_new_chunks.sort_by(|a, b| {
                    let bbox_a = &a.segments[0].bbox;
                    let bbox_b = &b.segments[0].bbox;
                    let (x_a, y_a) = bbox_a.centroid();
                    let (x_b, y_b) = bbox_b.centroid();

                    // Safe comparison function that handles NaN/infinite
                    fn safe_f32_cmp(a: f32, b: f32) -> std::cmp::Ordering {
                        match (a.is_finite(), b.is_finite()) {
                            (true, true) => a.partial_cmp(&b).unwrap(),
                            (false, true) => std::cmp::Ordering::Greater, // NaN/inf sorts last
                            (true, false) => std::cmp::Ordering::Less,
                            (false, false) => std::cmp::Ordering::Equal,
                        }
                    }

                    match pattern {
                        ReadingPattern::RowBased => {
                            let y_cmp = safe_f32_cmp(y_a, y_b);
                            if y_cmp == std::cmp::Ordering::Equal
                                || (y_a.is_finite() && y_b.is_finite() && (y_a - y_b).abs() <= 20.0)
                            {
                                safe_f32_cmp(x_a, x_b)
                            } else {
                                y_cmp
                            }
                        }
                        ReadingPattern::ColumnBased => {
                            let x_cmp = safe_f32_cmp(x_a, x_b);
                            if x_cmp == std::cmp::Ordering::Equal
                                || (x_a.is_finite() && x_b.is_finite() && (x_a - x_b).abs() <= 20.0)
                            {
                                safe_f32_cmp(y_a, y_b)
                            } else {
                                x_cmp
                            }
                        }
                    }
                });
            }
        }

        // Insert each new chunk at its appropriate position
        for new_chunk in sorted_new_chunks {
            let insertion_point = Self::find_insertion_point(&new_chunk, &result, pattern.clone());
            result.insert(insertion_point, new_chunk);
        }

        result
    }

    /// Check if two ranges overlap
    fn ranges_overlap(range1: &str, range2: &str) -> Result<bool, Box<dyn Error + Send + Sync>> {
        let indices1 = parse_range(range1)?;
        let indices2 = parse_range(range2)?;

        // Check if there's any overlap in rows and columns
        let row_overlap =
            indices1.start_row <= indices2.end_row && indices1.end_row >= indices2.start_row;
        let col_overlap =
            indices1.start_col <= indices2.end_col && indices1.end_col >= indices2.start_col;

        Ok(row_overlap && col_overlap)
    }

    pub fn to_chunks(
        self,
        tracer: &opentelemetry::global::BoxedTracer,
        context: &Context,
    ) -> Result<Vec<Chunk>, Box<dyn Error + Send + Sync>> {
        let html_content = fs::read_to_string(self.html_file.path()).unwrap();
        let sheet_range = indices_to_range(&Indices {
            start_row: self.sheet_info.start_row.unwrap_or(0) as usize,
            start_col: self.sheet_info.start_column.unwrap_or(0) as usize,
            end_row: self.sheet_info.end_row.unwrap_or(0) as usize,
            end_col: self.sheet_info.end_column.unwrap_or(0) as usize,
        });
        let all_cells = extract_cells_from_ranges(&html_content, Some(&sheet_range), None).unwrap();

        // Convert elements to chunks (preserving LLM reading order)
        let mut element_chunks = Vec::new();
        if let Some(elements) = self.elements {
            // Get embedded image ranges for filtering
            let embedded_image_ranges: Vec<String> = self
                .embedded_images
                .iter()
                .filter_map(|img| img.range.clone())
                .collect();

            // Filter out picture elements that overlap with embedded images
            let filtered_elements: Vec<Element> = elements
                .into_par_iter()
                .filter(|element| {
                    // Keep all non-picture elements
                    if element.segment_type != SegmentType::Picture {
                        return true;
                    }

                    // For picture elements, check if they overlap with any embedded image
                    !embedded_image_ranges.par_iter().any(|embedded_range| {
                        Self::ranges_overlap(&element.range, embedded_range).unwrap_or(false)
                    })
                })
                .collect();
            let mut span = tracer.start_with_context(
                otel_config::SpanName::ConvertTablesToChunks.to_string(),
                context,
            );
            span.set_attribute(opentelemetry::KeyValue::new(
                "element_count",
                filtered_elements.len() as i64,
            ));

            let element_result: Result<Vec<Chunk>, _> = filtered_elements
                .into_par_iter()
                .filter_map(
                    |element| -> Option<Result<Chunk, Box<dyn Error + Send + Sync>>> {
                        match Segment::new_from_elements(
                            element.element_id.clone(),
                            Some(1.0),
                            element.header_range,
                            self.sheet_capture.height as f32,
                            self.sheet_capture.width as f32,
                            self.sheet_info.sheet_number,
                            Some(element.range.clone()),
                            self.sheet_info.name.clone(),
                            html_content.clone(),
                            self.sheet_capture.clone(),
                            element.segment_type,
                            tracer,
                            context,
                        ) {
                            Ok(segment) => Some(Ok(Chunk::new(vec![segment]))),
                            Err(SegmentCreationError::NoBoundingBox(msg)) => {
                                println!(
                                    "Skipping element as no bounding box found {} - {}",
                                    element.element_id, msg
                                );
                                None
                            }
                            Err(e) => Some(Err(e.into())),
                        }
                    },
                )
                .collect();

            element_chunks = element_result.inspect_err(|e| {
                span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                span.record_error(e.as_ref());
                span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
            })?;
            span.end();
        }

        // Analyze reading pattern from element chunks
        let reading_pattern = Self::analyze_reading_pattern(&element_chunks);

        // Convert embedded images to chunks
        let mut span = tracer.start_with_context(
            otel_config::SpanName::ConvertImagesToChunks.to_string(),
            context,
        );

        let image_result: Result<Vec<Chunk>, _> = self
            .embedded_images
            .clone()
            .into_par_iter()
            .filter_map(
                |image| -> Option<Result<Chunk, Box<dyn Error + Send + Sync>>> {
                    let range = match image.range.clone() {
                        Some(range) => range,
                        None => {
                            return Some(Err("No range found".into()));
                        }
                    };

                    match Segment::new_from_images(
                        image.image_id.clone(),
                        Some(1.0),
                        self.sheet_capture.height as f32,
                        self.sheet_capture.width as f32,
                        self.sheet_info.sheet_number,
                        range,
                        self.sheet_info.name.clone(),
                        html_content.clone(),
                        self.sheet_capture.clone(),
                        image.html_reference.clone(),
                        tracer,
                        context,
                    ) {
                        Ok(segment) => Some(Ok(Chunk::new(vec![segment]))),
                        Err(SegmentCreationError::NoBoundingBox(_msg)) => None,
                        Err(e) => Some(Err(e.into())),
                    }
                },
            )
            .collect();

        let image_chunks = image_result.inspect_err(|e| {
            span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            span.record_error(e.as_ref());
            span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;

        span.set_attribute(opentelemetry::KeyValue::new(
            "image_count",
            self.embedded_images.clone().len() as i64,
        ));
        span.end();

        // Get all cell ranges that are already used by table and image segments
        let mut all_existing_chunks = element_chunks.clone();
        all_existing_chunks.extend(image_chunks.clone());

        let used_cell_ranges: Vec<String> = all_existing_chunks
            .iter()
            .flat_map(|chunk| &chunk.segments)
            .filter_map(|segment| segment.ss_cells.as_ref())
            .flatten()
            .map(|cell| cell.range.clone())
            .collect();

        // Filter out cells that are already in table or image segments
        let remaining_cells: Vec<Cell> = all_cells
            .into_par_iter()
            .filter(|cell| !used_cell_ranges.contains(&cell.range))
            .collect();

        // Create chunks from remaining cells
        let mut span = tracer.start_with_context(
            otel_config::SpanName::ConvertRemainingCellsToChunks.to_string(),
            context,
        );

        let remaining_cells_result: Result<Vec<Chunk>, _> = remaining_cells
            .clone()
            .into_par_iter()
            .filter(|cell| !cell.text.trim().is_empty())
            .filter_map(
                |cell| -> Option<Result<Chunk, Box<dyn Error + Send + Sync>>> {
                    match Segment::new_from_remaining_cells(
                        generate_uuid(),
                        Some(1.0),
                        self.sheet_capture.height as f32,
                        self.sheet_capture.width as f32,
                        self.sheet_info.sheet_number,
                        cell.range.clone(),
                        self.sheet_info.name.clone(),
                        self.sheet_capture.clone(),
                        vec![cell],
                        tracer,
                        context,
                    ) {
                        Ok(segment) => Some(Ok(Chunk::new(vec![segment]))),
                        Err(SegmentCreationError::NoBoundingBox(_msg)) => None,
                        Err(e) => Some(Err(e.into())),
                    }
                },
            )
            .collect();

        let remaining_cell_chunks = remaining_cells_result.inspect_err(|e| {
            span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            span.record_error(e.as_ref());
            span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;

        span.set_attribute(opentelemetry::KeyValue::new(
            "remaining_cells_count",
            remaining_cells.clone().len() as i64,
        ));
        span.end();

        // Insert images and remaining cells while preserving element reading order
        let mut span = tracer.start_with_context(
            otel_config::SpanName::ApplyReadingOrder.to_string(),
            context,
        );

        // First, insert image chunks
        let chunks_with_images = Self::insert_chunks_preserving_order(
            element_chunks,
            image_chunks,
            reading_pattern.clone(),
        );

        // Then, insert remaining cell chunks
        let final_chunks = Self::insert_chunks_preserving_order(
            chunks_with_images,
            remaining_cell_chunks,
            reading_pattern.clone(),
        );

        span.set_attribute(opentelemetry::KeyValue::new(
            "final_chunk_count",
            final_chunks.len() as i64,
        ));
        span.end();

        Ok(final_chunks)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, EnumString)]
pub enum PipelineStep {
    #[cfg(feature = "azure")]
    #[strum(serialize = "azure_analysis")]
    AzureAnalysis,
    #[strum(serialize = "chunking")]
    Chunking,
    #[strum(serialize = "chunkr_analysis")]
    ChunkrAnalysis,
    #[strum(serialize = "convert_to_images")]
    ConvertToImages,
    #[strum(serialize = "crop")]
    Crop,
    #[strum(serialize = "segment_processing")]
    SegmentProcessing,
    #[strum(serialize = "convert_excel_to_html")]
    ConvertExcelToHtml,
    #[strum(serialize = "identify_elements_in_sheet")]
    IdentifyElementsInSheet,
}

pub trait PipelineStepMessages {
    fn start_message(&self) -> String;
    fn error_message(&self) -> String;
}

impl PipelineStepMessages for PipelineStep {
    fn start_message(&self) -> String {
        match self {
            #[cfg(feature = "azure")]
            PipelineStep::AzureAnalysis => "Running Azure analysis".to_string(),
            PipelineStep::Chunking => "Chunking".to_string(),
            PipelineStep::ChunkrAnalysis => "Running Chunkr analysis".to_string(),
            PipelineStep::ConvertToImages => "Converting pages to images".to_string(),
            PipelineStep::Crop => "Cropping segments".to_string(),
            PipelineStep::SegmentProcessing => "Processing segments".to_string(),
            PipelineStep::ConvertExcelToHtml => "Preparing spreadsheet for analysis".to_string(),
            PipelineStep::IdentifyElementsInSheet => "Analyzing tables in spreadsheet".to_string(),
        }
    }

    fn error_message(&self) -> String {
        match self {
            #[cfg(feature = "azure")]
            PipelineStep::AzureAnalysis => "Failed to run Azure analysis".to_string(),
            PipelineStep::Chunking => "Failed to chunk".to_string(),
            PipelineStep::ChunkrAnalysis => "Failed to run Chunkr analysis".to_string(),
            PipelineStep::ConvertToImages => "Failed to convert pages to images".to_string(),
            PipelineStep::Crop => "Failed to crop segments".to_string(),
            PipelineStep::SegmentProcessing => {
                "Failed to process segments - LLM processing error".to_string()
            }
            PipelineStep::ConvertExcelToHtml => {
                "Failed to prepare spreadsheet for analysis".to_string()
            }
            PipelineStep::IdentifyElementsInSheet => {
                "Failed to analyze tables in spreadsheet".to_string()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub input_file: Option<Arc<NamedTempFile>>,
    pub chunks: Vec<Chunk>,
    pub page_images: Option<Vec<Arc<NamedTempFile>>>,
    pub pdf_file: Option<Arc<NamedTempFile>>,
    pub segment_images: DashMap<String, Arc<NamedTempFile>>,
    pub task: Option<Task>,
    pub task_payload: Option<TaskPayload>,
    pub html_file: Option<Arc<NamedTempFile>>,
    pub sheets: Option<Vec<Sheet>>,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            input_file: None,
            chunks: Vec::new(),
            page_images: None,
            pdf_file: None,
            segment_images: DashMap::new(),
            task: None,
            task_payload: None,
            html_file: None,
            sheets: None,
        }
    }

    pub async fn init(&mut self, task_payload: TaskPayload) -> Result<(), Box<dyn Error>> {
        let mut task = Task::get(&task_payload.task_id, &task_payload.user_info.user_id).await?;
        task.update(
            Some(Status::Processing),
            Some("Task started".to_string()),
            None,
            None,
            Some(Utc::now()),
            None,
            None,
            None,
            false,
        )
        .await?;
        if task.status == Status::Cancelled {
            println!("Task was cancelled, checking for previous configuration");
            if task_payload.previous_configuration.is_some() {
                println!("Reverting to previous configuration");
                task.update(
                    task_payload.previous_status,
                    task_payload.previous_message,
                    task_payload.previous_configuration,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
                .await?;
            }
            return Ok(());
        }
        let is_spreadsheet = check_is_spreadsheet(
            task.mime_type
                .as_ref()
                .ok_or("No mime type found")?
                .as_str(),
        )?;
        self.input_file = Some(Arc::new(
            download_to_tempfile(&task.input_location, None, task.mime_type.as_ref().unwrap())
                .await?,
        ));

        self.pdf_file = match task.mime_type.as_ref().unwrap().as_str() {
            "application/pdf" => Some(self.input_file.clone().unwrap()),
            _ if is_spreadsheet => None, // PDF will be set later for spreadsheets
            _ => Some(Arc::new(convert_to_pdf(
                self.input_file.as_ref().unwrap(),
                None,
            )?)),
        };
        println!("Task initialized with input file");
        let page_count = if is_spreadsheet {
            count_sheets(
                self.input_file
                    .as_ref()
                    .ok_or("No input file found for page count")?
                    .path()
                    .to_str()
                    .ok_or("No input file path found for page count")
                    .map(Path::new)?,
            )?
        } else {
            count_pages(
                self.pdf_file
                    .as_ref()
                    .ok_or("No PDF file found for page count")?,
            )?
        };
        task.update(
            Some(Status::Processing),
            Some("Task initialized".to_string()),
            None,
            Some(page_count),
            None,
            None,
            None,
            None,
            false,
        )
        .await
        .map_err(|e| -> Box<dyn Error> { format!("Task update error: {e:?}").into() })?;
        self.task_payload = Some(task_payload.clone());
        self.task = Some(task.clone());
        Ok(())
    }

    pub async fn set_spreadsheet_assets(
        &mut self,
        mut sheets: Vec<Sheet>,
        html_file: Arc<NamedTempFile>,
        tracer: &opentelemetry::global::BoxedTracer,
        context: &Context,
    ) -> Result<(), Box<dyn Error>> {
        let mut task = self.get_task()?;
        task.update(
            Some(Status::Processing),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .await?;
        self.sheets = Some(sheets.clone());
        self.html_file = Some(html_file);
        let span = tracer.start_with_context(
            otel_config::SpanName::ConvertSheetImagesToPdf.to_string(),
            context,
        );
        let convert_sheet_context = context.with_span(span);
        sheets
            .par_iter_mut()
            .try_for_each(
                |sheet: &mut Sheet| -> Result<(), Box<dyn Error + Send + Sync>> {
                    let mut span = tracer.start_with_context(
                        otel_config::SpanName::ConvertImageToPdf.to_string(),
                        &convert_sheet_context,
                    );
                    span.set_attribute(opentelemetry::KeyValue::new(
                        "sheet_name",
                        sheet.sheet_info.name.clone(),
                    ));
                    span.set_attribute(opentelemetry::KeyValue::new(
                        "sheet_number",
                        sheet.sheet_info.sheet_number.to_string(),
                    ));
                    let pdf_file =
                        create_pdf_from_image(&sheet.sheet_capture.image).inspect_err(|e| {
                            span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                            span.record_error(e.as_ref());
                            span.set_attribute(opentelemetry::KeyValue::new(
                                "error",
                                e.to_string(),
                            ));
                        })?;
                    sheet.pdf_file = Some(Arc::new(pdf_file));
                    Ok(())
                },
            )
            .map_err(|e| -> Box<dyn Error> { e.to_string().into() })
            .inspect_err(|e| {
                let span = convert_sheet_context.span();
                span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                span.record_error(e.as_ref());
                span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
            })?;
        convert_sheet_context.span().end();

        let mut span =
            tracer.start_with_context(otel_config::SpanName::CombinePdfs.to_string(), context);
        self.pdf_file = Some(Arc::new(
            combine_pdfs(
                sheets
                    .iter()
                    .map(|sheet| sheet.pdf_file.as_ref().unwrap().as_ref())
                    .collect(),
            )
            .inspect_err(|e| {
                span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                span.record_error(e.as_ref());
                span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
            })?,
        ));
        span.end();

        // Add page images to output
        self.page_images = self
            .sheets
            .as_ref()
            .unwrap()
            .iter()
            .map(|sheet| Some(sheet.sheet_capture.image.clone()))
            .collect();

        Ok(())
    }

    pub fn get_task(&self) -> Result<Task, Box<dyn Error>> {
        self.task
            .as_ref()
            .ok_or_else(|| "Task is not initialized".into())
            .cloned()
    }

    pub fn get_task_payload(&self) -> Result<TaskPayload, Box<dyn Error>> {
        self.task_payload
            .as_ref()
            .ok_or_else(|| "Task payload is not initialized".into())
            .cloned()
    }

    pub fn get_mime_type(&self) -> Result<String, Box<dyn Error>> {
        Ok(self.get_task()?.mime_type.as_ref().unwrap().clone())
    }

    pub fn get_scaling_factor(&self) -> Result<f32, Box<dyn Error>> {
        if self.get_mime_type()?.starts_with("image/") {
            Ok(1.0)
        } else {
            let worker_config = worker_config::Config::from_env()?;
            if self.get_task()?.configuration.high_resolution {
                Ok(worker_config.high_res_scaling_factor)
            } else {
                Ok(1.0)
            }
        }
    }

    pub fn get_file(&self) -> Result<Arc<NamedTempFile>, Box<dyn Error>> {
        if self.get_mime_type()?.starts_with("image/") {
            Ok(self.input_file.as_ref().unwrap().clone())
        } else {
            Ok(self.pdf_file.as_ref().unwrap().clone())
        }
    }

    /// Convert all sheets to chunks using the TryFrom trait implementation
    ///
    /// This method iterates through all sheets and converts each one into chunks
    /// where each segment is either a table or picture.
    /// The segments within each chunk are sorted by reading order (left to right, top to bottom).
    pub fn sheets_to_chunks(
        &mut self,
        tracer: &opentelemetry::global::BoxedTracer,
        context: &Context,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let sheets = self.sheets.as_ref().ok_or("No sheets available")?;
        let span = tracer.start_with_context(
            otel_config::SpanName::ConvertSheetsToChunks.to_string(),
            context,
        );
        let context = context.with_span(span);
        self.chunks = sheets
            .par_iter()
            .map(
                |sheet| -> Result<Vec<Chunk>, Box<dyn Error + Send + Sync>> {
                    let mut span = tracer.start_with_context(
                        otel_config::SpanName::ConvertSheetToChunks.to_string(),
                        &context,
                    );
                    span.set_attribute(opentelemetry::KeyValue::new(
                        "sheet_name",
                        sheet.sheet_info.name.clone(),
                    ));
                    span.set_attribute(opentelemetry::KeyValue::new(
                        "sheet_number",
                        sheet.sheet_info.sheet_number.to_string(),
                    ));
                    let context = context.with_span(span);
                    sheet.clone().to_chunks(tracer, &context)
                },
            )
            .collect::<Result<Vec<Vec<Chunk>>, _>>()
            .inspect_err(|e| {
                let span = context.span();
                span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                span.record_error(e.as_ref());
                span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
            })?
            .into_iter()
            .flatten()
            .collect::<Vec<Chunk>>();
        Ok(())
    }

    /// Get HTML file from tables using segment_id across all sheets
    ///
    /// Since segment_id corresponds to table_id for table segments, this method
    /// searches all sheets for a table with the matching ID.
    ///
    /// ### Arguments
    /// * `segment_id` - The segment_id (same as table_id for table segments)
    ///
    /// ### Returns
    /// Option containing Arc<NamedTempFile> of the HTML file, or None if not found
    pub fn get_table_html_by_segment_id(&self, segment_id: &str) -> Option<Arc<NamedTempFile>> {
        let sheets = self.sheets.as_ref()?;

        for sheet in sheets {
            if let Some(html_file) = sheet.get_table_html_by_id(segment_id) {
                return Some(html_file);
            }
        }
        None
    }

    /// Get image file from embedded images using segment_id across all sheets
    ///
    /// Since segment_id corresponds to image_id for picture segments, this method
    /// searches all sheets for an embedded image with the matching ID.
    ///
    /// ### Arguments
    /// * `segment_id` - The segment_id (same as image_id for picture segments)
    ///
    /// ### Returns
    /// Option containing Arc<NamedTempFile> of the image file, or None if not found
    pub fn get_embedded_image_by_segment_id(&self, segment_id: &str) -> Option<Arc<NamedTempFile>> {
        let sheets = self.sheets.as_ref()?;

        for sheet in sheets {
            if let Some(image_file) = sheet.get_embedded_image_by_id(segment_id) {
                return Some(image_file);
            }
        }
        None
    }

    /// Get sheet by sheet number
    ///
    /// This method returns the sheet with the given sheet number.
    ///
    /// ### Arguments
    /// * `sheet_number` - The sheet number
    ///
    /// ### Returns
    /// Option containing &Sheet, or None if not found
    pub fn get_sheet_by_sheet_number(&self, sheet_number: u32) -> Option<&Sheet> {
        self.sheets.as_ref()?.get((sheet_number as usize) - 1)
    }

    /// Execute a step in the pipeline
    #[cfg_attr(feature = "memory_profiling", track_mem)]
    pub async fn execute_step(
        &mut self,
        step: PipelineStep,
        max_retries: u32,
        tracer: &opentelemetry::global::BoxedTracer,
    ) -> Result<(), Box<dyn Error>> {
        let start = std::time::Instant::now();
        let mut task = self.get_task()?;
        let mut retries = 0;
        let mut last_error: Option<String> = None;
        while retries < max_retries {
            let mut span = tracer.start_with_context(step.to_string(), &Context::current());
            span.set_attribute(opentelemetry::KeyValue::new(
                "retry_count",
                retries.to_string(),
            ));
            let _guard = Context::current().with_span(span).attach();

            // Update task status to processing and message to step start message
            let message = match retries > 0 {
                true => format!(
                    "{} | retry {}/{}",
                    step.start_message(),
                    retries + 1,
                    max_retries
                ),
                false => step.start_message(),
            };
            println!("Executing step: {message}");
            task.update(
                Some(Status::Processing),
                Some(message),
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .await?;

            // Execute step
            let result = match step {
                #[cfg(feature = "azure")]
                PipelineStep::AzureAnalysis => crate::pipeline::azure_analysis::process(self).await,
                PipelineStep::Chunking => crate::pipeline::chunking::process(self).await,
                PipelineStep::ConvertToImages => {
                    crate::pipeline::convert_to_images::process(self).await
                }
                PipelineStep::Crop => crate::pipeline::crop::process(self),
                PipelineStep::ChunkrAnalysis => {
                    crate::pipeline::chunkr_analysis::process(self).await
                }
                PipelineStep::SegmentProcessing => {
                    crate::pipeline::segment_processing::process(self, tracer).await
                }
                PipelineStep::ConvertExcelToHtml => {
                    crate::pipeline::convert_excel_to_html::process(self, tracer).await
                }
                PipelineStep::IdentifyElementsInSheet => {
                    match crate::pipeline::identify_elements_in_sheet::process(self, tracer).await {
                        Ok(_) => Ok(()),
                        Err(e) => Err(e),
                    }
                }
            };

            let duration = start.elapsed();

            // Check if step succeeded or failed
            match result {
                Ok(_) => {
                    println!(
                        "Step {} took {:?} with page count {:?}",
                        step,
                        duration,
                        self.get_task()?.page_count.unwrap_or(0)
                    );
                    Context::current().span().end();
                    return Ok(());
                }
                Err(e) => {
                    println!("Error {e} in step {step}");
                    last_error = Some(e.to_string());
                    retries += 1;
                    let context = Context::current();
                    context
                        .span()
                        .set_status(opentelemetry::trace::Status::error(e.to_string()));
                    context.span().record_error(e.as_ref());
                    context
                        .span()
                        .set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
                    context.span().end();

                    if retries < max_retries {
                        task.update(
                            Some(Status::Processing),
                            Some(step.error_message()),
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            false,
                        )
                        .await?;

                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    }
                }
            }
        }
        // If step failed after max_retries, complete task with failed status and error message
        self.complete(Status::Failed, Some(step.error_message()))
            .await?;
        Err(last_error
            .unwrap_or("Maximum retries exceeded".into())
            .into())
    }

    pub async fn complete(
        &mut self,
        status: Status,
        message: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        let mut task = self.get_task()?;
        let task_payload = self.get_task_payload()?;
        let finished_at = Utc::now();
        let expires_at = task
            .configuration
            .expires_in
            .map(|seconds| finished_at + chrono::Duration::seconds(seconds as i64));

        async fn revert_to_previous(
            task: &mut Task,
            payload: &TaskPayload,
        ) -> Result<(), Box<dyn Error>> {
            if payload.previous_configuration.is_some() {
                task.update(
                    payload.previous_status.clone(),
                    payload.previous_message.clone(),
                    payload.previous_configuration.clone(),
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
                .await?;
            } else {
                task.update(
                    Some(Status::Failed),
                    Some("Failed to upload files".to_string()),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
                .await?;
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        async fn update_success(
            task: &mut Task,
            status: Status,
            message: Option<String>,
            page_images: Vec<Arc<NamedTempFile>>,
            segment_images: &DashMap<String, Arc<NamedTempFile>>,
            chunks: Vec<Chunk>,
            pdf_file: Arc<NamedTempFile>,
            finished_at: DateTime<Utc>,
            expires_at: Option<DateTime<Utc>>,
        ) -> Result<(), Box<dyn Error>> {
            let segment_count = chunks.iter().map(|c| c.segments.len() as u32).sum();
            task.upload_artifacts(page_images, segment_images, chunks, &pdf_file)
                .await?;
            task.update(
                Some(status),
                message,
                None,
                None,
                None,
                Some(finished_at),
                expires_at,
                Some(segment_count),
                false,
            )
            .await?;
            Ok(())
        }

        if status == Status::Failed {
            if task_payload.previous_configuration.is_none() {
                task.update(
                    Some(status),
                    message,
                    None,
                    None,
                    None,
                    Some(finished_at),
                    expires_at,
                    None,
                    false,
                )
                .await?;
                Ok(())
            } else {
                revert_to_previous(&mut task, &task_payload).await
            }
        } else {
            match update_success(
                &mut task,
                status,
                message,
                self.page_images.clone().unwrap(),
                &self.segment_images,
                self.chunks.clone(),
                self.pdf_file.clone().unwrap(),
                finished_at,
                expires_at,
            )
            .await
            {
                Ok(_) => Ok(()),
                Err(e) => {
                    println!("Error in completing task: {e:?}");
                    revert_to_previous(&mut task, &task_payload).await?;
                    Err(e)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indices_to_range() {
        // Single cell
        let indices = Indices {
            start_row: 0,
            start_col: 0,
            end_row: 0,
            end_col: 0,
        };
        assert_eq!(indices_to_range(&indices), "A1");

        // Simple range
        let indices = Indices {
            start_row: 0,
            start_col: 0,
            end_row: 2,
            end_col: 3,
        };
        assert_eq!(indices_to_range(&indices), "A1:D3");

        // Starting from different position
        let indices = Indices {
            start_row: 1,
            start_col: 2,
            end_row: 3,
            end_col: 5,
        };
        assert_eq!(indices_to_range(&indices), "C2:F4");

        // Large column range
        let indices = Indices {
            start_row: 0,
            start_col: 26,
            end_row: 1,
            end_col: 28,
        };
        assert_eq!(indices_to_range(&indices), "AA1:AC2");
    }

    #[test]
    fn test_from_indices() {
        // Table without header
        let table = Element::from_indices(
            Indices {
                start_row: 0,
                start_col: 0,
                end_row: 4,
                end_col: 3,
            },
            None,
            SegmentType::Table,
        );
        assert_eq!(table.range, "A1:D5");
        assert_eq!(table.header_range, None);

        // Table with header
        let table_with_header = Element::from_indices(
            Indices {
                start_row: 0,
                start_col: 0,
                end_row: 4,
                end_col: 3,
            },
            Some(Indices {
                start_row: 0,
                start_col: 0,
                end_row: 0,
                end_col: 3,
            }),
            SegmentType::Table,
        );
        assert_eq!(table_with_header.range, "A1:D5");
        assert_eq!(table_with_header.header_range, Some("A1:D1".to_string()));

        // Table with multi-row header
        let table_multi_header = Element::from_indices(
            Indices {
                start_row: 0,
                start_col: 1,
                end_row: 6,
                end_col: 4,
            },
            Some(Indices {
                start_row: 0,
                start_col: 1,
                end_row: 1,
                end_col: 4,
            }),
            SegmentType::Table,
        );
        assert_eq!(table_multi_header.range, "B1:E7");
        assert_eq!(table_multi_header.header_range, Some("B1:E2".to_string()));
    }

    #[test]
    fn test_from_sheet_info() {
        let sheet_info = SheetInfo {
            name: "Sheet1".to_string(),
            start_row: Some(0),
            start_column: Some(1),
            end_row: Some(4),
            end_column: Some(3),
            sheet_number: 1,
        };

        let table: Element = sheet_info.into();
        assert_eq!(table.range, "B1:D5");
        assert_eq!(table.header_range, None);
    }

    #[test]
    fn test_create_html() {
        // Create mock sheet HTML content
        let sheet_html = r#"<table>
<tr>
<td data-cell-ref="A1">Header1</td>
<td data-cell-ref="B1">Header2</td>
<td data-cell-ref="C1">Header3</td>
</tr>
<tr>
<td data-cell-ref="A2">Data1</td>
<td data-cell-ref="B2">Data2</td>
<td data-cell-ref="C2">Data3</td>
</tr>
<tr>
<td data-cell-ref="A3">Data4</td>
<td data-cell-ref="B3">Data5</td>
<td data-cell-ref="C3">Data6</td>
</tr>
<tr>
<td data-cell-ref="A4">Data7</td>
<td data-cell-ref="B4">Data8</td>
<td data-cell-ref="C4">Data9</td>
</tr>
</table>"#;

        // Test table with header range
        let mut table_with_header = Element::new(
            "A1:C4".to_string(),
            Some("A1:C1".to_string()),
            SegmentType::Table,
        )
        .unwrap();
        let result = table_with_header.create_html(sheet_html, None, None);
        assert!(result.is_ok());

        // Verify the HTML file was created
        assert!(table_with_header.html.is_some());

        // Read and verify the generated HTML content
        let html_content =
            std::fs::read_to_string(table_with_header.html.as_ref().unwrap().path()).unwrap();

        // Should contain table structure
        assert!(html_content.contains("<table>"));
        assert!(html_content.contains("</table>"));

        // Should contain thead with header content
        assert!(html_content.contains("<thead>"));
        assert!(html_content.contains("Header1"));
        assert!(html_content.contains("Header2"));
        assert!(html_content.contains("Header3"));

        // Should contain tbody with data content (excluding header row)
        assert!(html_content.contains("<tbody>"));
        assert!(html_content.contains("Data1"));
        assert!(html_content.contains("Data2"));
        assert!(html_content.contains("Data9"));

        // Test table without header range
        let mut table_no_header =
            Element::new("A2:C4".to_string(), None, SegmentType::Table).unwrap();
        let result_no_header = table_no_header.create_html(sheet_html, None, None);
        assert!(result_no_header.is_ok());

        // Verify the HTML file was created
        assert!(table_no_header.html.is_some());

        // Read and verify the generated HTML content
        let html_content_no_header =
            std::fs::read_to_string(table_no_header.html.as_ref().unwrap().path()).unwrap();

        // Should contain table structure
        assert!(html_content_no_header.contains("<table>"));
        assert!(html_content_no_header.contains("</table>"));

        // Should NOT contain thead since no header range
        assert!(!html_content_no_header.contains("<thead>"));

        // Should contain tbody with data content
        assert!(html_content_no_header.contains("<tbody>"));
        assert!(html_content_no_header.contains("Data1"));
        assert!(html_content_no_header.contains("Data9"));

        // Should NOT contain header content
        assert!(!html_content_no_header.contains("Header1"));
    }

    #[test]
    fn test_create_html_with_overlap_removal() {
        // Create mock sheet HTML content
        let sheet_html = r#"<table>
<tr>
<td data-cell-ref="A1">Header1</td>
<td data-cell-ref="B1">Header2</td>
</tr>
<tr>
<td data-cell-ref="A2">Data1</td>
<td data-cell-ref="B2">Data2</td>
</tr>
<tr>
<td data-cell-ref="A3">Data3</td>
<td data-cell-ref="B3">Data4</td>
</tr>
</table>"#;

        // Test table where header and table ranges overlap
        let mut table_with_overlap = Element::new(
            "A1:B3".to_string(),
            Some("A1:B1".to_string()),
            SegmentType::Table,
        )
        .unwrap();
        let result = table_with_overlap.create_html(sheet_html, None, None);
        assert!(result.is_ok());

        // Read and verify the generated HTML content
        let html_content =
            std::fs::read_to_string(table_with_overlap.html.as_ref().unwrap().path()).unwrap();

        // Should contain table structure
        assert!(html_content.contains("<table>"));
        assert!(html_content.contains("</table>"));

        // Should contain thead with header content
        assert!(html_content.contains("<thead>"));
        assert!(html_content.contains("Header1"));
        assert!(html_content.contains("Header2"));

        // Should contain tbody with data content (without the header row)
        assert!(html_content.contains("<tbody>"));
        assert!(html_content.contains("Data1"));
        assert!(html_content.contains("Data4"));

        // The header row should not appear twice
        let header_count = html_content.matches("Header1").count();
        assert_eq!(header_count, 1);
    }

    #[test]
    fn test_create_html_invalid_range() {
        let sheet_html = r#"<table>
<tr>
<td data-cell-ref="A1">Data1</td>
</tr>
</table>"#;

        // Test with invalid range format that will cause parse error
        let mut table_invalid =
            Element::new("A1:Z99999".to_string(), None, SegmentType::Table).unwrap(); // Valid format but out of bounds
        let result = table_invalid.create_html(sheet_html, None, None);
        // This should fail because the range A1:Z99999 is out of bounds for our 1-row table
        assert!(result.is_err());

        // Test another invalid scenario - empty range
        let mut table_empty = Element::new(
            "A1:A1".to_string(),
            Some("A2:A2".to_string()),
            SegmentType::Table,
        )
        .unwrap(); // Header after table
        let result_empty = table_empty.create_html(sheet_html, None, None);
        // This should also fail because header range A2:A2 is out of bounds for our 1-row table
        assert!(result_empty.is_err());
    }

    #[test]
    fn test_calculate_header_alignment() {
        let table = Element::new("A1:D5".to_string(), None, SegmentType::Table).unwrap();

        // Test perfect alignment - no padding needed
        let result = table.calculate_header_alignment("A1:D1", "A2:D5").unwrap();
        assert_eq!(result, Some((0, 0)));

        // Test header needs left padding (C3:K3 vs A4:J20)
        // Header: C3:K3 (cols 2-10, 9 columns), Table: A4:J20 (cols 0-9, 10 columns)
        // Left padding: header_start - table_start = 2 - 0 = 2
        // Right padding: table_end - header_end = 9 - 10 = -1 -> max(0, -1) = 0
        let result = table.calculate_header_alignment("C3:K3", "A4:J20").unwrap();
        assert_eq!(result, Some((2, 0))); // 2 left padding, 0 right padding

        // Test header needs right padding (B2:D2 vs A4:F20)
        // Header: B2:D2 (cols 1-3, 3 columns), Table: A4:F20 (cols 0-5, 6 columns)
        // Left padding: header_start - table_start = 1 - 0 = 1
        // Right padding: table_end - header_end = 5 - 3 = 2
        let result = table.calculate_header_alignment("B2:D2", "A4:F20").unwrap();
        assert_eq!(result, Some((1, 2))); // 1 left padding, 2 right padding

        // Test header larger than table - should return None
        let result = table.calculate_header_alignment("A1:Z1", "B2:E5").unwrap();
        assert_eq!(result, None);

        // Test header completely outside table range - should return None
        let result = table.calculate_header_alignment("M3:X3", "A4:J20").unwrap();
        assert_eq!(result, None);

        // Test reasonable offset - header starts slightly before table
        // Header: A1:C1 (cols 0-2, 3 columns), Table: B2:F5 (cols 1-5, 5 columns)
        // Left padding: header_start - table_start = 0 - 1 = -1 -> max(0, -1) = 0
        // Right padding: table_end - header_end = 5 - 2 = 3
        let result = table.calculate_header_alignment("A1:C1", "B2:F5").unwrap();
        assert_eq!(result, Some((0, 3))); // 0 left padding, 3 right padding

        // Test edge case - excessive padding needed
        let result = table.calculate_header_alignment("A1:C1", "L2:N5").unwrap();
        assert_eq!(result, None); // Should be None due to excessive padding (11 total)
    }

    #[test]
    fn test_add_padding_to_header() {
        let table = Element::new("A1:D5".to_string(), None, SegmentType::Table).unwrap();

        // Test basic padding addition
        let header_html = r#"<thead>
<tr>
<td>Header1</td>
<td>Header2</td>
</tr>
</thead>"#;

        let result = table.add_padding_to_header(header_html, 2, 1).unwrap();

        // Should contain original headers plus padding cells
        assert!(result.contains("<thead>"));
        assert!(result.contains("</thead>"));
        assert!(result.contains("Header1"));
        assert!(result.contains("Header2"));
        assert!(result.contains("<td></td>")); // Should have empty padding cells

        // Count the number of td elements (should be 5: 2 padding + 2 original + 1 padding)
        let td_count = result.matches("<td").count();
        assert_eq!(td_count, 5);

        // Test no padding needed
        let result_no_padding = table.add_padding_to_header(header_html, 0, 0).unwrap();
        assert_eq!(result_no_padding, header_html);

        // Test with different wrapper (tbody)
        let tbody_html = r#"<tbody>
<tr>
<td>Data1</td>
<td>Data2</td>
</tr>
</tbody>"#;

        let result_tbody = table.add_padding_to_header(tbody_html, 1, 1).unwrap();
        assert!(result_tbody.contains("<tbody>"));
        assert!(result_tbody.contains("</tbody>"));

        let td_count_tbody = result_tbody.matches("<td").count();
        assert_eq!(td_count_tbody, 4); // 1 padding + 2 original + 1 padding

        // Test with single-line HTML (the format that was causing the original issue)
        let single_line_html = r#"<thead><tr> <td data-cell-ref="C42">Contract (X)</td> <td data-cell-ref="D42">TF/Tower</td> <td data-cell-ref="E42">Actual Headcount</td> </tr> </thead>"#;

        let result_single_line = table.add_padding_to_header(single_line_html, 2, 0).unwrap();

        // Should contain original headers plus padding cells
        assert!(result_single_line.contains("<thead>"));
        assert!(result_single_line.contains("</thead>"));
        assert!(result_single_line.contains("Contract (X)"));
        assert!(result_single_line.contains("TF/Tower"));
        assert!(result_single_line.contains("Actual Headcount"));
        assert!(result_single_line.contains("<td></td>")); // Should have empty padding cells

        // Count the number of td elements (should be 5: 2 padding + 3 original)
        let td_count_single_line = result_single_line.matches("<td").count();
        assert_eq!(td_count_single_line, 5);

        // Verify the padding cells are at the beginning (right after <tr>)
        assert!(result_single_line
            .contains("<tr><td></td><td></td> <td data-cell-ref=\"C42\">Contract (X)</td>"));
    }

    #[test]
    fn test_ranges_overlap() {
        // Test exact overlap
        assert!(Sheet::ranges_overlap("A1:D5", "A1:D5").unwrap());

        // Test partial overlap
        assert!(Sheet::ranges_overlap("A1:D5", "C3:F7").unwrap());
        assert!(Sheet::ranges_overlap("C3:F7", "A1:D5").unwrap());

        // Test no overlap - different rows
        assert!(!Sheet::ranges_overlap("A1:D3", "A5:D7").unwrap());

        // Test no overlap - different columns
        assert!(!Sheet::ranges_overlap("A1:C5", "E1:G5").unwrap());

        // Test adjacent ranges (should not overlap)
        assert!(!Sheet::ranges_overlap("A1:D3", "A4:D6").unwrap());
        assert!(!Sheet::ranges_overlap("A1:C5", "D1:F5").unwrap());

        // Test contained ranges
        assert!(Sheet::ranges_overlap("A1:F6", "B2:E5").unwrap());
        assert!(Sheet::ranges_overlap("B2:E5", "A1:F6").unwrap());

        // Test single cell overlap
        assert!(Sheet::ranges_overlap("A1:A1", "A1:A1").unwrap());
        assert!(Sheet::ranges_overlap("A1:D5", "D5:D5").unwrap());

        // Test single cell no overlap
        assert!(!Sheet::ranges_overlap("A1:A1", "B1:B1").unwrap());
    }

    #[test]
    fn test_create_html_with_header_alignment() {
        // Create mock sheet HTML content with wider range
        let sheet_html = r#"<table>
<tr>
<td data-cell-ref="A1">A1</td>
<td data-cell-ref="B1">B1</td>
<td data-cell-ref="C1">Header1</td>
<td data-cell-ref="D1">Header2</td>
<td data-cell-ref="E1">Header3</td>
<td data-cell-ref="F1">F1</td>
</tr>
<tr>
<td data-cell-ref="A2">Data1</td>
<td data-cell-ref="B2">Data2</td>
<td data-cell-ref="C2">Data3</td>
<td data-cell-ref="D2">Data4</td>
<td data-cell-ref="E2">Data5</td>
<td data-cell-ref="F2">Data6</td>
</tr>
<tr>
<td data-cell-ref="A3">Data7</td>
<td data-cell-ref="B3">Data8</td>
<td data-cell-ref="C3">Data9</td>
<td data-cell-ref="D3">Data10</td>
<td data-cell-ref="E3">Data11</td>
<td data-cell-ref="F3">Data12</td>
</tr>
</table>"#;

        // Test table with header alignment (header: C1:E1, table: A2:F3)
        let mut table_with_alignment = Element::new(
            "A2:F3".to_string(),
            Some("C1:E1".to_string()),
            SegmentType::Table,
        )
        .unwrap();
        let result = table_with_alignment.create_html(sheet_html, None, None);
        assert!(result.is_ok());

        // Read and verify the generated HTML content
        let html_content =
            std::fs::read_to_string(table_with_alignment.html.as_ref().unwrap().path()).unwrap();

        // Should contain table structure
        assert!(html_content.contains("<table>"));
        assert!(html_content.contains("</table>"));

        // Should contain thead and tbody
        assert!(html_content.contains("<thead>"));
        assert!(html_content.contains("<tbody>"));

        // Should contain original header content
        assert!(html_content.contains("Header1"));
        assert!(html_content.contains("Header2"));
        assert!(html_content.contains("Header3"));

        // Should contain table data
        assert!(html_content.contains("Data1"));
        assert!(html_content.contains("Data12"));

        // The header should have padding cells to align with table body
        // Header (C1:E1) needs 2 left padding cells and 1 right padding cell to align with table (A2:F3)
        let thead_section = html_content
            .split("<thead>")
            .nth(1)
            .unwrap()
            .split("</thead>")
            .next()
            .unwrap();
        let empty_cell_count = thead_section.matches("<td></td>").count();
        assert_eq!(empty_cell_count, 3); // 2 left + 1 right padding
    }
}
