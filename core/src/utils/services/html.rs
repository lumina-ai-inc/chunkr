use nipper::Document;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use tempfile::NamedTempFile;

use crate::models::output::{Cell, CellStyle};
use crate::models::pipeline::Indices;

static IMG_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"<img(?:[^>]*?alt=["']([^"']*?)["'])?[^>]*>"#).unwrap());

/// Generate Excel-style cell reference (e.g., A1, B1, AA1)
///
/// ### Arguments
/// * `row` - Zero-based row index
/// * `col` - Zero-based column index
/// * `colspan` - Number of columns the cell spans (default 1)
/// * `rowspan` - Number of rows the cell spans (default 1)
///
/// ### Returns
/// String representation of the cell reference
///
/// ### Examples
///
/// ```
/// let cell_ref = get_cell_reference(0, 0, 1, 1);
/// assert_eq!(cell_ref, "A1");
/// ```
fn get_cell_reference(row: usize, col: usize, colspan: usize, rowspan: usize) -> String {
    let col_name = column_index_to_name(col);
    let row_name = row + 1; // Excel rows are 1-indexed

    if colspan > 1 || rowspan > 1 {
        let end_col = col + colspan - 1;
        let end_row = row + rowspan - 1;
        let end_col_name = column_index_to_name(end_col);
        let end_row_name = end_row + 1;
        format!("{col_name}{row_name}:{end_col_name}{end_row_name}")
    } else {
        format!("{col_name}{row_name}")
    }
}

/// Convert column index to Excel column name (0->A, 1->B, 25->Z, 26->AA, etc.)
///
/// ### Arguments
/// * `col` - Zero-based column index
///
/// ### Returns
/// String representation of the column name
///
/// ### Examples
///
/// ```
/// let col_name = column_index_to_name(0);
/// assert_eq!(col_name, "A");
/// ```
pub fn column_index_to_name(mut col: usize) -> String {
    let mut result = String::new();

    loop {
        let remainder = col % 26;
        result = format!("{}{}", (b'A' + remainder as u8) as char, result);
        if col < 26 {
            break;
        }
        col = (col / 26) - 1;
    }

    result
}

/// Convert Indices struct to Excel range notation (e.g., "A1:D10" or "A1" for single cell)
///
/// ### Arguments
/// * `indices` - The Indices struct containing start and end row/column positions
///
/// ### Returns
/// String representation of the Excel range
///
/// ### Examples
///
/// ```
/// let indices = Indices { start_row: 0, start_col: 0, end_row: 0, end_col: 0 };
/// let range = indices_to_range(&indices);
/// assert_eq!(range, "A1");
///
/// let indices = Indices { start_row: 0, start_col: 0, end_row: 2, end_col: 3 };
/// let range = indices_to_range(&indices);
/// assert_eq!(range, "A1:D3");
/// ```
pub fn indices_to_range(indices: &Indices) -> String {
    let start_cell = format!(
        "{}{}",
        column_index_to_name(indices.start_col),
        indices.start_row + 1
    );
    let end_cell = format!(
        "{}{}",
        column_index_to_name(indices.end_col),
        indices.end_row + 1
    );

    if start_cell == end_cell {
        start_cell
    } else {
        format!("{start_cell}:{end_cell}")
    }
}

/// Parse an Excel range string to extract row and column indices
///
/// ### Arguments
/// * `range` - Excel range string (e.g., "A1:D8" or "A1")
///
/// ### Returns
/// Tuple of (start_row, start_col, end_row, end_col) as zero-based indices
///
/// ### Examples
///
/// ```
/// let range = "A1:D8";
/// let (start_row, start_col, end_row, end_col) = parse_range(range).unwrap();
/// assert_eq!(start_row, 0);
/// assert_eq!(start_col, 0);
pub fn parse_range(range: &str) -> Result<Indices, Box<dyn Error + Send + Sync>> {
    fn parse_cell(cell: &str) -> Result<(usize, usize), Box<dyn Error + Send + Sync>> {
        let mut col_end = 0;
        for (i, c) in cell.chars().enumerate() {
            if c.is_ascii_digit() {
                col_end = i;
                break;
            }
        }

        let col_str = &cell[..col_end];
        let row_str = &cell[col_end..];

        // Validate that we have column letters
        if col_str.is_empty() {
            return Err(format!("Invalid cell format: no column letters found {cell}").into());
        }

        // Validate that all characters in col_str are valid letters (A-Z)
        if !col_str.chars().all(|c: char| c.is_ascii_uppercase()) {
            return Err(format!(
                "Invalid cell format: column must contain only uppercase letters A-Z {col_str}"
            )
            .into());
        }

        // Validate that we have row digits
        if row_str.is_empty() {
            return Err(format!("Invalid cell format: no row number found {cell}").into());
        }

        // Convert column letters to zero-based index
        let mut col_index = 0;
        for c in col_str.chars() {
            col_index = col_index * 26 + (c as usize - 'A' as usize + 1);
        }

        // Convert to zero-based - we know col_index > 0 because we validated col_str is not empty
        col_index -= 1;

        // Convert row to zero-based index
        let row_index = row_str.parse::<usize>()? - 1;

        Ok((row_index, col_index))
    }

    if range.contains(':') {
        let parts: Vec<&str> = range.split(':').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid range format {range}").into());
        }
        let (start_row, start_col) = parse_cell(parts[0])?;
        let (end_row, end_col) = parse_cell(parts[1])?;
        Ok(Indices {
            start_row,
            start_col,
            end_row,
            end_col,
        })
    } else {
        let (row, col) = parse_cell(range)?;
        Ok(Indices {
            start_row: row,
            start_col: col,
            end_row: row,
            end_col: col,
        })
    }
}

/// Helper function to check if a table row contains only empty/styled cells
/// Returns true if the row only contains cells with br tags, whitespace, or empty content
fn is_row_empty(row: &nipper::Selection) -> bool {
    let cells = row.select("td, th");

    // Consider row empty if it has no cells
    if cells.length() == 0 {
        return true;
    }

    // Check if all cells in the row are effectively empty
    cells.iter().all(|cell| {
        let text_content = cell.text();
        let trimmed = text_content.trim();

        // Consider the cell empty if:
        // 1. It has no text content, OR
        // 2. It only contains whitespace or line breaks
        trimmed.is_empty() || trimmed.chars().all(|c| c.is_whitespace())
    })
}

/// Remove the first empty row(s) from an HTML table
///
/// This function removes empty rows from the beginning of a table to handle
/// LibreOffice's inconsistent HTML export behavior where styled empty rows
/// are sometimes included at the top.
///
/// ### Arguments
/// * `html_content` - The HTML content containing a table
///
/// ### Returns
/// Modified HTML content with leading empty rows removed
pub fn remove_leading_empty_rows(
    html_content: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let document = Document::from(html_content);

    // Find the first table
    let table = document.select("table").first();
    if !table.exists() {
        return Ok(html_content.to_string());
    }

    let rows = table.select("tr");
    let mut rows_to_keep = Vec::new();
    let mut found_content = false;

    // Collect rows, skipping empty ones at the beginning
    for row in rows.iter() {
        if !found_content && is_row_empty(&row) {
            continue; // Skip this empty row at the beginning
        }

        // Once we find the first non-empty row, keep all subsequent rows
        found_content = true;
        rows_to_keep.push(row.html().to_string());
    }

    // If no rows to keep, return original content
    if rows_to_keep.is_empty() {
        return Ok(html_content.to_string());
    }

    // Get the table's opening tag with all its attributes
    let table_html = table.html().to_string();
    let table_start_tag = if let Some(end) = table_html.find('>') {
        &table_html[..=end]
    } else {
        "<table>"
    };

    // Reconstruct the table with filtered rows
    let new_table_html = format!("{}{}</table>", table_start_tag, rows_to_keep.join(""));

    // Replace the original table with the new one
    let mut table_mut = table;
    table_mut.replace_with_html(new_table_html.as_str());

    Ok(document.html().to_string())
}

/// Add Excel-style cell references as data attributes to HTML table cells
///
/// ### Arguments
/// * `html_content` - The HTML content containing a table
/// * `start_row` - The starting row offset (0-based)  
/// * `start_col` - The starting column offset (0-based)
///
/// ### Returns
/// Modified HTML content with data-cell-ref attributes added to table cells
pub fn add_cell_references_to_html(
    html_content: &str,
    start_row: usize,
    start_col: usize,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let document = Document::from(html_content);

    // Find the first table and process it
    let table = document.select("table").first();
    if table.exists() {
        let mut current_row = start_row;

        // Process each row
        table.select("tr").iter().for_each(|row| {
            let mut current_col = start_col;

            // Process each cell in the row
            row.select("td, th").iter().for_each(|mut cell| {
                // Get colspan and rowspan attributes
                let colspan = cell
                    .attr("colspan")
                    .and_then(|val| val.parse::<usize>().ok())
                    .unwrap_or(1);
                let rowspan = cell
                    .attr("rowspan")
                    .and_then(|val| val.parse::<usize>().ok())
                    .unwrap_or(1);

                // Generate cell reference
                let cell_ref = get_cell_reference(current_row, current_col, colspan, rowspan);

                // Add the data-cell-ref attribute
                cell.set_attr("data-cell-ref", &cell_ref);

                current_col += colspan;
            });

            current_row += 1;
        });
    }

    Ok(document.html().to_string())
}

/// Cleans the image tags from the HTML
///
/// Replaces HTML image tags with their alt text (if available) or removes them entirely.
/// Useful for converting HTML to plain text while preserving image descriptions.
///
/// ### Examples
///
/// ```
/// let html = r#"<p>Text <img src="pic.jpg" alt="A picture"> more text</p>"#;
/// let cleaned = clean_img_tags(html);
/// assert_eq!(cleaned, "<p>Text A picture more text</p>");
/// ```
pub fn clean_img_tags(html: &str) -> String {
    IMG_REGEX
        .replace_all(html, |caps: &regex::Captures| {
            caps.get(1)
                .map_or("".to_string(), |m| m.as_str().to_string())
        })
        .to_string()
}

pub fn convert_html_to_markdown(html: String) -> Result<String, Box<dyn std::error::Error>> {
    // Use pandoc to convert HTML to markdown
    let mut child = Command::new("pandoc")
        .arg("-f")
        .arg("html")
        .arg("-t")
        .arg("markdown")
        .arg("--wrap=none")
        .arg("--to")
        .arg("markdown_strict+pipe_tables")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start pandoc: {e}. Make sure pandoc is installed."))?;

    // Write HTML to pandoc's stdin
    if let Some(stdin) = child.stdin.take() {
        let mut stdin = stdin;
        stdin
            .write_all(html.as_bytes())
            .map_err(|e| format!("Failed to write to pandoc stdin: {e}"))?;
    }

    // Wait for pandoc to finish and get output
    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait for pandoc: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Pandoc failed: {stderr}").into());
    }

    let markdown = String::from_utf8(output.stdout)
        .map_err(|e| format!("Failed to parse pandoc output as UTF-8: {e}"))?;

    Ok(markdown.trim().to_string())
}

/// Extract individual table HTML content from HTML generated by LibreOffice from Excel files
///
/// This function parses the HTML and returns the table HTML content in order.
pub fn extract_sheets_from_html(html_file: &NamedTempFile) -> Result<Vec<String>, Box<dyn Error>> {
    // Read the HTML content
    let html_content = fs::read_to_string(html_file.path())?;
    let document = Document::from(&html_content);

    // Find all table elements in order
    let tables = document.select("table");
    let mut sheet_htmls = Vec::new();

    for table in tables.iter() {
        // Create a complete HTML document for this sheet
        let sheet_html = create_sheet_html(&table, &document);
        sheet_htmls.push(sheet_html);
    }

    Ok(sheet_htmls)
}

/// Create a complete HTML document for a single sheet
fn create_sheet_html(table: &nipper::Selection, original_document: &Document) -> String {
    // Extract any styles from the original document
    let styles_selection = original_document.select("style");
    let mut styles = String::new();

    for style_element in styles_selection.iter() {
        styles.push_str(style_element.html().as_ref());
        styles.push('\n');
    }

    // Add CSS to hide scrollbars and ensure proper sizing
    let additional_css = r#"
        <style>
            /* Hide scrollbars */
            html, body {
                overflow: hidden !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            
            /* Ensure table takes full width without scrollbars */
            table {
                table-layout: auto !important;
                width: 100% !important;
                overflow: visible !important;
            }
        </style>
    "#;

    // Create the initial sheet HTML
    let initial_sheet_html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Excel Sheet</title>
    {}
    {}
</head>
<body>
    {}
</body>
</html>"#,
        styles,
        additional_css,
        table.html()
    );

    // Remove leading empty rows from the sheet
    match remove_leading_empty_rows(&initial_sheet_html) {
        Ok(cleaned_html) => cleaned_html,
        Err(_) => initial_sheet_html, // Fallback to original if preprocessing fails
    }
}

pub fn get_img_sources(html: &str) -> Vec<String> {
    let document = Document::from(html);
    document
        .select("img")
        .iter()
        .map(|img| img.attr("src").unwrap().to_string())
        .collect()
}

/// Get the data-cell-ref attribute for a table cell containing an image with the given src
///
/// This function searches through the HTML content to find an img tag with the specified src,
/// then traverses up the DOM tree to find the containing td or th element and returns its
/// data-cell-ref attribute.
///
/// ### Arguments
/// * `html_content` - The HTML content to search through
/// * `img_src` - The src attribute value of the image to find
///
/// ### Returns
/// Result<String, Box<dyn Error + Send + Sync>> containing the data-cell-ref attribute value, or an error if not found
///
/// ### Examples
///
/// ```
/// let html = r#"<table><tr><td data-cell-ref="A1"><img src="image.jpg"></td></tr></table>"#;
/// let cell_ref = get_cell_ref_for_image_src(html, "image.jpg").unwrap();
/// assert_eq!(cell_ref, "A1".to_string());
/// ```
pub fn get_cell_ref_for_image_src(
    html_content: &str,
    img_src: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let document = Document::from(html_content);

    // Find all img tags
    for img in document.select("img").iter() {
        // Check if this img has the matching src attribute
        if let Some(src) = img.attr("src") {
            if src.as_ref() == img_src {
                // Found the matching image, now traverse up to find the containing td or th
                let mut current = img;

                // Keep going up the DOM tree until we find a td or th element
                loop {
                    // Check if current element is a td or th
                    if current.is("td") || current.is("th") {
                        // Found the cell, check if it has data-cell-ref attribute
                        if let Some(cell_ref) = current.attr("data-cell-ref") {
                            return Ok(cell_ref.to_string());
                        } else {
                            return Err(format!("Found image '{img_src}' in table cell but cell has no data-cell-ref attribute").into());
                        }
                    }

                    // Try to get the parent element
                    let parent = current.parent();
                    if parent.exists() {
                        current = parent;
                    } else {
                        // No more parents, we've reached the root without finding a cell
                        return Err(format!(
                            "Found image '{img_src}' but it's not contained within a table cell"
                        )
                        .into());
                    }
                }
            }
        }
    }

    // Image src not found
    Err(format!("Image with src '{img_src}' not found in HTML content").into())
}

/// Add Excel-style row and column headers to an HTML table
///
/// This function takes HTML content containing a table and adds:
/// - Column headers starting from the specified column (A, B, C, etc.)
/// - Row numbers starting from the specified row (1, 2, 3, etc.) as the first column of each row
/// - Excel-like styling only to the added header elements (preserves existing table styling)
///
/// ### Arguments
/// * `html_content` - The HTML content containing a table
/// * `start_col` - The starting column index (0 = A, 1 = B, etc.)
/// * `start_row` - The starting row index (0 = row 1, 1 = row 2, etc.)
///
/// ### Returns
/// Modified HTML content with Excel-style headers added
pub fn add_excel_headers_to_html(
    html_content: &str,
    start_col: usize,
    start_row: usize,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let document = Document::from(html_content);

    // Find the first table
    let mut table = document.select("table").first();
    if !table.exists() {
        return Err("No table found".into());
    }

    // Get all existing rows to determine max columns
    let rows = table.select("tr");
    let mut max_cols = 0;

    for row in rows.iter() {
        let cols = row.select("td, th").length();
        if cols > max_cols {
            max_cols = cols;
        }
    }

    if max_cols == 0 {
        return Ok(html_content.to_string());
    }

    // Create column header row HTML
    let mut header_html = String::from("<tr>");

    // Add empty cell for row numbers column
    header_html.push_str(r#"<th style="background:#E0E0E0; font-weight:bold; text-align:center; border:1px solid #888; font-size:10pt; width:30px; font-family:Arial,sans-serif;"></th>"#);

    // Add column headers starting from start_col (A, B, C, etc.)
    for i in 0..max_cols {
        let col_name = column_index_to_name(start_col + i);
        header_html.push_str(&format!(
            r#"<th style="background:#E0E0E0; font-weight:bold; text-align:center; border:1px solid #888; font-size:10pt; min-width:50px; font-family:Arial,sans-serif;">{col_name}</th>"#
        ));
    }
    header_html.push_str("</tr>");

    // Collect all existing rows and add row numbers
    let mut modified_rows = Vec::new();

    // First add the header row
    modified_rows.push(header_html);

    // Process existing rows and add row numbers starting from start_row
    for (row_counter, row) in rows.iter().enumerate() {
        let row_num_cell = format!(
            r#"<td style="background:#E0E0E0; font-weight:bold; text-align:center; border:1px solid #888; font-size:10pt; width:30px; font-family:Arial,sans-serif;">{}</td>"#,
            start_row + row_counter + 1 // +1 because Excel rows are 1-indexed
        );

        // Get the row's HTML and extract the content inside <tr>...</tr>
        let row_html = row.html().to_string();

        // Find the opening and closing tr tags
        if let Some(start) = row_html.find('>') {
            if let Some(end) = row_html.rfind("</tr>") {
                let row_content = &row_html[start + 1..end];
                let modified_row = format!("<tr>{row_num_cell}{row_content}</tr>");
                modified_rows.push(modified_row);
            }
        }
    }

    // Get the table opening tag with all its attributes
    let table_html = table.html().to_string();
    let table_start_tag = if let Some(end) = table_html.find('>') {
        &table_html[..=end]
    } else {
        "<table>"
    };

    // Reconstruct the entire table with Excel headers
    let new_table_html = format!("{}{}</table>", table_start_tag, modified_rows.join(""));

    // Replace the original table with the new one
    table.replace_with_html(new_table_html.as_str());

    Ok(document.html().to_string())
}

/// Clean HTML by removing unnecessary attributes to reduce token count for LLM processing
///
/// Keeps only essential attributes needed for table structure analysis:
/// - data-cell-ref: Important for cell identification
/// - bgcolor: Useful for visual table structure
/// - style: May contain important border/structure info
/// - colspan, rowspan: Important for table structure
///
/// Removes unnecessary attributes and cleans up font tags and empty br tags.
///
/// ### Arguments
/// * `html_content` - The HTML content to clean
///
/// ### Returns
/// Cleaned HTML content with reduced token count
pub fn clean_html_for_llm(html_content: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    let document = Document::from(html_content);

    // Attributes to keep (whitelist approach)
    let keep_attributes = ["data-cell-ref", "bgcolor", "style", "colspan", "rowspan"];

    // Clean all elements - remove unnecessary attributes
    document.select("*").iter().for_each(|mut element| {
        // Common attributes that might need to be removed
        let common_attributes = [
            "data-sheets-value",
            "face",
            "color",
            "valign",
            "height",
            "align",
            "width",
            "size",
            "border",
            "cellpadding",
            "cellspacing",
            "class",
            "id",
        ];

        // Remove attributes that are not in the whitelist
        for attr_name in common_attributes {
            if !keep_attributes.contains(&attr_name) {
                element.remove_attr(attr_name);
            }
        }
    });

    // Remove font tags but keep their content (unwrap)
    let fonts: Vec<_> = document.select("font").iter().collect();
    for mut font_tag in fonts {
        let content = font_tag.html().to_string();
        // Extract just the inner HTML (content without the font tags)
        if let (Some(start), Some(end)) = (content.find('>'), content.rfind("</font>")) {
            let inner_content = &content[start + 1..end];
            font_tag.replace_with_html(inner_content);
        }
    }

    // Remove empty br tags
    document.select("br").iter().for_each(|mut br_tag| {
        // Check if br tag has any text content (it shouldn't, but just to be safe)
        if br_tag.text().trim().is_empty() {
            br_tag.remove();
        }
    });

    Ok(document.html().to_string())
}

/// Parse HTML tag and return proper opening and closing tags
///
/// ### Arguments
/// * `wrapper_tag` - Optional HTML tag string (e.g., "<table>" or "<div class='wrapper'>")
///
/// ### Returns
/// Tuple of (opening_tag, closing_tag)
fn parse_html_tag(
    wrapper_tag: Option<&str>,
) -> Result<(String, String), Box<dyn Error + Send + Sync>> {
    match wrapper_tag {
        Some(tag) => {
            // Validate that it's a proper HTML tag
            if !tag.starts_with('<') || !tag.ends_with('>') {
                return Err("Invalid HTML tag format".into());
            }

            // Extract tag name from opening tag
            let tag_content = &tag[1..tag.len() - 1]; // Remove < and >
            let tag_name = tag_content
                .split_whitespace()
                .next()
                .ok_or("Empty tag name")?;

            // Validate tag name (basic HTML tag name validation)
            if tag_name.is_empty()
                || !tag_name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-')
            {
                return Err("Invalid tag name".into());
            }

            let opening_tag = tag.to_string();
            let closing_tag = format!("</{tag_name}>");

            Ok((opening_tag, closing_tag))
        }
        None => {
            // No wrapper
            Ok((String::new(), String::new()))
        }
    }
}

pub fn extract_table_from_html(html: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    let document = Document::from(html);
    let table = document.select("table").first();
    if !table.exists() {
        return Err("No table found".into());
    }
    Ok(table.html().to_string())
}

pub fn extract_rows_from_indicies(
    html: &str,
    indices: &Indices,
    wrapper_tag: Option<&str>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let document = Document::from(html);
    let table = document.select("table").first();
    if !table.exists() {
        return Err("No table found".into());
    }

    // Get all rows from the table
    let rows = table.select("tr");
    let total_rows = rows.length();

    // Validate indices
    if indices.start_row >= total_rows || indices.end_row >= total_rows {
        return Err(format!(
            "Row indices are out of bounds | start_row: {}, end_row: {}, total_rows: {}",
            indices.start_row, indices.end_row, total_rows
        )
        .into());
    }

    if indices.start_row > indices.end_row {
        return Err("Start row must be less than or equal to end row".into());
    }

    // Helper function to check if a cell reference falls within the column range
    fn is_cell_in_column_range(
        cell_ref: &str,
        start_col: usize,
        end_col: usize,
    ) -> Result<bool, Box<dyn Error + Send + Sync>> {
        // Parse the cell reference to get column index
        let mut col_end = 0;
        for (i, c) in cell_ref.chars().enumerate() {
            if c.is_ascii_digit() {
                col_end = i;
                break;
            }
        }

        if col_end == 0 {
            return Err("Invalid cell reference format".into());
        }

        let col_str = &cell_ref[..col_end];

        // Convert column letters to zero-based index
        let mut col_index = 0;
        for c in col_str.chars() {
            col_index = col_index * 26 + (c as usize - 'A' as usize + 1);
        }
        col_index -= 1; // Convert to zero-based

        Ok(col_index >= start_col && col_index <= end_col)
    }

    // Extract rows from start_row to end_row (inclusive) and get their HTML
    let mut extracted_rows = Vec::new();
    for i in indices.start_row..=indices.end_row {
        let row = rows.get(i);
        if row.is_none() {
            return Err("Row not found".into());
        }
        extracted_rows.push(row.unwrap().html().to_string());
    }

    let all_rows_cells: Result<Vec<Vec<String>>, Box<dyn Error + Send + Sync>> = extracted_rows
        .par_iter()
        .map(
            |row_html| -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
                // Parse this row HTML to extract cells
                // Wrap in table tags so HTML parser can properly find td elements
                let wrapped_row_html = format!("<table>{row_html}</table>");
                let row_document = Document::from(wrapped_row_html.as_str());
                let cells = row_document.select("td, th");

                let cell_htmls = cells
                    .iter()
                    .map(|cell| cell.html().to_string())
                    .collect::<Vec<String>>();

                // Check if any cells have data-cell-ref attributes to determine which mode to use
                let has_cell_refs = cell_htmls.iter().any(|cell_html| {
                    let wrapped_cell_html = format!("<table><tr>{cell_html}</tr></table>");
                    let cell = Document::from(wrapped_cell_html.as_str());
                    let only_cell = cell.select("td, th").first();
                    only_cell.attr("data-cell-ref").is_some()
                });

                let filtered_cells: Result<Vec<String>, Box<dyn Error + Send + Sync>> =
                    if has_cell_refs {
                        // Filter cells by data-cell-ref attribute
                        cell_htmls
                            .par_iter()
                            .filter_map(|cell_html| {
                                let wrapped_cell_html =
                                    format!("<table><tr>{cell_html}</tr></table>");
                                let cell = Document::from(wrapped_cell_html.as_str());
                                let only_cell = cell.select("td, th").first();
                                if let Some(cell_ref) = only_cell.attr("data-cell-ref") {
                                    // Check if this cell falls within our desired column range
                                    match is_cell_in_column_range(
                                        &cell_ref,
                                        indices.start_col,
                                        indices.end_col,
                                    ) {
                                        Ok(true) => {
                                            // Extract clean text content without font/br tags
                                            let text_content = only_cell.text();
                                            let clean_content = text_content.trim();

                                            let mut attributes = Vec::new();

                                            // Preserve structural attributes
                                            if let Some(colspan) = only_cell.attr("colspan") {
                                                attributes.push(format!("colspan=\"{colspan}\""));
                                            }
                                            if let Some(rowspan) = only_cell.attr("rowspan") {
                                                attributes.push(format!("rowspan=\"{rowspan}\""));
                                            }
                                            attributes
                                                .push(format!("data-cell-ref=\"{cell_ref}\""));

                                            // Preserve styling attributes
                                            if let Some(formula) =
                                                only_cell.attr("data-sheets-formula")
                                            {
                                                // HTML escape the formula to handle quotes and special characters
                                                let escaped_formula = formula
                                                    .replace('&', "&amp;")
                                                    .replace('"', "&quot;")
                                                    .replace('<', "&lt;")
                                                    .replace('>', "&gt;");
                                                attributes.push(format!(
                                                    "data-sheets-formula=\"{escaped_formula}\""
                                                ));
                                            }

                                            // Extract font information from font tags within the cell and convert to CSS
                                            let mut style_parts = Vec::new();

                                            // Convert bgcolor to CSS background-color
                                            let bgcolor = only_cell
                                                .attr("bgcolor")
                                                .unwrap_or_else(|| "#FFFFFF".into());
                                            let bg_color_value = if bgcolor.starts_with('#') {
                                                bgcolor.to_string()
                                            } else {
                                                format!("#{bgcolor}")
                                            };
                                            style_parts.push(format!(
                                                "background-color: {bg_color_value}"
                                            ));

                                            let font_elements = only_cell.select("font");
                                            if font_elements.length() > 0 {
                                                let font = font_elements.first();
                                                let color_value =
                                                    if let Some(color) = font.attr("color") {
                                                        // Handle both #RRGGBB and RRGGBB formats
                                                        if color.starts_with('#') {
                                                            color.to_string()
                                                        } else {
                                                            format!("#{color}")
                                                        }
                                                    } else {
                                                        // Default to black if no color specified
                                                        "#000000".to_string()
                                                    };
                                                style_parts.push(format!("color: {color_value}"));
                                            } else {
                                                style_parts.push("color: #000000".to_string());
                                            }

                                            // Add CSS style attribute if we have styles
                                            if !style_parts.is_empty() {
                                                attributes.push(format!(
                                                    "style=\"{}\"",
                                                    style_parts.join("; ")
                                                ));
                                            }

                                            // Create td element preserving all styling (convert th to td)
                                            let attrs_str = if attributes.is_empty() {
                                                String::new()
                                            } else {
                                                format!(" {}", attributes.join(" "))
                                            };

                                            let cell_html =
                                                format!("<td{attrs_str}>{clean_content}</td>");
                                            Some(Ok(cell_html))
                                        }
                                        Ok(false) => None,
                                        Err(e) => Some(Err(e)),
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect()
                    } else {
                        // Fallback to array-based indexing when no data-cell-ref attributes
                        cell_htmls
                            .into_iter()
                            .enumerate()
                            .filter_map(|(cell_index, cell_html)| {
                                // Check if this cell index falls within our desired column range
                                if cell_index >= indices.start_col && cell_index <= indices.end_col
                                {
                                    Some(Ok(cell_html))
                                } else {
                                    None
                                }
                            })
                            .collect()
                    };

                let row_cells = filtered_cells?;

                Ok(row_cells)
            },
        )
        .collect();

    let all_rows_cells = all_rows_cells?;

    // Parse and validate the wrapper tag
    let (opening_tag, closing_tag) = parse_html_tag(wrapper_tag)?;

    let mut result = String::from(&opening_tag);
    for row_cells in all_rows_cells {
        result.push_str("<tr>\n");
        for cell in row_cells {
            result.push_str(&cell);
            result.push('\n');
        }
        result.push_str("</tr>\n");
    }
    result.push_str(&closing_tag);

    Ok(result)
}

/// Get actual end row and column indices from HTML table structure
///
/// This function iterates through the HTML table to determine the actual maximum
/// row and column indices, accounting for the fact that LibreOffice ignores hidden
/// rows and columns while Calamine includes them.
///
/// ### Arguments
/// * `html_content` - The HTML content containing a table
///
/// ### Returns
/// Tuple of (end_row, end_col) as zero-based indices
/// Returns (0, 0) if no table is found or table is empty
///
/// ### Examples
///
/// ```
/// let html = r#"<table><tr><td>A1</td><td>B1</td></tr><tr><td>A2</td></tr></table>"#;
/// let (end_row, end_col) = get_actual_table_bounds(html).unwrap();
/// assert_eq!(end_row, 1); // 2 rows (0-1)
/// assert_eq!(end_col, 1); // 2 columns max (0-1)
/// ```
pub fn get_html_table_bounds(
    html_content: &str,
) -> Result<(usize, usize), Box<dyn Error + Send + Sync>> {
    let document = Document::from(html_content);
    let table = document.select("table").first();

    if !table.exists() {
        return Ok((0, 0));
    }

    let rows = table.select("tr");
    let row_count = rows.length();

    if row_count == 0 {
        return Ok((0, 0));
    }

    // Extract all row HTMLs as strings first for parallel processing
    let row_htmls: Vec<String> = rows.iter().map(|row| row.html().to_string()).collect();

    // Process rows in parallel to get column counts
    let col_counts: Result<Vec<usize>, Box<dyn Error + Send + Sync>> = row_htmls
        .par_iter()
        .map(|row_html| -> Result<usize, Box<dyn Error + Send + Sync>> {
            // Wrap in table tags so HTML parser can properly find td/th elements
            let wrapped_row_html = format!("<table>{row_html}</table>");
            let row_document = Document::from(wrapped_row_html.as_str());
            let cells = row_document.select("td, th");

            // Extract cell HTMLs and process in parallel
            let cell_htmls: Vec<String> =
                cells.iter().map(|cell| cell.html().to_string()).collect();

            let total_colspan: Result<usize, Box<dyn Error + Send + Sync>> = cell_htmls
                .par_iter()
                .map(|cell_html| -> Result<usize, Box<dyn Error + Send + Sync>> {
                    let wrapped_cell_html = format!("<table><tr>{cell_html}</tr></table>");
                    let cell_document = Document::from(wrapped_cell_html.as_str());
                    let cell = cell_document.select("td, th").first();

                    let colspan = cell
                        .attr("colspan")
                        .and_then(|val| val.parse::<usize>().ok())
                        .unwrap_or(1);

                    Ok(colspan)
                })
                .try_reduce(|| 0, |acc, colspan| Ok(acc + colspan));

            total_colspan
        })
        .collect();

    let col_counts = col_counts?;
    let max_col = col_counts.into_iter().max().unwrap_or(0);

    // Return zero-based indices
    let end_row = if row_count > 0 { row_count - 1 } else { 0 };
    let end_col = if max_col > 0 { max_col - 1 } else { 0 };

    Ok((end_row, end_col))
}

/// Extract cell information from HTML content based on given ranges
///
/// This function extracts cell data including text content, formulas, and formatting
/// from HTML table cells that fall within the specified ranges. It combines cells
/// from both table_range and header_range, removing duplicates.
///
/// ### Arguments
/// * `html_content` - The HTML content containing table cells with data-cell-ref attributes
/// * `table_range` - Optional range for the main table data (e.g., "A1:D10")
/// * `header_range` - Optional range for the table headers (e.g., "A1:D1")
///
/// ### Returns
/// Vector of Cell structs containing the extracted cell information
///
/// ### Examples
///
/// ```
/// let html = r#"<table><tr><td data-cell-ref="A1" data-sheets-formula="=SUM(B1:B10)">Total</td></tr></table>"#;
/// let cells = extract_cells_from_ranges(html, Some("A1:A1"), None).unwrap();
/// ```
pub fn extract_cells_from_ranges(
    html_content: &str,
    table_range: Option<&str>,
    header_range: Option<&str>,
) -> Result<Vec<Cell>, Box<dyn Error + Send + Sync>> {
    let document = Document::from(html_content);
    let mut cells = Vec::new();
    let mut processed_refs = std::collections::HashSet::new();

    // Function to check if a cell reference falls within a given range
    let is_cell_in_range =
        |cell_ref: &str, range: &str| -> Result<bool, Box<dyn Error + Send + Sync>> {
            let range_indices = parse_range(range)?;
            let cell_indices = parse_range(cell_ref)?;

            Ok(cell_indices.start_row >= range_indices.start_row
                && cell_indices.end_row <= range_indices.end_row
                && cell_indices.start_col >= range_indices.start_col
                && cell_indices.end_col <= range_indices.end_col)
        };

    // Process all table cells
    for cell in document.select("td, th").iter() {
        if let Some(cell_ref_attr) = cell.attr("data-cell-ref") {
            let cell_ref = cell_ref_attr.as_ref();
            let mut should_include = false;

            // Check if cell falls within table range
            if let Some(range) = table_range {
                match is_cell_in_range(cell_ref, range) {
                    Ok(true) => should_include = true,
                    Ok(false) => {}
                    Err(_) => continue, // Skip cells with invalid references
                }
            }

            // Check if cell falls within header range
            if !should_include {
                if let Some(range) = header_range {
                    match is_cell_in_range(cell_ref, range) {
                        Ok(true) => should_include = true,
                        Ok(false) => {}
                        Err(_) => continue, // Skip cells with invalid references
                    }
                }
            }

            // If no ranges specified, include all cells
            if table_range.is_none() && header_range.is_none() {
                should_include = true;
            }

            // Extract cell information if it should be included
            if should_include {
                // Skip if we've already processed this cell reference
                if processed_refs.contains(cell_ref) {
                    continue;
                }

                // Extract text content
                let text = cell.text().trim().to_string();

                // Extract formula if present
                let formula = cell.attr("data-sheets-formula").map(|f| f.to_string());

                // Extract value if present
                let value = cell.attr("sdval").map(|v| v.to_string());

                // Extract styling information
                // Handle bgcolor with format normalization and default to white
                let bg_color = Some(
                    cell.attr("bgcolor")
                        .map(|c| {
                            if c.starts_with('#') {
                                c.to_string()
                            } else {
                                format!("#{c}")
                            }
                        })
                        .unwrap_or_else(|| "#FFFFFF".to_string()),
                );
                let font_elements = cell.select("font");
                let mut font_face = None;

                // Extract font styling if font elements exist
                let text_color = if font_elements.length() > 0 {
                    let font = font_elements.first();
                    font_face = font.attr("face").map(|f| f.to_string());
                    // Handle color with format normalization and default to black
                    Some(
                        font.attr("color")
                            .map(|c| {
                                if c.starts_with('#') {
                                    c.to_string()
                                } else {
                                    format!("#{c}")
                                }
                            })
                            .unwrap_or_else(|| "#000000".to_string()),
                    )
                } else {
                    // Default to black even when no font elements
                    Some("#000000".to_string())
                };

                let hyperlink = cell.select("a").first().attr("href").map(|h| h.to_string());
                let is_bold = if cell.select("b").length() > 0 {
                    Some(true)
                } else {
                    None
                };

                let align = cell.attr("align").and_then(|align| align.parse().ok());
                let valign = cell.attr("valign").and_then(|valign| valign.parse().ok());

                // Create CellStyle with defaults for consistent styling
                let style = Some(CellStyle {
                    bg_color,
                    text_color,
                    font_face,
                    is_bold,
                    align,
                    valign,
                });

                // Mark this cell reference as processed
                processed_refs.insert(cell_ref.to_string());

                let cell_info = Cell::new_with_style(
                    text,
                    cell_ref.to_string(),
                    formula,
                    value,
                    hyperlink,
                    style,
                );

                cells.push(cell_info);
            }
        }
    }

    Ok(cells)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_column_index_to_name() {
        assert_eq!(column_index_to_name(0), "A");
        assert_eq!(column_index_to_name(1), "B");
        assert_eq!(column_index_to_name(25), "Z");
        assert_eq!(column_index_to_name(26), "AA");
        assert_eq!(column_index_to_name(27), "AB");
        assert_eq!(column_index_to_name(701), "ZZ");
        assert_eq!(column_index_to_name(702), "AAA");
    }

    #[test]
    fn test_get_cell_reference() {
        // Single cell references
        assert_eq!(get_cell_reference(0, 0, 1, 1), "A1");
        assert_eq!(get_cell_reference(0, 1, 1, 1), "B1");
        assert_eq!(get_cell_reference(1, 0, 1, 1), "A2");
        assert_eq!(get_cell_reference(1, 25, 1, 1), "Z2");

        // Range references (colspan)
        assert_eq!(get_cell_reference(0, 0, 3, 1), "A1:C1");
        assert_eq!(get_cell_reference(1, 1, 2, 1), "B2:C2");

        // Range references (rowspan)
        assert_eq!(get_cell_reference(0, 0, 1, 3), "A1:A3");
        assert_eq!(get_cell_reference(1, 1, 1, 2), "B2:B3");

        // Range references (both colspan and rowspan)
        assert_eq!(get_cell_reference(0, 0, 2, 2), "A1:B2");
        assert_eq!(get_cell_reference(1, 2, 3, 2), "C2:E3");
    }

    #[test]
    fn test_add_cell_references_to_html() {
        let html = r#"<table>
<tr>
<td>A1</td>
<td colspan="2">B1</td>
</tr>
<tr>
<td>A2</td>
<td>B2</td>
<td>C2</td>
</tr>
</table>"#;

        let result = add_cell_references_to_html(html, 0, 0).unwrap();

        // Check that data-cell-ref attributes were added
        assert!(result.contains("data-cell-ref=\"A1\""));
        assert!(result.contains("data-cell-ref=\"B1:C1\""));
        assert!(result.contains("data-cell-ref=\"A2\""));
        assert!(result.contains("data-cell-ref=\"B2\""));
        assert!(result.contains("data-cell-ref=\"C2\""));
    }

    #[test]
    fn test_file_to_sheets() {
        let mut html_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        html_file_path.push("output/excel/html/input.html");
        let html_content = fs::read_to_string(html_file_path).unwrap();
        let result = add_cell_references_to_html(&html_content, 0, 0).unwrap();
        fs::write("output/excel/html/input_with_cell_refs.html", result).unwrap();
    }

    #[test]
    fn test_add_excel_headers_to_html() {
        let html = r#"<table>
<tr>
<td>Data1</td>
<td>Data2</td>
</tr>
<tr>
<td>Data3</td>
<td>Data4</td>
</tr>
</table>"#;

        let result = add_excel_headers_to_html(html, 0, 0).unwrap();

        // Check that column headers were added (A, B)
        assert!(result.contains(">A</th>"));
        assert!(result.contains(">B</th>"));

        // Check that row numbers were added (1, 2)
        assert!(result.contains(">1</td>"));
        assert!(result.contains(">2</td>"));

        // Check that Excel header styling is present in the elements themselves
        assert!(result.contains("background:#E0E0E0"));
    }

    #[test]
    fn test_add_excel_headers_with_start_col() {
        let html = r#"<table>
<tr>
<td>Data1</td>
<td>Data2</td>
</tr>
</table>"#;

        // Start from column C (index 2) and row 1 (index 0)
        let result = add_excel_headers_to_html(html, 2, 0).unwrap();

        // Check that column headers start from C, D instead of A, B
        assert!(result.contains(">C</th>"));
        assert!(result.contains(">D</th>"));
        assert!(!result.contains(">A</th>"));
        assert!(!result.contains(">B</th>"));

        // Check that row numbers were still added
        assert!(result.contains(">1</td>"));
    }

    #[test]
    fn test_add_excel_headers_with_start_row() {
        let html = r#"<table>
<tr>
<td>Data1</td>
<td>Data2</td>
</tr>
<tr>
<td>Data3</td>
<td>Data4</td>
</tr>
</table>"#;

        // Start from column A (index 0) and row 5 (index 4)
        let result = add_excel_headers_to_html(html, 0, 4).unwrap();

        // Check that column headers start from A, B
        assert!(result.contains(">A</th>"));
        assert!(result.contains(">B</th>"));

        // Check that row numbers start from 5, 6 instead of 1, 2
        assert!(result.contains(">5</td>"));
        assert!(result.contains(">6</td>"));
        assert!(!result.contains(">1</td>"));
        assert!(!result.contains(">2</td>"));
    }

    #[test]
    fn test_add_excel_headers_with_both_offsets() {
        let html = r#"<table>
<tr>
<td>Data1</td>
<td>Data2</td>
<td>Data3</td>
</tr>
<tr>
<td>Data4</td>
<td>Data5</td>
<td>Data6</td>
</tr>
</table>"#;

        // Start from column C (index 2) and row 10 (index 9)
        let result = add_excel_headers_to_html(html, 2, 9).unwrap();

        // Check that column headers start from C, D, E
        assert!(result.contains(">C</th>"));
        assert!(result.contains(">D</th>"));
        assert!(result.contains(">E</th>"));
        assert!(!result.contains(">A</th>"));
        assert!(!result.contains(">B</th>"));

        // Check that row numbers start from 10, 11html_content
        assert!(!result.contains(">2</td>"));
    }

    #[test]
    fn test_clean_html_for_llm() {
        let html = r##"<table border="1" cellpadding="2" cellspacing="0" class="test-table" id="main-table">
<tr>
<td data-cell-ref="A1" bgcolor="#FFFFFF" style="border:1px solid black" colspan="2" data-sheets-value="test" face="Arial" color="red" valign="top" height="20px" align="center">
<font size="12" color="blue" face="Arial">Cell A1 Content</font>
</td>
<td data-cell-ref="B1" width="100px">
Regular text with <font color="#FF0000">red font text</font> and more content
</td>
</tr>
<tr>
<td>
Text before<br />
<br/>
Text after empty br tags
</td>
<td>Just text</td>
</tr>
</table>"##;

        let result = clean_html_for_llm(html).unwrap();

        // Check that essential attributes are preserved
        assert!(result.contains("data-cell-ref=\"A1\""));
        assert!(result.contains("data-cell-ref=\"B1\""));
        assert!(result.contains("bgcolor=\"#FFFFFF\""));
        assert!(result.contains("style=\"border:1px solid black\""));
        assert!(result.contains("colspan=\"2\""));

        // Check that unnecessary attributes are removed
        assert!(!result.contains("data-sheets-value"));
        assert!(!result.contains("face="));
        assert!(!result.contains(" color="));
        assert!(!result.contains("valign="));
        assert!(!result.contains("height="));
        assert!(!result.contains("align="));
        assert!(!result.contains("width="));
        assert!(!result.contains("border="));
        assert!(!result.contains("cellpadding="));
        assert!(!result.contains("cellspacing="));
        assert!(!result.contains("class="));
        assert!(!result.contains("id="));
        assert!(!result.contains("size="));

        // Check that font tags are removed but text content is preserved
        assert!(!result.contains("<font"));
        assert!(!result.contains("</font>"));
        assert!(result.contains("Cell A1 Content")); // Font tag content preserved
        assert!(result.contains("red font text")); // Nested font tag content preserved

        // Check that empty br tags are removed but text content around them is preserved
        assert!(result.contains("Text before"));
        assert!(result.contains("Text after empty br tags"));

        // Check that the overall structure is maintained
        assert!(result.contains("<table"));
        assert!(result.contains("<tr"));
        assert!(result.contains("<td"));
        assert!(result.contains("Regular text"));
        assert!(result.contains("Just text"));
    }

    #[test]
    fn test_clean_html_for_llm_preserves_text_content() {
        // Test specifically focused on text preservation
        let html = r##"<div>
<p>
Before font: <font color="red" size="3">Important text in font tag</font> after font.
</p>
<p>
Line 1<br />
<br/>
Line 2 after empty br
</p>
<span data-cell-ref="test" class="remove-me" style="keep-me">Span content</span>
</div>"##;

        let result = clean_html_for_llm(html).unwrap();

        // Verify text content is preserved
        assert!(result.contains("Before font: Important text in font tag after font."));
        assert!(result.contains("Line 1"));
        assert!(result.contains("Line 2 after empty br"));
        assert!(result.contains("Span content"));

        // Verify font tags are removed
        assert!(!result.contains("<font"));
        assert!(!result.contains("</font>"));

        // Verify essential attributes are kept and unnecessary ones removed
        assert!(result.contains("data-cell-ref=\"test\""));
        assert!(result.contains("style=\"keep-me\""));
        assert!(!result.contains("class=\"remove-me\""));
    }

    #[test]
    fn html_file_clean_stats() {
        let mut html_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        html_file_path.push("output/excel/html/input.html");
        let html_content = fs::read_to_string(html_file_path).unwrap();
        let result = clean_html_for_llm(&html_content).unwrap();

        println!(
            "HTML reduced to {}% of its original size",
            (result.len() as f32 / html_content.len() as f32) * 100.0
        );
    }

    #[test]
    fn test_extract_table_from_html() {
        let html = r#"<!DOCTYPE html>
<html>
<head>
<title>Test Table</title>
</head>
<body>
<table>
<tr>
<td>Data1</td>
<td>Data2</td>
</tr>
</table>
</body>
</html>
"#;
        let result = extract_table_from_html(html).unwrap();
        assert!(result.contains("<table>"));
        assert!(result.contains("<tr>"));
        assert!(result.contains("<td>"));
        assert!(result.contains("Data1"));
        assert!(result.contains("Data2"));
        assert!(!result.contains("<!DOCTYPE html>"));
        assert!(!result.contains("<html>"));
        assert!(!result.contains("<head>"));
        assert!(!result.contains("<title>"));
        assert!(!result.contains("</head>"));
        assert!(!result.contains("</body>"));
        assert!(!result.contains("</html>"));
    }

    #[test]
    fn test_get_html_table_bounds() {
        // Test basic table
        let html = r#"<table>
<tr>
<td>A1</td>
<td>B1</td>
<td>C1</td>
</tr>
<tr>
<td>A2</td>
<td>B2</td>
</tr>
<tr>
<td>A3</td>
</tr>
</table>"#;

        let (end_row, end_col) = get_html_table_bounds(html).unwrap();
        assert_eq!(end_row, 2); // 3 rows (0-2)
        assert_eq!(end_col, 2); // 3 columns max (0-2)

        // Test table with colspan
        let html_with_colspan = r#"<table>
<tr>
<td colspan="3">Merged cell</td>
</tr>
<tr>
<td>A2</td>
<td>B2</td>
</tr>
</table>"#;

        let (end_row, end_col) = get_html_table_bounds(html_with_colspan).unwrap();
        assert_eq!(end_row, 1); // 2 rows (0-1)
        assert_eq!(end_col, 2); // 3 columns max due to colspan (0-2)

        // Test empty table
        let empty_html = r#"<table></table>"#;
        let (end_row, end_col) = get_html_table_bounds(empty_html).unwrap();
        assert_eq!(end_row, 0);
        assert_eq!(end_col, 0);

        // Test no table
        let no_table_html = r#"<div>No table here</div>"#;
        let (end_row, end_col) = get_html_table_bounds(no_table_html).unwrap();
        assert_eq!(end_row, 0);
        assert_eq!(end_col, 0);

        // Test mixed td and th elements
        let mixed_html = r#"<table>
<tr>
<th>Header1</th>
<th>Header2</th>
<th>Header3</th>
<th>Header4</th>
</tr>
<tr>
<td>Data1</td>
<td>Data2</td>
</tr>
</table>"#;

        let (end_row, end_col) = get_html_table_bounds(mixed_html).unwrap();
        assert_eq!(end_row, 1); // 2 rows (0-1)
        assert_eq!(end_col, 3); // 4 columns max (0-3)
    }

    #[test]
    fn test_extract_rows_from_indicies_with_cell_refs() {
        // Test HTML with data-cell-ref attributes (simulating LibreOffice with column offset)
        let html_with_cell_refs = r#"<table>
<tr>
<td data-cell-ref="C1">Data at C1</td>
<td data-cell-ref="D1">Data at D1</td>
<td data-cell-ref="E1">Data at E1</td>
<td data-cell-ref="F1">Data at F1</td>
</tr>
<tr>
<td data-cell-ref="C2">Data at C2</td>
<td data-cell-ref="D2">Data at D2</td>
<td data-cell-ref="E2">Data at E2</td>
<td data-cell-ref="F2">Data at F2</td>
</tr>
<tr>
<td data-cell-ref="C3">Data at C3</td>
<td data-cell-ref="D3">Data at D3</td>
<td data-cell-ref="E3">Data at E3</td>
<td data-cell-ref="F3">Data at F3</td>
</tr>
</table>"#;

        // Extract rows 0-1 (first two rows) and columns 3-4 (D and E columns)
        let indices = Indices {
            start_row: 0,
            start_col: 3, // Column D (0-based index 3)
            end_row: 1,
            end_col: 4, // Column E (0-based index 4)
        };

        let result =
            extract_rows_from_indicies(html_with_cell_refs, &indices, Some("<table>")).unwrap();

        // Should contain the wrapper table tags
        assert!(result.contains("<table>"));
        assert!(result.contains("</table>"));

        // Should contain the correct cells based on data-cell-ref attributes
        assert!(result.contains("Data at D1"));
        assert!(result.contains("Data at E1"));
        assert!(result.contains("Data at D2"));
        assert!(result.contains("Data at E2"));

        // Should NOT contain cells outside the range
        assert!(!result.contains("Data at C1"));
        assert!(!result.contains("Data at F1"));
        assert!(!result.contains("Data at C2"));
        assert!(!result.contains("Data at F2"));
        assert!(!result.contains("Data at C3")); // Row 2 is outside our range
        assert!(!result.contains("Data at D3")); // Row 2 is outside our range

        // Should preserve data-cell-ref attributes
        assert!(result.contains("data-cell-ref=\"D1\""));
        assert!(result.contains("data-cell-ref=\"E1\""));
        assert!(result.contains("data-cell-ref=\"D2\""));
        assert!(result.contains("data-cell-ref=\"E2\""));
    }

    #[test]
    fn test_extract_rows_from_indicies_fallback_without_cell_refs() {
        // Test HTML without data-cell-ref attributes (fallback to old behavior)
        let html_without_cell_refs = r#"<table>
<tr>
<td>Col0</td>
<td>Col1</td>
<td>Col2</td>
<td>Col3</td>
</tr>
<tr>
<td>Row1Col0</td>
<td>Row1Col1</td>
<td>Row1Col2</td>
<td>Row1Col3</td>
</tr>
</table>"#;

        // Extract rows 0-1 and columns 1-2
        let indices = Indices {
            start_row: 0,
            start_col: 1,
            end_row: 1,
            end_col: 2,
        };

        let result = extract_rows_from_indicies(html_without_cell_refs, &indices, None).unwrap();

        // Should contain the correct cells based on array indices
        assert!(result.contains("Col1"));
        assert!(result.contains("Col2"));
        assert!(result.contains("Row1Col1"));
        assert!(result.contains("Row1Col2"));

        // Should NOT contain cells outside the range
        assert!(!result.contains("Col0"));
        assert!(!result.contains("Col3"));
        assert!(!result.contains("Row1Col0"));
        assert!(!result.contains("Row1Col3"));
    }

    #[test]
    fn test_extract_rows_from_indicies_preserves_styling_attributes() {
        // Test HTML with styling attributes to ensure they are preserved
        let html_with_styling = r#"<table>
<tr>
<td data-cell-ref="A1" data-sheets-formula="=SUM(B1:B10)" bgcolor="FFFF00" style="font-weight:bold;">Total</td>
<td data-cell-ref="B1" sdval="100.5" data-sheets-numberformat="{ &quot;1&quot;: 2 }">100.50</td>
<td data-cell-ref="C1" align="center" valign="top"><font color="FF0000" face="Arial">Header</font></td>
</tr>
<tr>
<td data-cell-ref="A2" bgcolor="E0E0E0">Data 1</td>
<td data-cell-ref="B2" data-sheets-formula="=A2*2" style="text-align:right;">Data 2</td>
<td data-cell-ref="C2">Data 3</td>
</tr>
</table>"#;

        // Extract first row, all columns
        let indices = Indices {
            start_row: 0,
            start_col: 0,
            end_row: 0,
            end_col: 2,
        };

        let result =
            extract_rows_from_indicies(html_with_styling, &indices, Some("<table>")).unwrap();

        // Should preserve formula attributes
        assert!(result.contains("data-sheets-formula=\"=SUM(B1:B10)\""));

        // Should preserve background color
        assert!(result.contains("bgcolor=\"FFFF00\""));

        // Should preserve style attributes
        assert!(result.contains("style=\"font-weight:bold;\""));

        // Should extract and preserve font information as data attributes
        assert!(result.contains("data-font-color=\"FF0000\""));
        assert!(result.contains("data-font-face=\"Arial\""));

        // Should preserve inner HTML structure (font tags)
        assert!(result.contains("<font color=\"FF0000\" face=\"Arial\">Header</font>"));

        // Should still preserve cell references
        assert!(result.contains("data-cell-ref=\"A1\""));
        assert!(result.contains("data-cell-ref=\"B1\""));
        assert!(result.contains("data-cell-ref=\"C1\""));

        // Content should be preserved
        assert!(result.contains("Total"));
        assert!(result.contains("100.50"));
        assert!(result.contains("Header"));
    }

    #[test]
    fn test_get_cell_ref_for_image_src() {
        // Test direct image in cell
        let html_direct = r#"<table>
<tr>
<td data-cell-ref="A1"><img src="image1.jpg"></td>
<td data-cell-ref="B1">Text content</td>
</tr>
<tr>
<td data-cell-ref="A2"><img src="image2.png"></td>
<td data-cell-ref="B2"><img src="image3.gif"></td>
</tr>
</table>"#;

        assert_eq!(
            get_cell_ref_for_image_src(html_direct, "image1.jpg").unwrap(),
            "A1".to_string()
        );
        assert_eq!(
            get_cell_ref_for_image_src(html_direct, "image2.png").unwrap(),
            "A2".to_string()
        );
        assert_eq!(
            get_cell_ref_for_image_src(html_direct, "image3.gif").unwrap(),
            "B2".to_string()
        );

        // Test nested image in font tag
        let nested_html = "<table><tr><td colspan=\"3\" rowspan=\"4\" data-cell-ref=\"J1:L4\"><font><img src=\"temp_file.jpg\"></font></td></tr></table>";

        assert_eq!(
            get_cell_ref_for_image_src(nested_html, "temp_file.jpg").unwrap(),
            "J1:L4".to_string()
        );

        // Test image not found - should return error
        let result = get_cell_ref_for_image_src(html_direct, "nonexistent.jpg");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not found in HTML content"));

        // Test image in cell without data-cell-ref - should return error
        let html_no_cell_ref = r#"<table>
<tr>
<td><img src="image.jpg"></td>
</tr>
</table>"#;

        let result = get_cell_ref_for_image_src(html_no_cell_ref, "image.jpg");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("has no data-cell-ref attribute"));

        // Test deeply nested image
        let html_deep_nested = r#"<table>
<tr>
<td data-cell-ref="C3">
<div>
<span>
<font>
<img src="deep_image.jpg">
</font>
</span>
</div>
</td>
</tr>
</table>"#;

        assert_eq!(
            get_cell_ref_for_image_src(html_deep_nested, "deep_image.jpg").unwrap(),
            "C3".to_string()
        );
    }

    #[test]
    fn test_extract_cells_from_ranges() {
        // Test HTML with comprehensive cell data
        let html = r#"<table>
<tr>
<td data-cell-ref="A1" data-sheets-formula="=SUM(B1:B10)" data-sheets-numberformat="{ &quot;1&quot;: 2 }">Total</td>
<td data-cell-ref="B1" sdnum="1033">100.50</td>
<td data-cell-ref="C1">Header</td>
</tr>
<tr>
<td data-cell-ref="A2">Data 1</td>
<td data-cell-ref="B2" data-sheets-formula="=A2*2">Data 2</td>
<td data-cell-ref="C2">Data 3</td>
</tr>
<tr>
<td data-cell-ref="A3">Data 4</td>
<td data-cell-ref="B3">Data 5</td>
<td data-cell-ref="C3">Data 6</td>
</tr>
</table>"#;

        // Test extracting specific table range
        let cells = extract_cells_from_ranges(html, Some("A1:B2"), None).unwrap();
        assert_eq!(cells.len(), 4);

        // Check that the correct cells are included
        let cell_refs: Vec<&str> = cells.iter().map(|c| c.range.as_str()).collect();
        assert!(cell_refs.contains(&"A1"));
        assert!(cell_refs.contains(&"B1"));
        assert!(cell_refs.contains(&"A2"));
        assert!(cell_refs.contains(&"B2"));
        assert!(!cell_refs.contains(&"C1"));
        assert!(!cell_refs.contains(&"C2"));

        // Test extracting with header range (should merge and dedupe)
        let cells = extract_cells_from_ranges(html, Some("A2:B3"), Some("A1:C1")).unwrap();
        assert_eq!(cells.len(), 7); // A1, B1, C1, A2, B2, A3, B3

        // Check formula extraction
        let a1_cell = cells.iter().find(|c| c.range == "A1").unwrap();
        assert_eq!(a1_cell.formula.as_ref().unwrap(), "=SUM(B1:B10)");
        assert!(a1_cell.formula.is_some());
        assert_eq!(a1_cell.text, "Total");

        // Check B1 cell - should not have formula since sdnum is not a formula
        let b1_cell = cells.iter().find(|c| c.range == "B1").unwrap();
        assert!(b1_cell.formula.is_none()); // sdnum is not a formula
        assert_eq!(b1_cell.text, "100.50");

        // Test extracting all cells when no ranges specified
        let all_cells = extract_cells_from_ranges(html, None, None).unwrap();
        assert_eq!(all_cells.len(), 9); // All cells A1-C3

        // Test invalid range handling
        let cells = extract_cells_from_ranges(html, Some("Z99:Z100"), None).unwrap();
        assert_eq!(cells.len(), 0); // No cells in that range

        // Test duplicate prevention - overlapping ranges should not duplicate cells
        let cells = extract_cells_from_ranges(html, Some("A1:B2"), Some("B1:C2")).unwrap();
        assert_eq!(cells.len(), 6); // A1, B1, A2, B2, C1, C2 (B1, B2 not duplicated)
        let cell_refs: Vec<&str> = cells.iter().map(|c| c.range.as_str()).collect();
        assert_eq!(cell_refs.iter().filter(|&&r| r == "B1").count(), 1);
        assert_eq!(cell_refs.iter().filter(|&&r| r == "B2").count(), 1);
    }

    #[test]
    fn test_extract_cells_with_styling() {
        // Test HTML with styling information similar to the user's examples
        let html = "<table>
<tr>
<td data-cell-ref=\"A1\" data-sheets-formula=\"='EQUITY WATER FALL'!R[-6]C[-12]\">
  <font face=\"Daytona\" color=\"#000000\"> 9,697,582 </font>
</td>
<td data-cell-ref=\"B1\" align=\"left\" valign=\"bottom\" bgcolor=\"#DAE3F3\" data-sheets-value=\"{ &quot;1&quot;: 2, &quot;2&quot;: &quot;&quot;}\">
  <font face=\"Daytona\" color=\"#000000\"><br></font>
</td>
<td data-cell-ref=\"C1\" bgcolor=\"#FFFFFF\">
  <font face=\"Arial\" color=\"#FF0000\">Test Text</font>
</td>
</tr>
</table>";

        let cells = extract_cells_from_ranges(html, None, None).unwrap();
        assert_eq!(cells.len(), 3);

        // Check A1 cell - should have formula and font styling
        let a1_cell = cells.iter().find(|c| c.range == "A1").unwrap();
        assert_eq!(a1_cell.text.trim(), "9,697,582");
        assert_eq!(
            a1_cell.formula.as_ref().unwrap(),
            "='EQUITY WATER FALL'!R[-6]C[-12]"
        );
        let a1_style = a1_cell.style.as_ref().unwrap();
        assert_eq!(a1_style.font_face.as_ref().unwrap(), "Daytona");
        assert_eq!(a1_style.text_color.as_ref().unwrap(), "#000000");
        assert!(a1_style.bg_color.is_none()); // No bgcolor attribute

        // Check B1 cell - should have background color and font styling
        let b1_cell = cells.iter().find(|c| c.range == "B1").unwrap();
        let b1_style = b1_cell.style.as_ref().unwrap();
        assert_eq!(b1_style.bg_color.as_ref().unwrap(), "#DAE3F3");
        assert_eq!(b1_style.font_face.as_ref().unwrap(), "Daytona");
        assert_eq!(b1_style.text_color.as_ref().unwrap(), "#000000");

        // Check C1 cell - should have different font and colors
        let c1_cell = cells.iter().find(|c| c.range == "C1").unwrap();
        assert_eq!(c1_cell.text, "Test Text");
        let c1_style = c1_cell.style.as_ref().unwrap();
        assert_eq!(c1_style.bg_color.as_ref().unwrap(), "#FFFFFF");
        assert_eq!(c1_style.font_face.as_ref().unwrap(), "Arial");
        assert_eq!(c1_style.text_color.as_ref().unwrap(), "#FF0000");
    }

    #[test]
    fn test_extract_cells_with_hyperlinks_and_bold() {
        // Test HTML with hyperlinks and bold styling
        let html = "<table>
<tr>
<td data-cell-ref=\"A1\" bgcolor=\"#FFFFFF\">
<a href=\"https://www.google.com/maps/example\">Click Here</a>
</td>
<td data-cell-ref=\"B1\" bgcolor=\"#000000\">
<b><font face=\"Daytona\" color=\"#FFFFFF\">PROJECT BUDGET</font></b>
</td>
<td data-cell-ref=\"C1\" bgcolor=\"#FFFFFF\">
<font face=\"Arial\" color=\"#000000\">Regular Text</font>
</td>
</tr>
</table>";

        let cells = extract_cells_from_ranges(html, None, None).unwrap();
        assert_eq!(cells.len(), 3);

        // Check A1 cell - should have hyperlink
        let a1_cell = cells.iter().find(|c| c.range == "A1").unwrap();
        assert_eq!(a1_cell.text, "Click Here");
        assert!(a1_cell.hyperlink.is_some());
        assert_eq!(
            a1_cell.hyperlink.as_ref().unwrap(),
            "https://www.google.com/maps/example"
        );
        let a1_style = a1_cell.style.as_ref().unwrap();
        assert_eq!(a1_style.bg_color.as_ref().unwrap(), "#FFFFFF");
        assert!(a1_style.is_bold.is_none()); // No bold tag

        // Check B1 cell - should have bold styling
        let b1_cell = cells.iter().find(|c| c.range == "B1").unwrap();
        assert_eq!(b1_cell.text, "PROJECT BUDGET");
        assert!(b1_cell.hyperlink.is_none()); // No hyperlink
        let b1_style = b1_cell.style.as_ref().unwrap();
        assert_eq!(b1_style.is_bold, Some(true));
        assert_eq!(b1_style.font_face.as_ref().unwrap(), "Daytona");
        assert_eq!(b1_style.text_color.as_ref().unwrap(), "#FFFFFF");
        assert_eq!(b1_style.bg_color.as_ref().unwrap(), "#000000");

        // Check C1 cell - should have neither hyperlink nor bold
        let c1_cell = cells.iter().find(|c| c.range == "C1").unwrap();
        assert_eq!(c1_cell.text, "Regular Text");
        assert!(c1_cell.hyperlink.is_none());
        let c1_style = c1_cell.style.as_ref().unwrap();
        assert!(c1_style.is_bold.is_none());
        assert_eq!(c1_style.font_face.as_ref().unwrap(), "Arial");
        assert_eq!(c1_style.text_color.as_ref().unwrap(), "#000000");
        assert_eq!(c1_style.bg_color.as_ref().unwrap(), "#FFFFFF");
    }

    #[test]
    fn test_indices_to_range() {
        // Test single cell
        let indices = Indices {
            start_row: 0,
            start_col: 0,
            end_row: 0,
            end_col: 0,
        };
        let range = indices_to_range(&indices);
        assert_eq!(range, "A1");

        // Test multi-cell range
        let indices = Indices {
            start_row: 0,
            start_col: 0,
            end_row: 2,
            end_col: 3,
        };
        let range = indices_to_range(&indices);
        assert_eq!(range, "A1:D3");

        // Test range with different starting position
        let indices = Indices {
            start_row: 4,
            start_col: 25,
            end_row: 4,
            end_col: 27,
        };
        let range = indices_to_range(&indices);
        assert_eq!(range, "Z5:AB5");

        // Test range with columns beyond Z
        let indices = Indices {
            start_row: 0,
            start_col: 26,
            end_row: 0,
            end_col: 28,
        };
        let range = indices_to_range(&indices);
        assert_eq!(range, "AA1:AC1");

        // Test single cell in later columns
        let indices = Indices {
            start_row: 9,
            start_col: 701,
            end_row: 9,
            end_col: 701,
        };
        let range = indices_to_range(&indices);
        assert_eq!(range, "ZZ10");
    }

    #[test]
    fn test_remove_leading_empty_rows() {
        // Test HTML with empty rows at the beginning (LibreOffice style)
        let html_with_empty_rows = r#"<table>
<tr>
<td bgcolor="DAE3F3"><br></td>
<td bgcolor="DAE3F3"><br></td>
<td bgcolor="DAE3F3"><br></td>
</tr>
<tr>
<td bgcolor="DAE3F3">  </td>
<td bgcolor="DAE3F3">   </td>
<td bgcolor="DAE3F3"></td>
</tr>
<tr>
<td>Real Data 1</td>
<td>Real Data 2</td>
<td>Real Data 3</td>
</tr>
<tr>
<td>More Data 1</td>
<td>More Data 2</td>
<td>More Data 3</td>
</tr>
<tr>
<td></td>
<td>Empty in middle</td>
<td></td>
</tr>
</table>"#;

        let result = remove_leading_empty_rows(html_with_empty_rows).unwrap();

        // Should remove the first two empty rows but keep content
        assert!(result.contains("Real Data 1"));
        assert!(result.contains("More Data 1"));
        assert!(result.contains("Empty in middle")); // Should keep empty rows after content

        // Count the number of <tr> tags to verify row removal
        let original_row_count = html_with_empty_rows.matches("<tr>").count();
        let result_row_count = result.matches("<tr>").count();
        assert_eq!(result_row_count, original_row_count - 2); // Removed 2 empty rows

        // Test HTML with no empty rows at the beginning
        let html_no_empty_rows = r#"<table>
<tr>
<td>Data 1</td>
<td>Data 2</td>
</tr>
<tr>
<td>Data 3</td>
<td>Data 4</td>
</tr>
</table>"#;

        let result_no_change = remove_leading_empty_rows(html_no_empty_rows).unwrap();
        // Should be unchanged
        assert_eq!(result_no_change.matches("<tr>").count(), 2);
        assert!(result_no_change.contains("Data 1"));

        // Test HTML with all empty rows
        let html_all_empty = r#"<table>
<tr>
<td><br></td>
<td>  </td>
</tr>
<tr>
<td></td>
<td>   </td>
</tr>
</table>"#;

        let result_all_empty = remove_leading_empty_rows(html_all_empty).unwrap();
        // Should return original content if all rows are empty
        assert_eq!(result_all_empty, html_all_empty);

        // Test HTML with no table
        let html_no_table = r#"<div>No table here</div>"#;
        let result_no_table = remove_leading_empty_rows(html_no_table).unwrap();
        assert_eq!(result_no_table, html_no_table);
    }
}
