use once_cell::sync::Lazy;
use regex::Regex;

static TABLE_CONTENT_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<table[^>]*>(.*?)</table>").unwrap());
static TR_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<tr[^>]*>(.*?)<\/tr>").unwrap());
static TD_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<(?:td|th)\s*(?:colspan\s*=\s*['"]?(\d+)['"]?)?(?:\s*+rowspan\s*=\s*['"]?(\d+)['"]?)?[^>]*>(.*?)</(?:td|th)>"#).unwrap()
});
static IMG_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"<img(?:[^>]*?alt=["']([^"']*?)["'])?[^>]*>"#).unwrap());
static TAG_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"</?([a-zA-Z][a-zA-Z0-9]*).*?>").unwrap());

// TODO: Deal with multiple tables
pub fn extract_table_html(html: String) -> String {
    let mut contents = Vec::new();
    for caps in TABLE_CONTENT_REGEX.captures_iter(&html) {
        if let Some(content) = caps.get(1) {
            contents.push(format!("<table>{}</table>", content.as_str()));
        }
    }
    match contents.first() {
        Some(content) => content.to_string(),
        None => String::new(),
    }
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

pub fn validate_html(html: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut tag_stack = Vec::new();

    const VOID_ELEMENTS: [&str; 14] = [
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr",
    ];

    for cap in TAG_REGEX.captures_iter(html) {
        let tag = cap[1].to_string();
        let full_match = cap[0].to_string();

        if full_match.starts_with("</") {
            if let Some(last_tag) = tag_stack.pop() {
                if last_tag != tag {
                    return Err(format!(
                        "Mismatched HTML tags: expected </{}>, found </{}>",
                        last_tag, tag
                    )
                    .into());
                }
            } else {
                return Err(
                    format!("Found closing tag </{}> without matching opening tag", tag).into(),
                );
            }
        } else if !full_match.ends_with("/>") && !VOID_ELEMENTS.contains(&tag.as_str()) {
            tag_stack.push(tag);
        }
    }

    if !tag_stack.is_empty() {
        return Err(format!("Unclosed HTML tags: {}", tag_stack.join(", ")).into());
    }

    Ok(())
}

pub fn convert_table_to_markdown(html: String) -> String {
    let mut markdown = String::new();
    let result: Result<(), Box<dyn std::error::Error>> = (|| {
        let rows = TR_REGEX
            .captures_iter(&html)
            .filter(|m| m.get(1).is_some())
            .collect::<Vec<_>>();

        let mut header_count: usize = 0;

        // Get header count to create table matrix
        if let Some(first_row) = rows.first() {
            if let Some(row) = first_row.get(1) {
                for col_match in TD_REGEX.captures_iter(row.as_str()) {
                    if col_match.get(3).is_some() {
                        let colspan = col_match
                            .get(1)
                            .map_or(1, |m| m.as_str().parse::<i32>().unwrap_or(1));
                        header_count += colspan as usize;
                    }
                }
            }
        }

        let mut table: Vec<Vec<Option<String>>> = vec![vec![None; header_count]; rows.len() + 1];

        for (row_count, row_match) in rows.iter().enumerate() {
            let mut col_count = 0;
            if let Some(row) = row_match.get(1) {
                // Process row cells
                for col_match in TD_REGEX.captures_iter(row.as_str()) {
                    let mut row_index = if row_count == 0 {
                        row_count
                    } else {
                        row_count + 1
                    };

                    if let Some(col) = col_match.get(3) {
                        let rowspan = col_match
                            .get(2)
                            .map_or(1, |m| m.as_str().parse::<i32>().unwrap_or(1));
                        let colspan = col_match
                            .get(1)
                            .map_or(1, |m| m.as_str().parse::<i32>().unwrap_or(1));

                        // Find next available column
                        while col_count < header_count {
                            let row = table.get_mut(row_index).ok_or("Row index out of bounds")?;
                            let cell =
                                row.get_mut(col_count).ok_or("Column index out of bounds")?;

                            match cell {
                                Some(_) => col_count += 1,
                                None => break,
                            }
                        }

                        // Set main cell
                        let row = table.get_mut(row_index).ok_or("Row index out of bounds")?;
                        let cell = row.get_mut(col_count).ok_or("Column index out of bounds")?;
                        *cell = Some(col.as_str().to_string());
                        col_count += 1;

                        // Handle colspan
                        for _ in 0..(colspan - 1) {
                            let row = table.get_mut(row_index).ok_or("Row index out of bounds")?;
                            let cell =
                                row.get_mut(col_count).ok_or("Column index out of bounds")?;
                            *cell = Some("".to_string());
                            col_count += 1;
                        }

                        // Adjust row_index for rowspan processing
                        row_index += if row_count == 0 { 2 } else { 1 };

                        // Handle rowspan
                        for row_offset in 0..(rowspan - 1) {
                            col_count = 0;
                            row_index += row_offset as usize;
                            // Find next available column
                            while col_count < header_count {
                                let row =
                                    table.get_mut(row_index).ok_or("Row index out of bounds")?;
                                let cell =
                                    row.get_mut(col_count).ok_or("Column index out of bounds")?;

                                match cell {
                                    Some(_) => col_count += 1,
                                    None => break,
                                }
                            }
                            let row = table.get_mut(row_index).ok_or("Row index out of bounds")?;
                            let cell =
                                row.get_mut(col_count).ok_or("Column index out of bounds")?;
                            *cell = Some("".to_string());
                            col_count += 1;

                            for _ in 0..(colspan - 1) {
                                let row =
                                    table.get_mut(row_index).ok_or("Row index out of bounds")?;
                                let cell =
                                    row.get_mut(col_count).ok_or("Column index out of bounds")?;
                                *cell = Some("".to_string());
                                col_count += 1;
                            }
                        }
                    }
                }

                // Add separator row after header
                if row_count == 0 {
                    for i in 0..header_count {
                        table[1][i] = Some("---".to_string());
                    }
                }
            }
        }
        table.iter().for_each(|row| {
            row.iter().enumerate().for_each(|(i, cell)| {
                if let Some(cell) = cell {
                    if i == 0 {
                        markdown.push_str(format!("| {} |", cell).as_str());
                    } else {
                        markdown.push_str(format!(" {} |", cell).as_str());
                    }
                }
            });
            markdown.push('\n');
        });
        Ok(())
    })();

    if result.is_err() {
        println!("Error converting table to markdown: {:?}", result.err());
        return String::new();
    }

    markdown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_table_html_test() {
        let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Table Representation</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Year</th>
                <th>Susan</th>
                <th>Gerald</th>
                <th>Bobbie</th>
                <th>Keisha</th>
                <th>Art</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>2017</td>
                <td>570</td>
                <td>635</td>
                <td>684</td>
                <td>397</td>
                <td>678</td>
            </tr>
            <tr>
                <td>2018</td>
                <td>647</td>
                <td>325</td>
                <td>319</td>
                <td>601</td>
                <td>520</td>
            </tr>
            <tr>
                <td>2019</td>
                <td>343</td>
                <td>680</td>
                <td>687</td>
                <td>447</td>
                <td>674</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>425</td>
                <td>542</td>
                <td>553</td>
                <td>477</td>
                <td>648</td>
            </tr>
        </tbody>
    </table>
</body>
</html>"#;
        let table_html = extract_table_html(html.to_string());
        println!("table_html: {}", table_html);
    }
}
