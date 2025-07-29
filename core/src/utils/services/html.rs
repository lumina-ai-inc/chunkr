use once_cell::sync::Lazy;
use regex::Regex;
use std::io::Write;
use std::process::{Command, Stdio};

static TABLE_CONTENT_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<table[^>]*>(.*?)</table>").unwrap());
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
                        "Mismatched HTML tags: expected </{last_tag}>, found </{tag}>"
                    )
                    .into());
                }
            } else {
                return Err(
                    format!("Found closing tag </{tag}> without matching opening tag").into(),
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
        .map_err(|e| {
            format!(
                "Failed to start pandoc: {e}. Make sure pandoc is installed."
            )
        })?;

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
        println!("table_html: {table_html}");
    }
}
