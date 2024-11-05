use once_cell::sync::Lazy;
use regex::Regex;

static TABLE_CONTENT_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)<table[^>]*>(.*?)<\/table>").unwrap());
static TR_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)<tr[^>]*>(.*?)<\/tr>").unwrap());
static TD_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)<td\s*(?:colspan\s*=\s*['"]?(\d+)['"]?)?(?:\s*+rowspan\s*=\s*['"]?(\d+)['"]?)?[^>]*>(.*?)<\/td>"#).unwrap()
});

pub fn extract_table_html(html: String) -> String {
    let mut contents = Vec::new();
    for caps in TABLE_CONTENT_REGEX.captures_iter(&html) {
        if let Some(content) = caps.get(1) {
            contents.push(format!("<table>{}</table>", content.as_str()));
        }
    }
    contents.first().unwrap().to_string()
}

// TODO: handle rowspan empty cells
pub fn convert_table(html: String) -> String {
    let mut markdown = String::new();
    let mut row_count = 0;
    let mut cell_count = 0;

    for row_match in TR_REGEX.captures_iter(&html) {
        row_count += 1;
        if let Some(row) = row_match.get(1) {
            if row_count == 2 {
                markdown.push_str("|");
                for _ in 0..cell_count {
                    markdown.push_str("---|");
                }
                markdown.push_str("\n");
            }
            markdown.push_str("|");
            for col_match in TD_REGEX.captures_iter(row.as_str()) {
                cell_count += 1;
                let colspan = col_match
                    .get(1)
                    .map_or(1, |m| m.as_str().parse::<i32>().unwrap_or(1));
                let _ = col_match
                    .get(2)
                    .map_or(1, |m| m.as_str().parse::<i32>().unwrap_or(1));
                if let Some(col) = col_match.get(3) {
                    markdown.push_str(format!(" {} |", col.as_str()).as_str());
                }
                for _ in 0..(colspan - 1) {
                    markdown.push_str("  |");
                    cell_count += 1;
                }
            }
            markdown.push_str("\n");
        }
    }
    markdown
}
