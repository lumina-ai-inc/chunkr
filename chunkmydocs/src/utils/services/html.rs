use once_cell::sync::Lazy;
use regex::Regex;

static TABLE_CONTENT_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)<table[^>]*>(.*?)<\/table>").unwrap());
static TR_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)<tr[^>]*>(.*?)<\/tr>").unwrap());
static TD_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)<(?:td|th)\s*(?:colspan\s*=\s*['"]?(\d+)['"]?)?(?:\s*+rowspan\s*=\s*['"]?(\d+)['"]?)?[^>]*>(.*?)</(?:td|th)>"#).unwrap()
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
                    if let Some(_) = col_match.get(3) {
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
                if row_count == 0 {
                    for col_match in TD_REGEX.captures_iter(row.as_str()) {
                        let mut row_index: usize = row_count;
                        if let Some(col) = col_match.get(3) {
                            let rowspan = col_match
                                .get(2)
                                .map_or(1, |m| m.as_str().parse::<i32>().unwrap_or(1));
                            let colspan = col_match
                                .get(1)
                                .map_or(1, |m| m.as_str().parse::<i32>().unwrap_or(1));
                            table[row_index][col_count] = Some(col.as_str().to_string());
                            col_count += 1;
                            for _ in 0..(colspan - 1) {
                                table[row_index][col_count] = Some("".to_string());
                                col_count += 1;
                            }
                            row_index += 2;
                            for row_offset in 0..(rowspan - 1) {
                                col_count = 0;
                                row_index += row_offset as usize;
                                while col_count < header_count
                                    && table[row_index][col_count].is_some()
                                {
                                    col_count += 1;
                                }
                                table[row_index][col_count] = Some("".to_string());
                                col_count += 1;
                                for _ in 0..(colspan - 1) {
                                    table[row_index][col_count] = Some("".to_string());
                                    col_count += 1;
                                }
                            }
                        }
                    }
                    for i in 0..header_count {
                        table[1][i] = Some("---".to_string());
                    }
                } else {
                    for col_match in TD_REGEX.captures_iter(row.as_str()) {
                        let mut row_index: usize = row_count + 1;
                        if let Some(col) = col_match.get(3) {
                            let rowspan = col_match
                                .get(2)
                                .map_or(1, |m| m.as_str().parse::<i32>().unwrap_or(1));
                            let colspan = col_match
                                .get(1)
                                .map_or(1, |m| m.as_str().parse::<i32>().unwrap_or(1));
                            while col_count < header_count && table[row_index][col_count].is_some()
                            {
                                col_count += 1;
                            }
                            table[row_index][col_count] = Some(col.as_str().to_string());
                            col_count += 1;
                            for _ in 0..(colspan - 1) {
                                table[row_index][col_count] = Some("".to_string());
                                col_count += 1;
                            }
                            row_index += 1;
                            for row_offset in 0..(rowspan - 1) {
                                col_count = 0;
                                row_index += row_offset as usize;
                                while col_count < header_count
                                    && table[row_index][col_count].is_some()
                                {
                                    col_count += 1;
                                }
                                table[row_index][col_count] = Some("".to_string());
                                col_count += 1;
                                for _ in 0..(colspan - 1) {
                                    table[row_index][col_count] = Some("".to_string());
                                    col_count += 1;
                                }
                            }
                        }
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
            markdown.push_str("\n");
        });
        Ok(())
    })();

    if result.is_err() {
        return String::new();
    }

    markdown
}
