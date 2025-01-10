use once_cell::sync::Lazy;
use regex::Regex;

static MD_IMG_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"!\[(.*?)\]\((?:[^)]+)\)").unwrap());

/// Cleans the Markdown image tags
///
/// Replaces Markdown image syntax with just the alt text (if available) or removes them entirely.
/// Useful for converting Markdown to plain text while preserving image descriptions.
///
/// ### Examples
///
/// ```
/// let markdown = r#"Text ![A picture](pic.jpg) more text"#;
/// let cleaned = clean_markdown_img_tags(markdown);
/// assert_eq!(cleaned, "Text A picture more text");
/// ```
pub fn clean_img_tags(markdown: &str) -> String {
    MD_IMG_REGEX
        .replace_all(markdown, |caps: &regex::Captures| {
            caps.get(1)
                .map_or("".to_string(), |m| m.as_str().to_string())
        })
        .to_string()
}
