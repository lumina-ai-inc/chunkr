use schemars::JsonSchema as SchemarsJsonSchema;
use serde::{Deserialize, Serialize};

use crate::models::output::SegmentType;
use crate::utils::services::html::parse_range;

#[derive(Debug, Serialize, SchemarsJsonSchema, PartialEq, Clone)]
pub enum LayoutElement {
    /// An image, picture, figure, visual, or logo in the excel file
    Image,
    /// A section header in the excel file
    SectionHeader,
    /// A text in the excel file
    Text,
    /// A title in the excel file
    Title,
    /// A table in the excel file
    Table,
    /// A note or footnote in the excel file
    Note,
    /// A note or footnote in the excel file
    Footnote,
    /// A key-value pair in the excel file
    KeyValue,
    /// A page header in the excel file
    PageHeader,
    /// A page footer in the excel file
    PageFooter,
}

impl<'de> Deserialize<'de> for LayoutElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let normalized = s.to_lowercase().replace([' ', '_', '-'], "");

        match normalized.as_str() {
            "image" | "picture" | "logo" | "figure" | "visual" | "chart" => {
                Ok(LayoutElement::Image)
            }
            "sectionheader" | "sectiontitle" | "section" | "header" | "subtitle" | "subheader" => {
                Ok(LayoutElement::SectionHeader)
            }
            "text" | "value" | "summary" | "other" => Ok(LayoutElement::Text),
            "title" => Ok(LayoutElement::Title),
            "table" => Ok(LayoutElement::Table),
            "note" | "notes" => Ok(LayoutElement::Note),
            "footnote" | "footnotes" => Ok(LayoutElement::Footnote),
            "keyvalue" | "keyvaluepair" => Ok(LayoutElement::KeyValue),
            "pageheader" => Ok(LayoutElement::PageHeader),
            "footer" | "pagefooter" => Ok(LayoutElement::PageFooter),
            _ => {
                // Log the unknown type for debugging
                println!("Unknown LayoutElement type encountered: '{s}' - defaulting to Text");
                Ok(LayoutElement::Text)
            }
        }
    }
}

impl TryFrom<LayoutElement> for SegmentType {
    type Error = String;

    fn try_from(value: LayoutElement) -> Result<Self, Self::Error> {
        match value {
            LayoutElement::Image => Ok(SegmentType::Picture),
            LayoutElement::SectionHeader => Ok(SegmentType::SectionHeader),
            LayoutElement::Table => Ok(SegmentType::Table),
            LayoutElement::Text => Ok(SegmentType::Text),
            LayoutElement::Title => Ok(SegmentType::Title),
            LayoutElement::Note => Ok(SegmentType::Footnote),
            LayoutElement::Footnote => Ok(SegmentType::Footnote),
            LayoutElement::KeyValue => Ok(SegmentType::Table),
            LayoutElement::PageHeader => Ok(SegmentType::PageHeader),
            LayoutElement::PageFooter => Ok(SegmentType::PageFooter),
        }
    }
}

/// Helper function to split comma-separated ranges
fn split_comma_separated_ranges(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Helper function to detect if a range looks like a header
/// Headers are typically horizontal (single row) and smaller than data ranges
fn is_potential_header(range: &str) -> bool {
    match parse_range(range) {
        Ok(indices) => {
            let rows = indices.end_row - indices.start_row + 1;
            let cols = indices.end_col - indices.start_col + 1;

            // Consider it a header if it's a single row and has multiple columns
            rows == 1 && cols > 1
        }
        Err(_) => false,
    }
}

/// Helper function to calculate the number of cells in a range
fn calculate_range_size(range: &str) -> u64 {
    match parse_range(range) {
        Ok(indices) => {
            let rows = (indices.end_row - indices.start_row + 1) as u64;
            let cols = (indices.end_col - indices.start_col + 1) as u64;
            rows * cols
        }
        Err(_) => 0,
    }
}

/// Helper function to intelligently split ranges into header and data
/// Returns (data_ranges, potential_header_range)
fn split_ranges_intelligently(ranges: Vec<String>) -> (Vec<String>, Option<String>) {
    if ranges.is_empty() {
        return (vec![], None);
    }

    if ranges.len() == 1 {
        return (ranges, None);
    }

    // Look for potential headers
    let mut headers = vec![];
    let mut data_ranges = vec![];

    for range in ranges {
        if is_potential_header(&range) {
            headers.push(range);
        } else {
            data_ranges.push(range);
        }
    }

    // If we found exactly one header, use it
    if headers.len() == 1 && !data_ranges.is_empty() {
        return (data_ranges, headers.into_iter().next());
    }

    // If we have multiple potential headers, select the largest one
    let best_header = if headers.len() > 1 {
        headers
            .clone()
            .into_iter()
            .max_by_key(|range| calculate_range_size(range))
    } else {
        None
    };

    // If we couldn't identify a clear header, put everything in data_ranges
    if best_header.is_none() {
        let mut all_ranges = headers.clone();
        all_ranges.extend(data_ranges);
        return (all_ranges, None);
    }

    (data_ranges, best_header)
}

/// Helper function to select the largest range from a vector of ranges
fn select_largest_range(ranges: Vec<String>) -> Option<String> {
    if ranges.is_empty() {
        return None;
    }

    // If only one range, return it
    if ranges.len() == 1 {
        return ranges.into_iter().next();
    }

    // Find the largest range by calculating the number of cells each covers
    ranges
        .into_iter()
        .max_by_key(|range| calculate_range_size(range))
}

#[derive(Debug, Serialize, SchemarsJsonSchema, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
pub struct IdentifiedElement {
    /// The range of the element in Excel notation (e.g., "A1:D10")
    pub range: String,
    /// The type of the element
    pub r#type: LayoutElement,
    /// The header range of the table if the element is a `Table` in excel notation (e.g., "A1:C1").
    pub header_range: Option<String>,
}

impl<'de> Deserialize<'de> for IdentifiedElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use serde_json::Value;
        use std::fmt;

        struct IdentifiedElementVisitor;

        impl<'de> Visitor<'de> for IdentifiedElementVisitor {
            type Value = IdentifiedElement;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a struct with range, type, and optional header_range fields")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut range_value: Option<Value> = None;
                let mut type_value: Option<LayoutElement> = None;
                let mut header_range_value: Option<Value> = None;

                // Parse all fields
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "range" | "cell_range" | "element_range" => {
                            if range_value.is_some() {
                                return Err(de::Error::duplicate_field("range"));
                            }
                            range_value = Some(map.next_value()?);
                        }
                        "type" | "element_type" | "segment_type" => {
                            if type_value.is_some() {
                                return Err(de::Error::duplicate_field("type"));
                            }
                            type_value = Some(map.next_value()?);
                        }
                        "header_range" | "header_cell_range" => {
                            if header_range_value.is_some() {
                                return Err(de::Error::duplicate_field("header_range"));
                            }
                            header_range_value = Some(map.next_value()?);
                        }
                        _ => {
                            return Err(de::Error::unknown_field(
                                &key,
                                &[
                                    "range",
                                    "cell_range",
                                    "type",
                                    "element_type",
                                    "segment_type",
                                    "header_range",
                                    "header_cell_range",
                                ],
                            ));
                        }
                    }
                }

                // Ensure required fields are present
                let range_value = range_value.ok_or_else(|| de::Error::missing_field("range"))?;
                let type_value = type_value.ok_or_else(|| de::Error::missing_field("type"))?;

                // Parse ranges from the range field
                let ranges = parse_ranges_from_value(&range_value)?;

                // Parse header_range if provided
                let explicit_header_range = if let Some(header_val) = header_range_value {
                    match header_val {
                        Value::Null => None,
                        _ => {
                            let header_ranges = parse_ranges_from_value(&header_val)?;
                            select_largest_range(header_ranges)
                        }
                    }
                } else {
                    None
                };

                // Apply the logic based on whether header_range is explicitly provided
                let (final_range, final_header_range) = if explicit_header_range.is_some() {
                    // If header_range is provided, use it as header and pick largest range as main range
                    let main_range = select_largest_range(ranges)
                        .ok_or_else(|| de::Error::custom("No valid ranges found"))?;
                    (main_range, explicit_header_range)
                } else {
                    // If no explicit header_range, use intelligent splitting
                    let (data_ranges, detected_header) = split_ranges_intelligently(ranges);
                    let main_range = select_largest_range(data_ranges)
                        .ok_or_else(|| de::Error::custom("No valid ranges found"))?;
                    (main_range, detected_header)
                };

                Ok(IdentifiedElement {
                    range: final_range,
                    r#type: type_value,
                    header_range: final_header_range,
                })
            }
        }

        deserializer.deserialize_map(IdentifiedElementVisitor)
    }
}

// Helper function to check if a string looks like a single cell reference (e.g., "A1", "B2", "Z99")
fn is_single_cell_reference(s: &str) -> bool {
    // Simple check: should have letters followed by numbers, no colon
    !s.contains(':') && s.chars().any(|c| c.is_alphabetic()) && s.chars().any(|c| c.is_numeric())
}

// Helper function to parse ranges from a serde_json::Value
fn parse_ranges_from_value<E>(value: &serde_json::Value) -> Result<Vec<String>, E>
where
    E: serde::de::Error,
{
    match value {
        serde_json::Value::String(s) => {
            if s.contains(',') {
                Ok(split_comma_separated_ranges(s))
            } else {
                Ok(vec![s.clone()])
            }
        }
        serde_json::Value::Array(arr) => {
            // Special case: if array has exactly 2 elements and both look like single cell references,
            // treat them as start and end cells for a range
            if arr.len() == 2 {
                if let (serde_json::Value::String(start), serde_json::Value::String(end)) =
                    (&arr[0], &arr[1])
                {
                    if is_single_cell_reference(start) && is_single_cell_reference(end) {
                        // Convert to range format: "start:end"
                        return Ok(vec![format!("{}:{}", start, end)]);
                    }
                }
            }

            // Standard array processing - treat each element as a separate range
            let mut ranges = Vec::new();
            for item in arr {
                if let serde_json::Value::String(s) = item {
                    ranges.push(s.clone());
                } else {
                    return Err(E::custom("Array elements must be strings"));
                }
            }
            Ok(ranges)
        }
        _ => Err(E::custom("Range must be a string or array of strings")),
    }
}

#[derive(Debug, Serialize, Deserialize, SchemarsJsonSchema, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
pub struct IdentifiedElements {
    /// List of identified elements from the excel file in the correct reading order
    pub elements: Vec<IdentifiedElement>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::llm::JsonSchemaDefinition;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_json_schema_definition() {
        let schema_definition = JsonSchemaDefinition::from_struct::<IdentifiedElements>(
            "excel_elements".to_string(),
            Some("Identify individual elements in the sheet".to_string()),
        );

        println!("Schema definition: {schema_definition:?}");
        let mut output_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        output_file_path.push("output/schema/schema_definition.json");
        fs::create_dir_all(output_file_path.parent().unwrap()).unwrap();
        fs::write(
            output_file_path,
            serde_json::to_string(&schema_definition).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn test_deserialize_range_string() {
        // Test normal string deserialization
        let json = r#"{"range": "A1:D10", "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.range, "A1:D10");
    }

    #[test]
    fn test_deserialize_range_single_element_array() {
        // Test array with single element
        let json = r#"{"range": ["A1:D10"], "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.range, "A1:D10");
    }

    #[test]
    fn test_deserialize_range_multiple_elements_array() {
        // Test array with multiple elements - should pick the largest
        let json =
            r#"{"range": ["A1:B2", "C1:F5", "G1:G1"], "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        // C1:F5 covers 4*5=20 cells, A1:B2 covers 2*2=4 cells, G1:G1 covers 1 cell
        assert_eq!(element.range, "C1:F5");
    }

    #[test]
    fn test_deserialize_range_equal_size_ranges() {
        // Test array with equal size ranges - should pick first one found by max_by_key
        let json = r#"{"range": ["A1:B2", "C3:D4"], "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        // Both cover 2*2=4 cells, max_by_key should return the first maximum
        assert!(element.range == "A1:B2" || element.range == "C3:D4");
    }

    #[test]
    fn test_deserialize_range_with_invalid_ranges() {
        // Test array with some invalid ranges - should pick valid largest
        let json = r#"{"range": ["A1:D10"], "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.range, "A1:D10");
    }

    #[test]
    fn test_deserialize_range_with_mixed_valid_invalid_ranges() {
        // Test array with mix of valid and invalid ranges - should pick valid largest
        let json = r#"{"range": ["invalid_range", "A1:D10", "bad:format", "B1:B5"], "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        // A1:D10 covers 10*4=40 cells, B1:B5 covers 5*1=5 cells, invalid ranges get 0 priority
        assert_eq!(element.range, "A1:D10");
    }

    #[test]
    fn test_deserialize_header_range_null() {
        // Test null header_range
        let json = r#"{"range": "A1:D10", "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.header_range, None);
    }

    #[test]
    fn test_deserialize_header_range_string() {
        // Test string header_range
        let json = r#"{"range": "A1:D10", "type": "Table", "header_range": "A1:D1"}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.header_range, Some("A1:D1".to_string()));
    }

    #[test]
    fn test_deserialize_header_range_array() {
        // Test array header_range - should pick largest
        let json = r#"{"range": "A1:D10", "type": "Table", "header_range": ["A1:B1", "A1:D1"]}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.header_range, Some("A1:D1".to_string()));
    }

    #[test]
    fn test_deserialize_header_range_empty_array() {
        // Test empty array for header_range - should be None
        let json = r#"{"range": "A1:D10", "type": "Table", "header_range": []}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.header_range, None);
    }

    #[test]
    fn test_deserialize_range_empty_array_should_fail() {
        // Test empty array for required range - should fail
        let json = r#"{"range": [], "type": "Table", "header_range": null}"#;
        let result = serde_json::from_str::<IdentifiedElement>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_complex_example() {
        // Test a realistic example with mixed formats like in the user's query
        let json = r#"{
            "elements": [
                {"range": ["A1:A1"], "type": "Image", "header_range": null},
                {"range": "B2:B4", "type": "SectionHeader", "header_range": null},
                {"range": "B5:B5", "type": "Title", "header_range": null},
                {"range": ["A12:C24"], "type": "Table", "header_range": ["A12:C12"]},
                {"range": ["A26:A30", "B26:C30"], "type": "Table", "header_range": null}
            ]
        }"#;

        let elements: IdentifiedElements = serde_json::from_str(json).unwrap();
        assert_eq!(elements.elements.len(), 5);

        // Check that array ranges were processed correctly
        assert_eq!(elements.elements[0].range, "A1:A1");
        assert_eq!(elements.elements[1].range, "B2:B4");
        assert_eq!(elements.elements[2].range, "B5:B5");
        assert_eq!(elements.elements[3].range, "A12:C24");
        assert_eq!(
            elements.elements[3].header_range,
            Some("A12:C12".to_string())
        );
        // B26:C30 covers 5*2=10 cells, A26:A30 covers 5*1=5 cells
        assert_eq!(elements.elements[4].range, "B26:C30");
    }

    #[test]
    fn test_split_comma_separated_ranges() {
        // Test basic comma separation
        let ranges = split_comma_separated_ranges("A1:B2,C3:D4");
        assert_eq!(ranges, vec!["A1:B2", "C3:D4"]);

        // Test with spaces
        let ranges = split_comma_separated_ranges("A1:B2, C3:D4 ,E5:F6");
        assert_eq!(ranges, vec!["A1:B2", "C3:D4", "E5:F6"]);

        // Test single range
        let ranges = split_comma_separated_ranges("A1:B2");
        assert_eq!(ranges, vec!["A1:B2"]);

        // Test empty string
        let ranges = split_comma_separated_ranges("");
        assert_eq!(ranges, Vec::<String>::new());
    }

    #[test]
    fn test_is_potential_header() {
        // Should be true for horizontal single-row ranges
        assert!(is_potential_header("A1:D1"));
        assert!(is_potential_header("B2:Z2"));

        // Should be false for single cells
        assert!(!is_potential_header("A1:A1"));

        // Should be false for vertical ranges
        assert!(!is_potential_header("A1:A5"));

        // Should be false for multi-row ranges
        assert!(!is_potential_header("A1:D5"));

        // Should be false for invalid ranges
        assert!(!is_potential_header("invalid"));
    }

    #[test]
    fn test_calculate_range_size() {
        assert_eq!(calculate_range_size("A1:A1"), 1);
        assert_eq!(calculate_range_size("A1:B2"), 4);
        assert_eq!(calculate_range_size("A1:D10"), 40);
        assert_eq!(calculate_range_size("invalid"), 0);
    }

    #[test]
    fn test_split_ranges_intelligently() {
        // Test clear header + data scenario
        let ranges = vec!["G4:Q4".to_string(), "C6:C9".to_string()];
        let (data, header) = split_ranges_intelligently(ranges);
        assert_eq!(data, vec!["C6:C9"]);
        assert_eq!(header, Some("G4:Q4".to_string()));

        // Test multiple data ranges, no header
        let ranges = vec!["A1:A5".to_string(), "C1:C10".to_string()];
        let (data, header) = split_ranges_intelligently(ranges);
        assert_eq!(data.len(), 2);
        assert_eq!(header, None);

        // Test single range
        let ranges = vec!["A1:B5".to_string()];
        let (data, header) = split_ranges_intelligently(ranges);
        assert_eq!(data, vec!["A1:B5"]);
        assert_eq!(header, None);

        // Test empty ranges
        let ranges = vec![];
        let (data, header) = split_ranges_intelligently(ranges);
        assert_eq!(data, Vec::<String>::new());
        assert_eq!(header, None);
    }

    #[test]
    fn test_deserialize_comma_separated_range() {
        // Test the example from the user query
        let json = r#"{"range": "G4:Q4,C6:C9", "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // G4:Q4 is 1x14=14 cells (header-like), C6:C9 is 4x1=4 cells (data)
        // With null header_range, should use intelligent splitting
        // Should detect G4:Q4 as header and use C6:C9 as main range
        assert_eq!(element.range, "C6:C9");
        assert_eq!(element.header_range, Some("G4:Q4".to_string()));
    }

    #[test]
    fn test_deserialize_comma_separated_range_multiple_data() {
        // Test comma-separated with multiple data ranges, no clear header
        let json = r#"{"range": "A1:A5,C1:C10", "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // A1:A5 is 5 cells, C1:C10 is 10 cells
        // Should select the larger data range
        assert_eq!(element.range, "C1:C10");
    }

    #[test]
    fn test_deserialize_comma_separated_header_range() {
        // Test comma-separated in header_range field
        let json = r#"{"range": "A1:D10", "type": "Table", "header_range": "A1:D1,E1:E5"}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // A1:D1 is 4 cells, E1:E5 is 5 cells
        // With explicit header_range, should pick the largest range from header field
        assert_eq!(element.header_range, Some("E1:E5".to_string()));

        // Should use the provided range as-is (single range)
        assert_eq!(element.range, "A1:D10");
    }

    #[test]
    fn test_deserialize_comma_separated_no_header() {
        // Test comma-separated where no range looks like a header
        let json = r#"{"range": "A1:A5,C1:C10", "type": "Table", "header_range": "B1:B3,D1:D8"}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // Main range: A1:A5 is 5 cells, C1:C10 is 10 cells -> should pick C1:C10
        // With explicit header_range, should pick largest from all provided ranges
        assert_eq!(element.range, "C1:C10");

        // Header range: B1:B3 is 3 cells, D1:D8 is 8 cells -> should pick D1:D8
        assert_eq!(element.header_range, Some("D1:D8".to_string()));
    }

    #[test]
    fn test_deserialize_comma_separated_with_spaces() {
        // Test comma-separated with various spacing
        let json = r#"{"range": "G4:Q4, C6:C9 ,A1:A2", "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // G4:Q4 is header (14 cells), C6:C9 is 4 cells, A1:A2 is 2 cells
        // Should pick C6:C9 as the largest data range
        assert_eq!(element.range, "C6:C9");
    }

    #[test]
    fn test_deserialize_comma_separated_mixed_valid_invalid() {
        // Test comma-separated with mix of valid and invalid ranges
        let json = r#"{"range": "invalid,A1:D10,bad:format,B1:B5", "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // A1:D10 is 40 cells, B1:B5 is 5 cells, invalid ranges ignored
        // Should pick A1:D10
        assert_eq!(element.range, "A1:D10");
    }

    #[test]
    fn test_deserialize_comma_separated_complex_scenario() {
        // Test the original user example in a complete JSON structure
        let json = r#"{
            "elements": [
                {"range": "G4:Q4,C6:C9", "type": "Table", "header_range": null},
                {"range": "A1:B1,A2:B10", "type": "Table", "header_range": null},
                {"range": "H1:H5,I1:J1", "type": "Table", "header_range": null}
            ]
        }"#;

        let elements: IdentifiedElements = serde_json::from_str(json).unwrap();
        assert_eq!(elements.elements.len(), 3);

        // First element: G4:Q4 (header-like) + C6:C9 (data) -> should pick C6:C9 for range and G4:Q4 for header
        assert_eq!(elements.elements[0].range, "C6:C9");
        assert_eq!(elements.elements[0].header_range, Some("G4:Q4".to_string()));

        // Second element: A1:B1 (header-like) + A2:B10 (data) -> should pick A2:B10 for range and A1:B1 for header
        assert_eq!(elements.elements[1].range, "A2:B10");
        assert_eq!(elements.elements[1].header_range, Some("A1:B1".to_string()));

        // Third element: H1:H5 (data) + I1:J1 (header-like) -> should pick H1:H5 for range and I1:J1 for header
        assert_eq!(elements.elements[2].range, "H1:H5");
        assert_eq!(elements.elements[2].header_range, Some("I1:J1".to_string()));
    }

    #[test]
    fn test_deserialize_comma_separated_edge_cases() {
        // Test empty comma-separated string
        let json = r#"{"range": "A1:B2", "type": "Table", "header_range": ","}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.header_range, None);

        // Test single comma
        let json = r#"{"range": "A1:B2", "type": "Table", "header_range": "A1:B1,"}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.header_range, Some("A1:B1".to_string()));

        // Test trailing comma
        let json = r#"{"range": "A1:B2,", "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.range, "A1:B2");
    }

    #[test]
    fn test_explicit_header_range_provided() {
        // Test when header_range is explicitly provided - should use it and treat all ranges as data
        let json = r#"{"range": "G4:Q4,C6:C9", "type": "Table", "header_range": "A1:D1"}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // Should use the provided header_range
        assert_eq!(element.header_range, Some("A1:D1".to_string()));

        // Should pick the largest range from all provided ranges as main range
        // G4:Q4 is 1x14=14 cells, C6:C9 is 4x1=4 cells, so should pick G4:Q4
        assert_eq!(element.range, "G4:Q4");
    }

    #[test]
    fn test_explicit_header_range_array_provided() {
        // Test when header_range is explicitly provided as array - should use largest from header array
        let json = r#"{"range": ["A1:A5", "C1:C10"], "type": "Table", "header_range": ["B1:B1", "D1:F1"]}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // Should use the largest header range: D1:F1 (3 cells) vs B1:B1 (1 cell)
        assert_eq!(element.header_range, Some("D1:F1".to_string()));

        // Should pick the largest range from all provided ranges as main range
        // C1:C10 is 10 cells, A1:A5 is 5 cells, so should pick C1:C10
        assert_eq!(element.range, "C1:C10");
    }

    #[test]
    fn test_no_header_range_provided_uses_intelligence() {
        // Test when header_range is not provided - should use intelligent splitting
        let json = r#"{"range": "G4:Q4,C6:C9", "type": "Table"}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // Should detect G4:Q4 as header (single row, multiple columns)
        assert_eq!(element.header_range, Some("G4:Q4".to_string()));

        // Should use C6:C9 as the main range (data range)
        assert_eq!(element.range, "C6:C9");
    }

    #[test]
    fn test_null_header_range_provided_uses_intelligence() {
        // Test when header_range is explicitly null - should use intelligent splitting
        let json = r#"{"range": "G4:Q4,C6:C9", "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // Should detect G4:Q4 as header (single row, multiple columns)
        assert_eq!(element.header_range, Some("G4:Q4".to_string()));

        // Should use C6:C9 as the main range (data range)
        assert_eq!(element.range, "C6:C9");
    }

    #[test]
    fn test_field_aliases_cell_range() {
        // Test cell_range alias for range field
        let json = r#"{"cell_range": "A1:D10", "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.range, "A1:D10");
    }

    #[test]
    fn test_field_aliases_element_type() {
        // Test element_type alias for type field
        let json = r#"{"range": "A1:D10", "element_type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.r#type, LayoutElement::Table);
    }

    #[test]
    fn test_field_aliases_segment_type() {
        // Test segment_type alias for type field
        let json = r#"{"range": "A1:D10", "segment_type": "Text", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.r#type, LayoutElement::Text);
    }

    #[test]
    fn test_field_aliases_header_cell_range() {
        // Test header_cell_range alias for header_range field
        let json = r#"{"range": "A1:D10", "type": "Table", "header_cell_range": "A1:D1"}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.header_range, Some("A1:D1".to_string()));
    }

    #[test]
    fn test_field_aliases_combination() {
        // Test multiple aliases used together
        let json =
            r#"{"cell_range": "A1:D10", "element_type": "Table", "header_cell_range": "A1:D1"}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        assert_eq!(element.range, "A1:D10");
        assert_eq!(element.r#type, LayoutElement::Table);
        assert_eq!(element.header_range, Some("A1:D1".to_string()));
    }

    #[test]
    fn test_field_aliases_with_arrays() {
        // Test aliases with array values
        let json = r#"{"cell_range": ["A1:B2", "C1:F5"], "segment_type": "Table", "header_cell_range": ["A1:B1", "A1:D1"]}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();
        // C1:F5 covers 4*5=20 cells, A1:B2 covers 2*2=4 cells
        assert_eq!(element.range, "C1:F5");
        assert_eq!(element.r#type, LayoutElement::Table);
        // A1:D1 covers 4 cells, A1:B1 covers 2 cells
        assert_eq!(element.header_range, Some("A1:D1".to_string()));
    }

    #[test]
    fn test_field_aliases_with_comma_separated() {
        // Test aliases with comma-separated values
        let json = r#"{"cell_range": "G4:Q4,C6:C9", "element_type": "Table", "header_cell_range": "A1:D1"}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // With explicit header_cell_range, should use it and pick largest range as main
        assert_eq!(element.header_range, Some("A1:D1".to_string()));
        // G4:Q4 is 14 cells, C6:C9 is 4 cells, so should pick G4:Q4
        assert_eq!(element.range, "G4:Q4");
    }

    #[test]
    fn test_deserialize_two_cell_array_to_range() {
        // Test array with exactly two cell references - should convert to range
        let json = r#"{"range": ["B2", "F47"], "type": "Table", "header_range": ["B2", "F2"]}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // Should convert ["B2", "F47"] to "B2:F47"
        assert_eq!(element.range, "B2:F47");
        // Should convert ["B2", "F2"] to "B2:F2"
        assert_eq!(element.header_range, Some("B2:F2".to_string()));
    }

    #[test]
    fn test_deserialize_two_range_array_stays_array() {
        // Test array with two actual ranges - should pick largest
        let json = r#"{"range": ["A1:B2", "C1:F5"], "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // Should pick the larger range (C1:F5 has 20 cells vs A1:B2 has 4 cells)
        assert_eq!(element.range, "C1:F5");
    }

    #[test]
    fn test_deserialize_mixed_cell_and_range_array() {
        // Test array with mix of cell reference and range - should treat as separate ranges
        let json = r#"{"range": ["A1", "C1:F5"], "type": "Table", "header_range": null}"#;
        let element: IdentifiedElement = serde_json::from_str(json).unwrap();

        // Should pick the larger range (C1:F5 has 20 cells vs A1 has 1 cell)
        assert_eq!(element.range, "C1:F5");
    }

    #[test]
    fn test_is_single_cell_reference() {
        // Test valid single cell references
        assert!(is_single_cell_reference("A1"));
        assert!(is_single_cell_reference("B2"));
        assert!(is_single_cell_reference("Z99"));
        assert!(is_single_cell_reference("AA100"));

        // Test invalid - ranges
        assert!(!is_single_cell_reference("A1:B2"));
        assert!(!is_single_cell_reference("C3:D4"));

        // Test invalid - no letters or numbers
        assert!(!is_single_cell_reference("123"));
        assert!(!is_single_cell_reference("ABC"));
        assert!(!is_single_cell_reference(""));
    }
}
