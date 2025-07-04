use schemars::JsonSchema as SchemarsJsonSchema;
use serde::{Deserialize, Deserializer, Serialize};

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
            "image" | "picture" | "logo" | "figure" | "visual" => Ok(LayoutElement::Image),
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
    ranges.into_iter().max_by_key(|range| {
        match parse_range(range) {
            Ok(indices) => {
                let rows = (indices.end_row - indices.start_row + 1) as u64;
                let cols = (indices.end_col - indices.start_col + 1) as u64;
                rows * cols // Total number of cells
            }
            Err(_) => 0, // Invalid ranges get lowest priority
        }
    })
}

/// Custom deserializer for range fields that can handle both string and array inputs
/// If an array is provided, it selects the largest range (covering the most cells)
fn deserialize_range<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct RangeVisitor;

    impl<'de> Visitor<'de> for RangeVisitor {
        type Value = String;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or array of strings representing Excel ranges")
        }

        fn visit_str<E>(self, value: &str) -> Result<String, E>
        where
            E: de::Error,
        {
            Ok(value.to_string())
        }

        fn visit_string<E>(self, value: String) -> Result<String, E>
        where
            E: de::Error,
        {
            Ok(value)
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<String, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut ranges = Vec::new();

            while let Some(range) = seq.next_element::<String>()? {
                ranges.push(range);
            }

            select_largest_range(ranges).ok_or_else(|| de::Error::custom("Empty array of ranges"))
        }
    }

    deserializer.deserialize_any(RangeVisitor)
}

/// Custom deserializer for optional range fields that can handle both string and array inputs
/// If an array is provided, it selects the largest range (covering the most cells)
fn deserialize_optional_range<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct OptionalRangeVisitor;

    impl<'de> Visitor<'de> for OptionalRangeVisitor {
        type Value = Option<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, a string, or array of strings representing Excel ranges")
        }

        fn visit_none<E>(self) -> Result<Option<String>, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Option<String>, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_str<E>(self, value: &str) -> Result<Option<String>, E>
        where
            E: de::Error,
        {
            Ok(Some(value.to_string()))
        }

        fn visit_string<E>(self, value: String) -> Result<Option<String>, E>
        where
            E: de::Error,
        {
            Ok(Some(value))
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Option<String>, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut ranges = Vec::new();

            while let Some(range) = seq.next_element::<String>()? {
                ranges.push(range);
            }

            Ok(select_largest_range(ranges))
        }
    }

    deserializer.deserialize_any(OptionalRangeVisitor)
}

#[derive(Debug, Serialize, Deserialize, SchemarsJsonSchema, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
pub struct IdentifiedElement {
    /// The range of the element in Excel notation (e.g., "A1:D10")
    #[serde(deserialize_with = "deserialize_range")]
    pub range: String,
    /// The type of the element
    pub r#type: LayoutElement,
    /// The header range of the table if the element is a `Table` in excel notation (e.g., "A1:C1").
    #[serde(deserialize_with = "deserialize_optional_range", default)]
    pub header_range: Option<String>,
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
}
