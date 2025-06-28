use schemars::JsonSchema as SchemarsJsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, SchemarsJsonSchema, PartialEq, Clone)]
pub struct IdentifiedTable {
    /// The range of the table in Excel notation (e.g., "A1:D10")
    pub table_range: String,
    /// The range of the table header in Excel notation (e.g., "A1:D1")
    pub header_range: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, SchemarsJsonSchema, PartialEq, Clone)]
pub struct IdentifiedTables {
    /// List of identified tables with their ranges and header information
    pub tables: Vec<IdentifiedTable>,
}
