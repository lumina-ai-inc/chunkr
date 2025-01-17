use bytes::BytesMut;
use postgres_types::{FromSql, IsNull, ToSql, Type};
use serde::{Deserialize, Serialize};
use serde_json;
use std::error::Error;
use utoipa::ToSchema;

#[derive(Debug, Deserialize, ToSchema)]
pub struct ExtractionRequest {
    pub json_schema: JsonSchema,
    pub contents: Vec<String>,
    pub content_type: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ExtractionResponse {
    pub extracted_json: ExtractedJson,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// The JSON schema to be used for structured extraction.
pub struct JsonSchema {
    /// The title of the JSON schema. This can be used to identify the schema.
    pub title: String,
    /// The properties of the JSON schema. Each property is a field to be extracted from the document.
    pub properties: Vec<Property>,
    #[serde(alias = "type", skip_serializing_if = "Option::is_none")]
    #[deprecated]
    /// The type of the JSON schema.
    pub schema_type: Option<String>,
}

impl std::str::FromStr for JsonSchema {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// A property of the JSON schema.
pub struct Property {
    /// The identifier for the property in the extracted data.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated]
    /// A human-readable title for the property. [deprecated]
    pub title: Option<String>,
    #[serde(alias = "type")]
    /// The data type of the property
    pub prop_type: String,
    /// A description of what the property represents.
    /// This is optional and can be used increase the accuracy of the extraction.
    /// Available for string, int, float, bool, list, object.
    pub description: Option<String>,
    /// The default value for the property if no data is extracted.
    pub default: Option<String>,
}

#[derive(Debug)]
pub struct Field {
    pub name: String,
    pub description: String,
    pub field_type: String,
    pub default: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// Structured data extracted from the document using an LLM. It adheres to the JSON schema provided.
pub struct ExtractedJson {
    /// The title of the extracted JSON.
    pub title: String,
    /// The extracted fields. Each field is a key in the json schema provided.
    pub extracted_fields: Vec<ExtractedField>,
    #[serde(alias = "type", skip_serializing_if = "Option::is_none")]
    #[deprecated]
    /// The type of the extracted JSON.
    pub schema_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct ExtractedField {
    /// The identifier for the field in the extracted data.
    pub name: String,
    /// The data type of the field
    pub field_type: String,
    /// The value of the field extracted from the document.
    #[serde(
        serialize_with = "serialize_value",
        deserialize_with = "deserialize_value"
    )]
    pub value: serde_json::Value,
}

impl ToSql for ExtractedField {
    fn to_sql(
        &self,
        ty: &Type,
        out: &mut BytesMut,
    ) -> Result<IsNull, Box<dyn Error + Sync + Send>> {
        let json_str = serde_json::to_string(&self.value)?;
        json_str.to_sql(ty, out)
    }

    fn accepts(ty: &Type) -> bool {
        <String as ToSql>::accepts(ty)
    }

    fn to_sql_checked(
        &self,
        ty: &Type,
        out: &mut BytesMut,
    ) -> Result<IsNull, Box<dyn Error + Sync + Send>> {
        let json_str = serde_json::to_string(&self.value)?;
        json_str.to_sql_checked(ty, out)
    }
}

impl<'a> FromSql<'a> for ExtractedField {
    fn from_sql(ty: &Type, raw: &'a [u8]) -> Result<Self, Box<dyn Error + Sync + Send>> {
        let json_str = <String as FromSql>::from_sql(ty, raw)?;
        let value = serde_json::from_str(&json_str)?;
        Ok(ExtractedField {
            name: String::new(),
            field_type: String::new(),
            value,
        })
    }

    fn accepts(ty: &Type) -> bool {
        <String as FromSql>::accepts(ty)
    }
}

fn serialize_value<S>(value: &serde_json::Value, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    value.serialize(serializer)
}

fn deserialize_value<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde_json::Value::deserialize(deserializer)
}

impl JsonSchema {
    pub fn to_fields(&self) -> Vec<Field> {
        self.properties
            .iter()
            .map(|prop| Field {
                name: prop.name.clone(),
                description: prop.description.clone().unwrap_or_default(),
                field_type: prop.prop_type.clone(),
                default: prop.default.clone(),
            })
            .collect()
    }
}
