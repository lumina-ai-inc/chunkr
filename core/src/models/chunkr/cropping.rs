use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Display,
    EnumString,
    Eq,
    FromSql,
    PartialEq,
    Serialize,
    ToSchema,
    ToSql,
)]
/// Controls the cropping strategy for an item (e.g. segment, chunk, etc.)
/// - `All` crops all images in the item
/// - `Auto` crops images only if required for post-processing
pub enum CroppingStrategy {
    All,
    #[default]
    Auto,
}

#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Display,
    EnumString,
    Eq,
    FromSql,
    PartialEq,
    Serialize,
    ToSchema,
    ToSql,
)]
/// Controls the cropping strategy for an item (e.g. segment, chunk, etc.)
/// - `All` crops all images in the item
/// - `Auto` crops images only if required for post-processing
pub enum PictureCroppingStrategy {
    #[default]
    All,
    Auto,
}
