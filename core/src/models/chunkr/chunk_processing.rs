use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// Controls the setting for the chunking and post-processing of each chunk.
pub struct ChunkProcessing {
    #[serde(default = "default_target_length")]
    #[schema(value_type = i32, default = 512)]
    /// The target number of words in each chunk. If 0, each chunk will contain a single segment.
    pub target_length: i32,
}

impl Default for ChunkProcessing {
    fn default() -> Self {
        Self {
            target_length: default_target_length(),
        }
    }
}

pub fn default_target_length() -> i32 {
    512
}
