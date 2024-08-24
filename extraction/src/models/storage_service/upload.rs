use actix_multipart::form::{json::Json as MPJson, tempfile::TempFile, MultipartForm};
use serde::Deserialize;
use std::time::Duration;

#[derive(Debug, Deserialize)]
pub struct Metadata {
    pub location: String,
    #[serde(with = "humantime_serde")]
    pub expiration: Option<Duration>,
}

#[derive(Debug, MultipartForm)]
pub struct UploadForm {
    #[multipart(limit = "100MB")]
    pub file: TempFile,
    pub metadata: MPJson<Metadata>,
}
