use serde::{ Deserialize, Serialize };
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Opt {
    pub bucket: String,
    pub object: String,
    pub destination: PathBuf,
}
