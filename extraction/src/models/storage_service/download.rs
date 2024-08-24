use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize, Deserialize)]
pub struct DownloadPayload {
    pub location: String,
    #[serde(with = "humantime_serde")]
    pub expires_in: Option<Duration>,
}
