use crate::models::chunkr::output::Segment;
use std::error::Error;
use std::sync::Arc;
use tempfile::NamedTempFile;

pub async fn perform_segmentation(
    temp_file: Arc<NamedTempFile>,
) -> Result<(Vec<Segment>), Box<dyn Error>> {
    unimplemented!()
}
