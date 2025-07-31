use crate::models::structured_extraction::{
    StructuredExtractionRequest, StructuredExtractionResponse,
};
use crate::utils::services::structured_extraction::perform_structured_extraction;
use actix_web::{web, Error, HttpResponse};

/// Extract structured data from a document
///
/// Extracts structured data from a document according to a provided JSON schema.
/// The schema defines the fields to extract and their types.
#[utoipa::path(
    post,
    path = "/structured_extract",
    context_path = "",
    tag = "Structured Extraction",
    request_body = StructuredExtractionRequest,
    responses(
        (status = 200, description = "Successfully extracted structured data", body = StructuredExtractionResponse),
        (status = 500, description = "Internal server error during extraction", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
pub async fn handle_structured_extraction_route(
    req: web::Json<StructuredExtractionRequest>,
) -> Result<HttpResponse, Error> {
    let structured_extraction_request = req.into_inner();
    match perform_structured_extraction(structured_extraction_request).await {
        Ok(response) => Ok(HttpResponse::Ok().json(response)),
        Err(e) => Ok(HttpResponse::InternalServerError().json(format!("Error: {}", e))),
    }
}
