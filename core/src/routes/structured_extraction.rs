use crate::models::chunkr::auth::UserInfo;
use crate::models::chunkr::structured_extraction::{ExtractionRequest, ExtractionResponse};
use actix_web::{web, Error, HttpResponse};
use crate::utils::routes::structured_extraction::handle_structured_extraction;

/// Extract structured data from a document
///
/// Extracts structured data from a document according to a provided JSON schema.
/// The schema defines the fields to extract and their types.
#[utoipa::path(
    post,
    path = "/structured_extract",
    context_path = "/api/v1",
    tag = "Structured Extraction",
    request_body = ExtractionRequest,
    responses(
        (status = 200, description = "Successfully extracted structured data", body = ExtractionResponse),
        (status = 500, description = "Internal server error during extraction", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
pub async fn handle_structured_extraction_route(
    user_info: web::ReqData<UserInfo>,
    req: web::Json<ExtractionRequest>,
) -> Result<HttpResponse, Error> {
    let _user_info = user_info.into_inner();
    
    match handle_structured_extraction(req.into_inner()).await {
        Ok(response) => Ok(HttpResponse::Ok().json(response)),
        Err(e) => Ok(HttpResponse::InternalServerError().json(format!("Error: {}", e))),
    }
}
