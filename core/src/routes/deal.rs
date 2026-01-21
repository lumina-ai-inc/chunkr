use actix_multipart::form::{MultipartForm, tempfile::TempFile, text::Text};
use actix_web::{web, HttpResponse, Result};
use chrono::Utc;
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::models::auth::UserInfo;
use crate::models::deal::{CreateDealRequest, Deal, DealResponse, NewDeal, UpdateDeal};
use crate::models::document::{Document, DocumentResponse, NewDocument, UpdateDocument};
use crate::models::fact::{
    ApproveFactsRequest, Fact, FactResponse, FactType, NewFact, UpdateFact, UpdateFactValueRequest,
};
use crate::services::underwriting::{calculate_underwriting, UnderwritingInput, UnderwritingResult};
use crate::utils::clients::get_pg_client;

// POST /api/v1/deals - Create new deal
pub async fn create_deal_route(
    user_info: web::ReqData<UserInfo>,
    req: web::Json<CreateDealRequest>,
) -> Result<HttpResponse> {
    let user_id = user_info.user_id.clone();
    let deal_id = Uuid::new_v4().to_string();

    let new_deal = NewDeal {
        deal_id: deal_id.clone(),
        user_id,
        deal_name: req.deal_name.clone(),
        status: "draft".to_string(),
        metadata: None,
    };

    let client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let row = client.query_one(
        "INSERT INTO deals (deal_id, user_id, deal_name, status, metadata, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, NOW(), NOW()) RETURNING deal_id, user_id, deal_name, status, metadata, created_at, updated_at",
        &[&new_deal.deal_id, &new_deal.user_id, &new_deal.deal_name, &new_deal.status, &new_deal.metadata]
    ).await.map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Failed to create deal")
    })?;

    let response = DealResponse {
        deal_id: row.get("deal_id"),
        user_id: row.get("user_id"),
        deal_name: row.get("deal_name"),
        status: row.get("status"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
        metadata: row.get("metadata"),
        document_count: Some(0),
        fact_count: Some(0),
    };
    
    Ok(HttpResponse::Ok().json(response))
}

// GET /api/v1/deals - List user's deals
pub async fn get_deals_route(user_info: web::ReqData<UserInfo>) -> Result<HttpResponse> {
    let user_id = user_info.user_id.clone();
    
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let results = web::block(move || {
        use crate::data::schema::deals::dsl::*;
        use crate::data::schema::documents;
        use crate::data::schema::facts;
        
        let deal_list: Vec<Deal> = deals
            .filter(user_id.eq(&user_id))
            .order(created_at.desc())
            .load::<Deal>(&mut client)?;
        
        // Get counts for each deal
        let mut responses = Vec::new();
        for deal in deal_list {
            let doc_count: i64 = documents::table
                .filter(documents::deal_id.eq(&deal.deal_id))
                .count()
                .get_result(&mut client)?;
            
            let fact_count: i64 = facts::table
                .filter(facts::deal_id.eq(&deal.deal_id))
                .count()
                .get_result(&mut client)?;
            
            let mut response: DealResponse = deal.into();
            response.document_count = Some(doc_count);
            response.fact_count = Some(fact_count);
            responses.push(response);
        }
        
        Ok::<Vec<DealResponse>, diesel::result::Error>(responses)
    })
    .await
    .map_err(|e| {
        eprintln!("Error fetching deals: {:?}", e);
        actix_web::error::ErrorInternalServerError("Failed to fetch deals")
    })?
    .map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?;

    Ok(HttpResponse::Ok().json(results))
}

// GET /api/v1/deals/:deal_id - Get deal details
pub async fn get_deal_route(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
) -> Result<HttpResponse> {
    let deal_id = path.into_inner();
    let user_id = user_info.user_id.clone();
    
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let result = web::block(move || {
        use crate::data::schema::deals::dsl::*;
        use crate::data::schema::documents;
        use crate::data::schema::facts;
        
        let deal = deals
            .filter(deal_id.eq(&deal_id))
            .filter(user_id.eq(&user_id))
            .first::<Deal>(&mut client)?;
        
        let doc_count: i64 = documents::table
            .filter(documents::deal_id.eq(&deal.deal_id))
            .count()
            .get_result(&mut client)?;
        
        let fact_count: i64 = facts::table
            .filter(facts::deal_id.eq(&deal.deal_id))
            .count()
            .get_result(&mut client)?;
        
        let mut response: DealResponse = deal.into();
        response.document_count = Some(doc_count);
        response.fact_count = Some(fact_count);
        
        Ok::<DealResponse, diesel::result::Error>(response)
    })
    .await
    .map_err(|e| {
        eprintln!("Error fetching deal: {:?}", e);
        actix_web::error::ErrorNotFound("Deal not found")
    })?
    .map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?;

    Ok(HttpResponse::Ok().json(result))
}

#[derive(Debug, MultipartForm)]
pub struct UploadDocumentsForm {
    #[multipart(limit = "1 GB")]
    pub files: Vec<TempFile>,
    pub document_type: Text<String>,
}

// POST /api/v1/deals/:deal_id/documents - Upload documents to deal
pub async fn upload_deal_documents(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
    MultipartForm(form): MultipartForm<UploadDocumentsForm>,
) -> Result<HttpResponse> {
    let deal_id = path.into_inner();
    let user_id = user_info.user_id.clone();
    let doc_type = form.document_type.0;
    
    // Verify deal ownership
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let deal_exists = web::block({
        let deal_id_clone = deal_id.clone();
        let user_id_clone = user_id.clone();
        let mut client_clone = client.clone();
        move || {
            use crate::data::schema::deals::dsl::*;
            deals
                .filter(deal_id.eq(&deal_id_clone))
                .filter(user_id.eq(&user_id_clone))
                .first::<Deal>(&mut client_clone)
        }
    })
    .await
    .map_err(|e| {
        eprintln!("Error verifying deal: {:?}", e);
        actix_web::error::ErrorNotFound("Deal not found")
    })?;

    if deal_exists.is_err() {
        return Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Deal not found or access denied"
        })));
    }

    // Create document records for each file
    let mut document_responses = Vec::new();
    
    for file in form.files {
        let document_id = Uuid::new_v4().to_string();
        let file_name = file.file_name.unwrap_or_else(|| "unknown".to_string());
        
        // Upload file to S3
        let worker_config = crate::configs::worker_config::Config::from_env()
            .map_err(|e| {
                eprintln!("Failed to load worker config: {:?}", e);
                actix_web::error::ErrorInternalServerError("Configuration error")
            })?;
        
        let bucket_name = worker_config.s3_bucket;
        let s3_key = format!("{}/{}/{}", user_id, deal_id.clone(), file_name);
        let s3_location = format!("s3://{}/{}", bucket_name, s3_key);
        
        // Save temp file content
        let file_data = file.data;
        let temp_file = web::block(move || {
            let mut temp = tempfile::NamedTempFile::new()?;
            std::io::Write::write_all(&mut temp, &file_data)?;
            Ok::<tempfile::NamedTempFile, std::io::Error>(temp)
        })
        .await
        .map_err(|e| {
            eprintln!("Error creating temp file: {:?}", e);
            actix_web::error::ErrorInternalServerError("Failed to process file")
        })?
        .map_err(|e| {
            eprintln!("I/O error: {:?}", e);
            actix_web::error::ErrorInternalServerError("Failed to save file")
        })?;
        
        // Upload to S3
        crate::utils::storage::services::upload_to_s3(&s3_location, temp_file.path())
            .await
            .map_err(|e| {
                eprintln!("S3 upload error: {:?}", e);
                actix_web::error::ErrorInternalServerError("Failed to upload file")
            })?;
        
        let new_doc = NewDocument {
            document_id: document_id.clone(),
            deal_id: deal_id.clone(),
            file_name: file_name.clone(),
            document_type: doc_type.clone(),
            status: "processing".to_string(),
            storage_location: Some(s3_location.clone()),
            page_count: None,
            ocr_output: None,
        };

        let doc = web::block({
            let mut client_clone = client.clone();
            move || {
                use crate::data::schema::documents::dsl::*;
                diesel::insert_into(documents)
                    .values(&new_doc)
                    .get_result::<Document>(&mut client_clone)
            }
        })
        .await
        .map_err(|e| {
            eprintln!("Error creating document: {:?}", e);
            actix_web::error::ErrorInternalServerError("Failed to create document")
        })?
        .map_err(|e| {
            eprintln!("Database error: {:?}", e);
            actix_web::error::ErrorInternalServerError("Database error")
        })?;

        // Queue document processing task
        let processing_message = serde_json::json!({
            "document_id": document_id.clone(),
            "deal_id": deal_id.clone(),
            "user_id": user_id.clone(),
            "s3_location": s3_location,
            "file_name": file_name,
            "document_type": doc_type.clone(),
        });
        
        let pool = crate::utils::clients::get_redis_pool();
        let mut conn = pool.get().await.map_err(|e| {
            eprintln!("Redis connection error: {:?}", e);
            actix_web::error::ErrorInternalServerError("Queue error")
        })?;
        
        deadpool_redis::redis::cmd("RPUSH")
            .arg("deal_documents")
            .arg(serde_json::to_string(&processing_message).unwrap())
            .query_async::<i64>(&mut conn)
            .await
            .map_err(|e| {
                eprintln!("Redis queue error: {:?}", e);
                actix_web::error::ErrorInternalServerError("Failed to queue processing")
            })?;

        document_responses.push(DocumentResponse::from(doc));
    }

    Ok(HttpResponse::Ok().json(document_responses))
}

// GET /api/v1/deals/:deal_id/documents - List deal documents
pub async fn get_deal_documents(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
) -> Result<HttpResponse> {
    let deal_id = path.into_inner();
    let user_id = user_info.user_id.clone();
    
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let results = web::block(move || {
        use crate::data::schema::deals::dsl::*;
        use crate::data::schema::documents;
        use crate::data::schema::facts;
        
        // Verify deal ownership
        deals
            .filter(deal_id.eq(&deal_id))
            .filter(user_id.eq(&user_id))
            .first::<Deal>(&mut client)?;
        
        // Get documents
        let docs: Vec<Document> = documents::table
            .filter(documents::deal_id.eq(&deal_id))
            .order(documents::created_at.desc())
            .load::<Document>(&mut client)?;
        
        // Get fact counts for each document
        let mut responses = Vec::new();
        for doc in docs {
            let fact_count: i64 = facts::table
                .filter(facts::document_id.eq(&doc.document_id))
                .count()
                .get_result(&mut client)?;
            
            let mut response: DocumentResponse = doc.into();
            response.fact_count = Some(fact_count);
            responses.push(response);
        }
        
        Ok::<Vec<DocumentResponse>, diesel::result::Error>(responses)
    })
    .await
    .map_err(|e| {
        eprintln!("Error fetching documents: {:?}", e);
        actix_web::error::ErrorNotFound("Documents not found")
    })?
    .map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?;

    Ok(HttpResponse::Ok().json(results))
}

// POST /api/v1/deals/:deal_id/facts - Create a new fact
pub async fn create_fact_route(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
    req: web::Json<serde_json::Value>,
) -> Result<HttpResponse> {
    let deal_id = path.into_inner();
    let user_id = user_info.user_id.clone();
    
    // Extract fields from request
    let label = req.get("label")
        .and_then(|v| v.as_str())
        .ok_or_else(|| actix_web::error::ErrorBadRequest("Missing 'label' field"))?;
    let value = req.get("value")
        .and_then(|v| v.as_str())
        .ok_or_else(|| actix_web::error::ErrorBadRequest("Missing 'value' field"))?;
    let unit = req.get("unit").and_then(|v| v.as_str());
    let fact_type = req.get("fact_type")
        .and_then(|v| v.as_str())
        .unwrap_or("financial");
    
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let result = web::block(move || {
        use crate::data::schema::deals::dsl::*;
        use crate::data::schema::documents;
        use crate::data::schema::facts;
        
        // Verify deal ownership
        deals
            .filter(deal_id.eq(&deal_id))
            .filter(user_id.eq(&user_id))
            .first::<Deal>(&mut client)?;
        
        // Get first document for this deal, or create a placeholder document_id
        let document_id: Option<String> = documents::table
            .filter(documents::deal_id.eq(&deal_id))
            .select(documents::document_id)
            .first::<String>(&mut client)
            .optional()?;
        
        let document_id = document_id.unwrap_or_else(|| format!("doc-{}-placeholder", deal_id));
        
        // Create new fact
        let fact_id = Uuid::new_v4().to_string();
        let source_citation = serde_json::json!({
            "document": "Manual Entry",
            "page": 1
        });
        
        let new_fact = NewFact {
            fact_id: fact_id.clone(),
            document_id,
            deal_id: deal_id.clone(),
            fact_type: fact_type.to_string(),
            label: label.to_string(),
            value: value.to_string(),
            unit: unit.map(|s| s.to_string()),
            source_citation,
            status: "pending_approval".to_string(),
            confidence_score: Some(1.0), // User-entered facts have full confidence
        };
        
        diesel::insert_into(facts::table)
            .values(&new_fact)
            .get_result::<Fact>(&mut client)
    })
    .await
    .map_err(|e| {
        eprintln!("Error creating fact: {:?}", e);
        actix_web::error::ErrorBadRequest("Cannot create fact")
    })?
    .map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?;

    let response = result.to_response().map_err(|e| {
        eprintln!("Serialization error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Serialization error")
    })?;

    Ok(HttpResponse::Created().json(response))
}

// GET /api/v1/deals/:deal_id/facts - Get extracted facts
pub async fn get_deal_facts(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
) -> Result<HttpResponse> {
    let deal_id = path.into_inner();
    let user_id = user_info.user_id.clone();
    
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let results = web::block(move || {
        use crate::data::schema::deals::dsl::*;
        use crate::data::schema::facts;
        
        // Verify deal ownership
        deals
            .filter(deal_id.eq(&deal_id))
            .filter(user_id.eq(&user_id))
            .first::<Deal>(&mut client)?;
        
        // Get facts
        let fact_list: Vec<Fact> = facts::table
            .filter(facts::deal_id.eq(&deal_id))
            .order(facts::created_at.asc())
            .load::<Fact>(&mut client)?;
        
        // Convert to responses
        let responses: Result<Vec<FactResponse>, serde_json::Error> = fact_list
            .into_iter()
            .map(|f| f.to_response())
            .collect();
        
        responses.map_err(|_| diesel::result::Error::DeserializationError(Box::new(
            std::io::Error::new(std::io::ErrorKind::Other, "Failed to deserialize facts")
        )))
    })
    .await
    .map_err(|e| {
        eprintln!("Error fetching facts: {:?}", e);
        actix_web::error::ErrorNotFound("Facts not found")
    })?
    .map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?;

    Ok(HttpResponse::Ok().json(results))
}

// PATCH /api/v1/deals/:deal_id/facts/:fact_id - Update fact value
pub async fn update_fact_route(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<(String, String)>,
    req: web::Json<UpdateFactValueRequest>,
) -> Result<HttpResponse> {
    let (deal_id, fact_id) = path.into_inner();
    let user_id = user_info.user_id.clone();
    
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let result = web::block(move || {
        use crate::data::schema::deals::dsl::*;
        use crate::data::schema::facts;
        
        // Verify deal ownership
        deals
            .filter(deal_id.eq(&deal_id))
            .filter(user_id.eq(&user_id))
            .first::<Deal>(&mut client)?;
        
        // Check if fact is locked
        let fact: Fact = facts::table
            .filter(facts::fact_id.eq(&fact_id))
            .filter(facts::deal_id.eq(&deal_id))
            .first::<Fact>(&mut client)?;
        
        if fact.locked {
            return Err(diesel::result::Error::DatabaseError(
                diesel::result::DatabaseErrorKind::Unknown,
                Box::new("Cannot update locked fact".to_string())
            ));
        }
        
        // Update fact
        let update = UpdateFact {
            value: Some(req.value.clone()),
            unit: req.unit.clone(),
            status: None,
            approved_at: None,
            approved_by: None,
            locked: None,
        };
        
        diesel::update(facts::table.filter(facts::fact_id.eq(&fact_id)))
            .set(&update)
            .get_result::<Fact>(&mut client)
    })
    .await
    .map_err(|e| {
        eprintln!("Error updating fact: {:?}", e);
        actix_web::error::ErrorBadRequest("Cannot update fact")
    })?
    .map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?;

    let response = result.to_response().map_err(|e| {
        eprintln!("Serialization error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Serialization error")
    })?;

    Ok(HttpResponse::Ok().json(response))
}

// POST /api/v1/deals/:deal_id/facts/approve - Approve facts (lock them)
pub async fn approve_facts_route(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
    req: web::Json<ApproveFactsRequest>,
) -> Result<HttpResponse> {
    let deal_id = path.into_inner();
    let user_id = user_info.user_id.clone();
    let fact_ids = req.fact_ids.clone();
    
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let results = web::block(move || {
        use crate::data::schema::deals::dsl::*;
        use crate::data::schema::facts;
        
        // Verify deal ownership
        deals
            .filter(deal_id.eq(&deal_id))
            .filter(user_id.eq(&user_id))
            .first::<Deal>(&mut client)?;
        
        // Approve and lock facts
        let update = UpdateFact {
            value: None,
            unit: None,
            status: Some("approved".to_string()),
            approved_at: Some(Utc::now()),
            approved_by: Some(user_id.clone()),
            locked: Some(true),
        };
        
        diesel::update(
            facts::table
                .filter(facts::fact_id.eq_any(&fact_ids))
                .filter(facts::deal_id.eq(&deal_id))
        )
        .set(&update)
        .execute(&mut client)
    })
    .await
    .map_err(|e| {
        eprintln!("Error approving facts: {:?}", e);
        actix_web::error::ErrorBadRequest("Cannot approve facts")
    })?
    .map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "message": "Facts approved successfully",
        "count": results
    })))
}

// POST /api/v1/deals/:deal_id/facts/reset - Reset all facts to editable
pub async fn reset_facts_route(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
) -> Result<HttpResponse> {
    let deal_id = path.into_inner();
    let user_id = user_info.user_id.clone();
    
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let results = web::block(move || {
        use crate::data::schema::deals::dsl::*;
        use crate::data::schema::facts;
        
        // Verify deal ownership
        deals
            .filter(deal_id.eq(&deal_id))
            .filter(user_id.eq(&user_id))
            .first::<Deal>(&mut client)?;
        
        // Unlock and reset facts
        let update = UpdateFact {
            value: None,
            unit: None,
            status: Some("pending_approval".to_string()),
            approved_at: None,
            approved_by: None,
            locked: Some(false),
        };
        
        diesel::update(facts::table.filter(facts::deal_id.eq(&deal_id)))
            .set(&update)
            .execute(&mut client)
    })
    .await
    .map_err(|e| {
        eprintln!("Error resetting facts: {:?}", e);
        actix_web::error::ErrorBadRequest("Cannot reset facts")
    })?
    .map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "message": "Facts reset successfully",
        "count": results
    })))
}

// POST /api/v1/deals/:deal_id/underwrite - Run underwriting calculations
pub async fn calculate_underwriting_route(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
) -> Result<HttpResponse> {
    let deal_id = path.into_inner();
    let user_id = user_info.user_id.clone();
    
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let result = web::block(move || {
        use crate::data::schema::deals::dsl::*;
        use crate::data::schema::facts;
        
        // Verify deal ownership
        deals
            .filter(deal_id.eq(&deal_id))
            .filter(user_id.eq(&user_id))
            .first::<Deal>(&mut client)?;
        
        // Get approved facts for this deal
        let fact_list: Vec<Fact> = facts::table
            .filter(facts::deal_id.eq(&deal_id))
            .filter(facts::status.eq("approved"))
            .filter(facts::locked.eq(true))
            .load::<Fact>(&mut client)?;
        
        // Extract values from facts by type
        let mut unit_count: Option<i32> = None;
        let mut occupancy_rate: Option<f64> = None;
        let mut gross_scheduled_rent: Option<f64> = None;
        let mut collected_rent: Option<f64> = None;
        let mut operating_expenses: Option<f64> = None;
        let mut debt_service: Option<f64> = None;
        let mut property_value: Option<f64> = None;
        let mut mortgage_balance: Option<f64> = None;
        let mut interest_rate: Option<f64> = None;
        
        for fact in fact_list {
            let parsed_value = fact.value.parse::<f64>().ok();
            
            match fact.fact_type.as_str() {
                "unit_count" => {
                    unit_count = fact.value.parse::<i32>().ok();
                },
                "occupancy_rate" => {
                    occupancy_rate = parsed_value;
                },
                "gross_scheduled_rent" => {
                    gross_scheduled_rent = parsed_value;
                },
                "collected_rent" => {
                    collected_rent = parsed_value;
                },
                "operating_expenses" => {
                    operating_expenses = parsed_value;
                },
                "debt_service" => {
                    debt_service = parsed_value;
                },
                "property_value" => {
                    property_value = parsed_value;
                },
                "mortgage_balance" => {
                    mortgage_balance = parsed_value;
                },
                "interest_rate" => {
                    interest_rate = parsed_value;
                },
                _ => {}
            }
        }
        
        // Validate required fields
        if collected_rent.is_none() || operating_expenses.is_none() {
            return Err(diesel::result::Error::DatabaseError(
                diesel::result::DatabaseErrorKind::Unknown,
                Box::new("Missing required facts: collected_rent and operating_expenses".to_string())
            ));
        }
        
        let input = UnderwritingInput {
            unit_count,
            occupancy_rate,
            gross_scheduled_rent,
            collected_rent: collected_rent.unwrap(),
            operating_expenses: operating_expenses.unwrap(),
            debt_service,
            property_value,
            mortgage_balance,
            interest_rate,
        };
        
        Ok::<UnderwritingResult, diesel::result::Error>(calculate_underwriting(input))
    })
    .await
    .map_err(|e| {
        eprintln!("Error calculating underwriting: {:?}", e);
        actix_web::error::ErrorBadRequest("Cannot calculate underwriting")
    })?
    .map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?;

    Ok(HttpResponse::Ok().json(result))
}

