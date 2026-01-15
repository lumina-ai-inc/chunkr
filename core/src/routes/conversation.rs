use actix_web::{web, HttpResponse, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

// use crate::agents::chat_orchestrator::ChatOrchestratorAgent; // Temporarily disabled
use crate::models::auth::UserInfo;
use crate::utils::clients::get_pg_client;

// POST /api/v1/conversations - Create new conversation
#[derive(Debug, Deserialize)]
pub struct CreateConversationRequest {
    pub deal_id: Option<String>,
    pub title: Option<String>,
}

pub async fn create_conversation(
    user_info: web::ReqData<UserInfo>,
    req: web::Json<CreateConversationRequest>,
) -> Result<HttpResponse> {
    let user_id = user_info.user_id.clone();
    let conversation_id = Uuid::new_v4().to_string();
    let now = Utc::now().naive_utc();

    let new_conversation = NewConversation {
        conversation_id: conversation_id.clone(),
        user_id,
        deal_id: req.deal_id.clone(),
        title: req.title.clone(),
        context: None,
        created_at: now,
        updated_at: now,
    };

    let client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    client.execute(
        "INSERT INTO conversations (conversation_id, user_id, deal_id, title, context, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7)",
        &[&new_conversation.conversation_id, &new_conversation.user_id, &new_conversation.deal_id, &new_conversation.title, &new_conversation.context, &new_conversation.created_at, &new_conversation.updated_at]
    ).await.map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Failed to create conversation")
    })?;

    Ok(HttpResponse::Ok().json(json!({
        "conversation_id": conversation_id,
        "deal_id": req.deal_id,
        "title": req.title,
    })))
}

// GET /api/v1/conversations - List user's conversations
pub async fn list_conversations(user_info: web::ReqData<UserInfo>) -> Result<HttpResponse> {
    let user_id = user_info.user_id.clone();

    let client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let rows = client.query(
        "SELECT conversation_id, user_id, deal_id, title, context, created_at, updated_at FROM conversations WHERE user_id = $1 ORDER BY updated_at DESC",
        &[&user_id]
    ).await.map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Failed to fetch conversations")
    })?;

    let results: Vec<Conversation> = rows.iter().map(|row| Conversation {
        conversation_id: row.get("conversation_id"),
        user_id: row.get("user_id"),
        deal_id: row.get("deal_id"),
        title: row.get("title"),
        context: row.get("context"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }).collect();

    Ok(HttpResponse::Ok().json(results))
}

// GET /api/v1/conversations/:conversation_id - Get conversation details
pub async fn get_conversation(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
) -> Result<HttpResponse> {
    let user_id = user_info.user_id.clone();
    let conversation_id = path.into_inner();

    let client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    // Get conversation
    let conv_row = client.query_opt(
        "SELECT conversation_id, user_id, deal_id, title, context, created_at, updated_at FROM conversations WHERE conversation_id = $1 AND user_id = $2",
        &[&conversation_id, &user_id]
    ).await.map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?.ok_or_else(|| actix_web::error::ErrorNotFound("Conversation not found"))?;

    let conversation = Conversation {
        conversation_id: conv_row.get("conversation_id"),
        user_id: conv_row.get("user_id"),
        deal_id: conv_row.get("deal_id"),
        title: conv_row.get("title"),
        context: conv_row.get("context"),
        created_at: conv_row.get("created_at"),
        updated_at: conv_row.get("updated_at"),
    };

    // Get messages
    let msg_rows = client.query(
        "SELECT message_id, conversation_id, role, content, metadata, embedding_id, created_at FROM messages WHERE conversation_id = $1 ORDER BY created_at ASC",
        &[&conversation_id]
    ).await.map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Failed to fetch messages")
    })?;

    let msgs: Vec<Message> = msg_rows.iter().map(|row| Message {
        message_id: row.get("message_id"),
        conversation_id: row.get("conversation_id"),
        role: row.get("role"),
        content: row.get("content"),
        metadata: row.get("metadata"),
        embedding_id: row.get("embedding_id"),
        created_at: row.get("created_at"),
    }).collect();

    Ok(HttpResponse::Ok().json(json!({
        "conversation": conversation,
        "messages": msgs,
    })))
}

// POST /api/v1/conversations/:conversation_id/messages - Send message
#[derive(Debug, Deserialize)]
pub struct SendMessageRequest {
    pub content: String,
}

pub async fn send_message(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
    _req: web::Json<SendMessageRequest>,
) -> Result<HttpResponse> {
    let user_id = user_info.user_id.clone();
    let conversation_id = path.into_inner();

    // Verify conversation belongs to user and get deal_id
    let client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let conv_row = client.query_opt(
        "SELECT conversation_id, user_id, deal_id, title, context, created_at, updated_at FROM conversations WHERE conversation_id = $1 AND user_id = $2",
        &[&conversation_id, &user_id]
    ).await.map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database error")
    })?.ok_or_else(|| actix_web::error::ErrorNotFound("Conversation not found"))?;

    let _conversation = Conversation {
        conversation_id: conv_row.get("conversation_id"),
        user_id: conv_row.get("user_id"),
        deal_id: conv_row.get("deal_id"),
        title: conv_row.get("title"),
        context: conv_row.get("context"),
        created_at: conv_row.get("created_at"),
        updated_at: conv_row.get("updated_at"),
    };

    // TODO: Re-enable chat orchestrator after converting to tokio-postgres
    // For now, return a placeholder response
    let result = serde_json::json!({
        "response": "Chat orchestrator temporarily disabled during migration. Will be re-enabled soon.",
        "intent": "placeholder",
        "action": null,
        "context_used": 0
    });

    // Update conversation updated_at
    client.execute(
        "UPDATE conversations SET updated_at = $1 WHERE conversation_id = $2",
        &[&Utc::now().naive_utc(), &conversation_id]
    ).await.map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Failed to update conversation")
    })?;

    Ok(HttpResponse::Ok().json(result))
}

// DELETE /api/v1/conversations/:conversation_id - Delete conversation
pub async fn delete_conversation(
    user_info: web::ReqData<UserInfo>,
    path: web::Path<String>,
) -> Result<HttpResponse> {
    let user_id = user_info.user_id.clone();
    let conversation_id = path.into_inner();

    let client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    // Delete conversation (messages will cascade)
    let deleted = client.execute(
        "DELETE FROM conversations WHERE conversation_id = $1 AND user_id = $2",
        &[&conversation_id, &user_id]
    ).await.map_err(|e| {
        eprintln!("Database error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Failed to delete conversation")
    })?;

    if deleted == 0 {
        return Err(actix_web::error::ErrorNotFound("Conversation not found"));
    }

    Ok(HttpResponse::Ok().json(json!({
        "success": true,
    })))
}

// POST /api/v1/conversations/public/message - Public chat (no auth required)
#[derive(Debug, Deserialize)]
pub struct PublicMessageRequest {
    pub content: String,
    pub session_id: Option<String>, // For tracking conversation without auth
}

pub async fn send_public_message(
    req: web::Json<PublicMessageRequest>,
) -> Result<HttpResponse> {
    let session_id = req.session_id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());

    // TODO: Re-enable chat orchestrator after converting to tokio-postgres
    // For now, return a placeholder response
    Ok(HttpResponse::Ok().json(json!({
        "response": "Public chat temporarily disabled during migration. Please check back soon!",
        "session_id": session_id,
        "intent": "placeholder",
    })))
}

// Database models
#[derive(Debug, Clone, Serialize)]
struct Conversation {
    conversation_id: String,
    user_id: String,
    deal_id: Option<String>,
    title: Option<String>,
    context: Option<serde_json::Value>,
    created_at: chrono::NaiveDateTime,
    updated_at: chrono::NaiveDateTime,
}

#[derive(Debug, Clone)]
struct NewConversation {
    conversation_id: String,
    user_id: String,
    deal_id: Option<String>,
    title: Option<String>,
    context: Option<serde_json::Value>,
    created_at: chrono::NaiveDateTime,
    updated_at: chrono::NaiveDateTime,
}

#[derive(Debug, Clone, Serialize)]
struct Message {
    message_id: String,
    conversation_id: String,
    role: String,
    content: String,
    metadata: Option<serde_json::Value>,
    embedding_id: Option<String>,
    created_at: chrono::NaiveDateTime,
}

