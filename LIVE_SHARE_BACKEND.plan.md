---
name: Live Share Feature - Backend & Database
overview: Implement backend API, database schema, and utilities for the live share feature. This includes database migrations, Rust models, API routes, short ID generation, and data persistence.
todos:
  - id: db_schema
    content: Create database migration for live_shares, share_views, and investor_interest tables
    status: pending
  - id: backend_models
    content: Create Rust models for LiveShare, ShareView, and InvestorInterest
    status: pending
    dependencies:
      - db_schema
  - id: short_id_generator
    content: Implement short ID generator utility (8-12 char alphanumeric)
    status: pending
  - id: backend_routes
    content: Implement backend API routes for live shares (create, list, get, public endpoints)
    status: pending
    dependencies:
      - backend_models
      - short_id_generator
  - id: register_routes
    content: Register live share routes in core/src/lib.rs
    status: pending
    dependencies:
      - backend_routes
---

# Live Share Feature - Backend & Database Implementation

## Overview

This plan covers all backend and database changes required for the live share feature, including database schema, API routes, models, and utilities.

## Architecture

```
Database Layer
    ↓
Rust Models (Serde, Validation)
    ↓
API Routes (Actix Web)
    ↓
Short ID Generator Utility
```

## Implementation Steps

### 1. Database Schema

**File**: `core/migrations/YYYY-MM-DD-HHMMSS_create_live_shares/up.sql`

Create three new tables:

```sql
-- Table: live_shares
CREATE TABLE live_shares (
    id TEXT PRIMARY KEY,
    deal_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    short_id TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    view_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (deal_id) REFERENCES deals(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Table: share_views
CREATE TABLE share_views (
    id TEXT PRIMARY KEY,
    live_share_id TEXT NOT NULL,
    viewed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address TEXT,
    user_agent TEXT,
    FOREIGN KEY (live_share_id) REFERENCES live_shares(id) ON DELETE CASCADE
);

-- Table: investor_interest
CREATE TABLE investor_interest (
    id TEXT PRIMARY KEY,
    live_share_id TEXT NOT NULL,
    name TEXT NOT NULL,
    amount DECIMAL(12,2),
    status TEXT NOT NULL CHECK (status IN ('Interested', 'Maybe', 'Passed')),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (live_share_id) REFERENCES live_shares(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX idx_live_shares_user_id ON live_shares(user_id);
CREATE INDEX idx_live_shares_deal_id ON live_shares(deal_id);
CREATE INDEX idx_live_shares_short_id ON live_shares(short_id);
CREATE INDEX idx_share_views_live_share_id ON share_views(live_share_id);
CREATE INDEX idx_investor_interest_live_share_id ON investor_interest(live_share_id);
```

**File**: `core/migrations/YYYY-MM-DD-HHMMSS_create_live_shares/down.sql`

```sql
DROP TABLE IF EXISTS investor_interest;
DROP TABLE IF EXISTS share_views;
DROP TABLE IF EXISTS live_shares;
```

**Run migration**:
```bash
# Development
cargo run --bin migrations up

# Production
docker exec -it reflow-backend cargo run --bin migrations up
```

---

### 2. Short ID Generator Utility

**File**: `core/src/utils/short_id.rs` (new file)

```rust
use rand::Rng;

const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
const SHORT_ID_LENGTH: usize = 10;

/// Generate a random URL-safe short ID (e.g., "x7k2m9p4q1")
pub fn generate_short_id() -> String {
    let mut rng = rand::thread_rng();
    (0..SHORT_ID_LENGTH)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// Check if short ID already exists in database (collision detection)
pub async fn ensure_unique_short_id() -> Result<String, Box<dyn std::error::Error>> {
    use crate::utils::clients::get_pg_client;
    
    loop {
        let short_id = generate_short_id();
        let client = get_pg_client().await?;
        let exists = client
            .query_opt(
                "SELECT 1 FROM live_shares WHERE short_id = $1",
                &[&short_id],
            )
            .await?;
        
        if exists.is_none() {
            return Ok(short_id);
        }
        // Collision detected, try again (very rare)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_id_length() {
        let id = generate_short_id();
        assert_eq!(id.len(), SHORT_ID_LENGTH);
    }

    #[test]
    fn test_short_id_charset() {
        let id = generate_short_id();
        assert!(id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()));
    }

    #[test]
    fn test_short_id_uniqueness() {
        let id1 = generate_short_id();
        let id2 = generate_short_id();
        // Very unlikely to be equal
        assert_ne!(id1, id2);
    }
}
```

**File**: `core/src/utils/mod.rs`

Add module:
```rust
pub mod short_id;
```

---

### 3. Backend Models

**File**: `core/src/models/live_share.rs` (new file)

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::error::Error;
use tokio_postgres::Row;

#[derive(Debug, Serialize, Deserialize)]
pub struct LiveShare {
    pub id: String,
    pub deal_id: String,
    pub user_id: String,
    pub short_id: String,
    pub expires_at: DateTime<Utc>,
    pub view_count: i32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct CreateLiveShareRequest {
    pub deal_id: String,
    pub expires_in_days: i32, // 7 or 30
}

#[derive(Debug, Serialize)]
pub struct LiveShareResponse {
    pub id: String,
    pub deal_id: String,
    pub short_id: String,
    pub share_url: String, // Full URL: http://localhost:5173/share/{short_id}
    pub expires_at: DateTime<Utc>,
    pub view_count: i32,
    pub created_at: DateTime<Utc>,
    pub is_expired: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ShareView {
    pub id: String,
    pub live_share_id: String,
    pub viewed_at: DateTime<Utc>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TrackViewRequest {
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InvestorInterest {
    pub id: String,
    pub live_share_id: String,
    pub name: String,
    pub amount: Option<rust_decimal::Decimal>,
    pub status: InterestStatus,
    pub notes: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "PascalCase")]
pub enum InterestStatus {
    Interested,
    Maybe,
    Passed,
}

impl std::fmt::Display for InterestStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            InterestStatus::Interested => write!(f, "Interested"),
            InterestStatus::Maybe => write!(f, "Maybe"),
            InterestStatus::Passed => write!(f, "Passed"),
        }
    }
}

impl std::str::FromStr for InterestStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Interested" => Ok(InterestStatus::Interested),
            "Maybe" => Ok(InterestStatus::Maybe),
            "Passed" => Ok(InterestStatus::Passed),
            _ => Err(format!("Invalid status: {}", s)),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateInterestRequest {
    pub name: String,
    pub amount: Option<rust_decimal::Decimal>,
    pub status: InterestStatus,
    pub notes: Option<String>,
}

impl LiveShare {
    pub fn from_row(row: &Row) -> Result<Self, Box<dyn Error>> {
        Ok(LiveShare {
            id: row.get("id"),
            deal_id: row.get("deal_id"),
            user_id: row.get("user_id"),
            short_id: row.get("short_id"),
            expires_at: row.get("expires_at"),
            view_count: row.get("view_count"),
            created_at: row.get("created_at"),
        })
    }

    pub fn to_response(&self, base_url: &str) -> LiveShareResponse {
        let is_expired = Utc::now() > self.expires_at;
        LiveShareResponse {
            id: self.id.clone(),
            deal_id: self.deal_id.clone(),
            short_id: self.short_id.clone(),
            share_url: format!("{}/share/{}", base_url, self.short_id),
            expires_at: self.expires_at,
            view_count: self.view_count,
            created_at: self.created_at,
            is_expired,
        }
    }
}

impl ShareView {
    pub fn from_row(row: &Row) -> Result<Self, Box<dyn Error>> {
        Ok(ShareView {
            id: row.get("id"),
            live_share_id: row.get("live_share_id"),
            viewed_at: row.get("viewed_at"),
            ip_address: row.get("ip_address"),
            user_agent: row.get("user_agent"),
        })
    }
}

impl InvestorInterest {
    pub fn from_row(row: &Row) -> Result<Self, Box<dyn Error>> {
        use std::str::FromStr;
        Ok(InvestorInterest {
            id: row.get("id"),
            live_share_id: row.get("live_share_id"),
            name: row.get("name"),
            amount: row.get("amount"),
            status: InterestStatus::from_str(&row.get::<_, String>("status"))?,
            notes: row.get("notes"),
            created_at: row.get("created_at"),
        })
    }
}
```

**File**: `core/src/models/mod.rs`

Add module:
```rust
pub mod live_share;
```

---

### 4. Backend API Routes

**File**: `core/src/routes/live_share.rs` (new file)

```rust
use actix_web::{delete, get, post, web, HttpRequest, HttpResponse, Result};
use chrono::{Duration, Utc};
use serde_json::json;
use uuid::Uuid;

use crate::models::live_share::{
    CreateInterestRequest, CreateLiveShareRequest, InvestorInterest, LiveShare, ShareView,
    TrackViewRequest,
};
use crate::utils::clients::get_pg_client;
use crate::utils::short_id::ensure_unique_short_id;

const BASE_URL: &str = "http://localhost:5173"; // TODO: Make this configurable

// ============================================================================
// AUTHENTICATED ROUTES (require user_id)
// ============================================================================

#[post("/api/v1/live-shares")]
async fn create_live_share(
    req: HttpRequest,
    payload: web::Json<CreateLiveShareRequest>,
) -> Result<HttpResponse> {
    let user_id = req
        .extensions()
        .get::<String>()
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Unauthorized"))?
        .clone();

    let client = get_pg_client().await.map_err(actix_web::error::ErrorInternalServerError)?;

    // Verify deal belongs to user
    let deal_check = client
        .query_opt(
            "SELECT id FROM deals WHERE id = $1 AND user_id = $2",
            &[&payload.deal_id, &user_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    if deal_check.is_none() {
        return Err(actix_web::error::ErrorNotFound("Deal not found"));
    }

    let id = Uuid::new_v4().to_string();
    let short_id = ensure_unique_short_id()
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;
    let expires_at = Utc::now() + Duration::days(payload.expires_in_days as i64);

    client
        .execute(
            "INSERT INTO live_shares (id, deal_id, user_id, short_id, expires_at, view_count, created_at)
             VALUES ($1, $2, $3, $4, $5, 0, NOW())",
            &[&id, &payload.deal_id, &user_id, &short_id, &expires_at],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let row = client
        .query_one("SELECT * FROM live_shares WHERE id = $1", &[&id])
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let live_share = LiveShare::from_row(&row)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(live_share.to_response(BASE_URL)))
}

#[get("/api/v1/live-shares")]
async fn list_live_shares(req: HttpRequest) -> Result<HttpResponse> {
    let user_id = req
        .extensions()
        .get::<String>()
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Unauthorized"))?
        .clone();

    let client = get_pg_client().await.map_err(actix_web::error::ErrorInternalServerError)?;

    let rows = client
        .query(
            "SELECT * FROM live_shares WHERE user_id = $1 ORDER BY created_at DESC",
            &[&user_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let shares: Vec<_> = rows
        .iter()
        .filter_map(|row| LiveShare::from_row(row).ok())
        .map(|share| share.to_response(BASE_URL))
        .collect();

    Ok(HttpResponse::Ok().json(shares))
}

#[get("/api/v1/live-shares/{id}")]
async fn get_live_share(req: HttpRequest, path: web::Path<String>) -> Result<HttpResponse> {
    let user_id = req
        .extensions()
        .get::<String>()
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Unauthorized"))?
        .clone();
    let share_id = path.into_inner();

    let client = get_pg_client().await.map_err(actix_web::error::ErrorInternalServerError)?;

    let row = client
        .query_opt(
            "SELECT * FROM live_shares WHERE id = $1 AND user_id = $2",
            &[&share_id, &user_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("Live share not found"))?;

    let live_share = LiveShare::from_row(&row)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(live_share.to_response(BASE_URL)))
}

#[get("/api/v1/live-shares/{id}/views")]
async fn get_share_views(req: HttpRequest, path: web::Path<String>) -> Result<HttpResponse> {
    let user_id = req
        .extensions()
        .get::<String>()
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Unauthorized"))?
        .clone();
    let share_id = path.into_inner();

    let client = get_pg_client().await.map_err(actix_web::error::ErrorInternalServerError)?;

    // Verify ownership
    let _ownership_check = client
        .query_opt(
            "SELECT 1 FROM live_shares WHERE id = $1 AND user_id = $2",
            &[&share_id, &user_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("Live share not found"))?;

    let rows = client
        .query(
            "SELECT * FROM share_views WHERE live_share_id = $1 ORDER BY viewed_at DESC",
            &[&share_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let views: Vec<ShareView> = rows
        .iter()
        .filter_map(|row| ShareView::from_row(row).ok())
        .collect();

    Ok(HttpResponse::Ok().json(views))
}

#[post("/api/v1/live-shares/{id}/interest")]
async fn add_investor_interest(
    req: HttpRequest,
    path: web::Path<String>,
    payload: web::Json<CreateInterestRequest>,
) -> Result<HttpResponse> {
    let user_id = req
        .extensions()
        .get::<String>()
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Unauthorized"))?
        .clone();
    let share_id = path.into_inner();

    // Validation: amount required for Interested/Maybe
    if matches!(payload.status, crate::models::live_share::InterestStatus::Interested | crate::models::live_share::InterestStatus::Maybe)
        && payload.amount.is_none()
    {
        return Err(actix_web::error::ErrorBadRequest(
            "Amount required for Interested/Maybe status",
        ));
    }

    let client = get_pg_client().await.map_err(actix_web::error::ErrorInternalServerError)?;

    // Verify ownership
    let _ownership_check = client
        .query_opt(
            "SELECT 1 FROM live_shares WHERE id = $1 AND user_id = $2",
            &[&share_id, &user_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("Live share not found"))?;

    let id = Uuid::new_v4().to_string();

    client
        .execute(
            "INSERT INTO investor_interest (id, live_share_id, name, amount, status, notes, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, NOW())",
            &[
                &id,
                &share_id,
                &payload.name,
                &payload.amount,
                &payload.status.to_string(),
                &payload.notes,
            ],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let row = client
        .query_one("SELECT * FROM investor_interest WHERE id = $1", &[&id])
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let interest = InvestorInterest::from_row(&row)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(interest))
}

#[get("/api/v1/live-shares/{id}/interest")]
async fn get_investor_interest(req: HttpRequest, path: web::Path<String>) -> Result<HttpResponse> {
    let user_id = req
        .extensions()
        .get::<String>()
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Unauthorized"))?
        .clone();
    let share_id = path.into_inner();

    let client = get_pg_client().await.map_err(actix_web::error::ErrorInternalServerError)?;

    // Verify ownership
    let _ownership_check = client
        .query_opt(
            "SELECT 1 FROM live_shares WHERE id = $1 AND user_id = $2",
            &[&share_id, &user_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("Live share not found"))?;

    let rows = client
        .query(
            "SELECT * FROM investor_interest WHERE live_share_id = $1 ORDER BY created_at DESC",
            &[&share_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let interests: Vec<InvestorInterest> = rows
        .iter()
        .filter_map(|row| InvestorInterest::from_row(row).ok())
        .collect();

    Ok(HttpResponse::Ok().json(interests))
}

#[delete("/api/v1/live-shares/{id}/interest/{interest_id}")]
async fn delete_investor_interest(
    req: HttpRequest,
    path: web::Path<(String, String)>,
) -> Result<HttpResponse> {
    let user_id = req
        .extensions()
        .get::<String>()
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Unauthorized"))?
        .clone();
    let (share_id, interest_id) = path.into_inner();

    let client = get_pg_client().await.map_err(actix_web::error::ErrorInternalServerError)?;

    // Verify ownership via live_share
    let _ownership_check = client
        .query_opt(
            "SELECT 1 FROM live_shares WHERE id = $1 AND user_id = $2",
            &[&share_id, &user_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("Live share not found"))?;

    let rows_affected = client
        .execute(
            "DELETE FROM investor_interest WHERE id = $1 AND live_share_id = $2",
            &[&interest_id, &share_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    if rows_affected == 0 {
        return Err(actix_web::error::ErrorNotFound("Interest entry not found"));
    }

    Ok(HttpResponse::Ok().json(json!({"message": "Interest entry deleted"})))
}

// ============================================================================
// PUBLIC ROUTES (no authentication required)
// ============================================================================

#[get("/api/v1/public/share/{short_id}")]
async fn get_public_share(path: web::Path<String>) -> Result<HttpResponse> {
    let short_id = path.into_inner();

    let client = get_pg_client().await.map_err(actix_web::error::ErrorInternalServerError)?;

    let row = client
        .query_opt(
            "SELECT * FROM live_shares WHERE short_id = $1",
            &[&short_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("Share not found"))?;

    let live_share = LiveShare::from_row(&row)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    // Check if expired
    if Utc::now() > live_share.expires_at {
        return Err(actix_web::error::ErrorGone("Share has expired"));
    }

    // TODO: Fetch associated deal data
    Ok(HttpResponse::Ok().json(json!({
        "share": live_share.to_response(BASE_URL),
        "deal": {} // Fetch deal data here
    })))
}

#[post("/api/v1/public/share/{short_id}/view")]
async fn track_public_view(
    path: web::Path<String>,
    payload: web::Json<TrackViewRequest>,
) -> Result<HttpResponse> {
    let short_id = path.into_inner();

    let client = get_pg_client().await.map_err(actix_web::error::ErrorInternalServerError)?;

    // Get live_share_id from short_id
    let row = client
        .query_opt(
            "SELECT id FROM live_shares WHERE short_id = $1",
            &[&short_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("Share not found"))?;

    let live_share_id: String = row.get("id");

    // Insert view record
    let view_id = Uuid::new_v4().to_string();
    client
        .execute(
            "INSERT INTO share_views (id, live_share_id, viewed_at, ip_address, user_agent)
             VALUES ($1, $2, NOW(), $3, $4)",
            &[
                &view_id,
                &live_share_id,
                &payload.ip_address,
                &payload.user_agent,
            ],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    // Increment view count
    client
        .execute(
            "UPDATE live_shares SET view_count = view_count + 1 WHERE id = $1",
            &[&live_share_id],
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(json!({"message": "View tracked"})))
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(create_live_share)
        .service(list_live_shares)
        .service(get_live_share)
        .service(get_share_views)
        .service(add_investor_interest)
        .service(get_investor_interest)
        .service(delete_investor_interest)
        .service(get_public_share)
        .service(track_public_view);
}
```

**File**: `core/src/routes/mod.rs`

Add module:
```rust
pub mod live_share;
```

**File**: `core/src/lib.rs`

Register routes in the configure function:
```rust
use crate::routes::live_share;

// In configure() function:
live_share::configure(cfg);
```

---

## Testing

### Manual Testing

```bash
# 1. Run migration
cd core
sqlx migrate run

# 2. Start backend
cargo run

# 3. Test create live share
curl -X POST http://localhost:8000/api/v1/live-shares \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"deal_id": "test-deal-id", "expires_in_days": 7}'

# 4. Test public share
curl http://localhost:8000/api/v1/public/share/abc123xyz

# 5. Test track view
curl -X POST http://localhost:8000/api/v1/public/share/abc123xyz/view \
  -H "Content-Type: application/json" \
  -d '{"ip_address": "192.168.1.1", "user_agent": "Mozilla/5.0"}'
```

### Unit Tests

Add tests to `core/src/utils/short_id.rs` (already included above).

---

## Success Criteria

- ✅ Database migrations run without errors
- ✅ Short IDs are unique and URL-safe
- ✅ All API endpoints return correct status codes
- ✅ View tracking increments count atomically
- ✅ Expired shares return 410 Gone status
- ✅ Foreign key constraints enforce data integrity
- ✅ Public endpoints work without authentication

## Security Considerations

- Short IDs should be unpredictable (use cryptographically secure random)
- Rate limit public view tracking to prevent abuse
- Sanitize IP addresses and user agents before storage
- Validate expiration dates (max 90 days)
- Ensure cascade deletes work correctly
