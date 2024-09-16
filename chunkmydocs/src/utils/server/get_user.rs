use crate::models::server::user::{InvoiceStatus, Tier, User};
use crate::utils::db::deadpool_postgres::{Client, Pool};
use std::str::FromStr;

pub async fn get_user(user_id: String, pool: &Pool) -> Result<User, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    let query = r#"
    SELECT 
        u.user_id,
        u.customer_id,
        u.email,
        u.first_name,
        u.last_name,
        array_agg(DISTINCT ak.key) as api_keys,
        u.tier,
        u.created_at,
        u.updated_at,
        u.invoice_status,
        json_agg(
            json_build_object(
                'usage', COALESCE(us.usage, 0),
                'usage_limit', COALESCE(us.usage_limit, 0),
                'usage_type', us.usage_type,
                'unit', us.unit,
                'created_at', us.created_at,
                'updated_at', us.created_at
            )
        )::text AS usages
    FROM 
        users u
    LEFT JOIN 
        USAGE us ON u.user_id = us.user_id
    LEFT JOIN
        API_KEYS ak ON u.user_id = ak.user_id
    WHERE 
        u.user_id = $1
    GROUP BY 
        u.user_id;
    "#;

    let row = client.query_opt(query, &[&user_id]).await?.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("User with id {} not found", user_id),
        )
    })?;

    let user = User {
        user_id: row.get("user_id"),
        customer_id: row.get("customer_id"),
        email: row.get("email"),
        first_name: row.get("first_name"),
        last_name: row.get("last_name"),
        api_keys: row.get("api_keys"),
        tier: row
            .get::<_, Option<String>>("tier")
            .and_then(|t| Tier::from_str(&t).ok())
            .unwrap_or(Tier::Free),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
        usages: serde_json::from_str(&row.get::<_, String>("usages")).unwrap_or_default(),
        invoice_status: Some(
            row.get::<_, Option<String>>("invoice_status")
                .and_then(|s| InvoiceStatus::from_str(&s).ok())
                .unwrap_or(InvoiceStatus::NoInvoice),
        ),
    };

    Ok(user)
}
